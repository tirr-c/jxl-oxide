use std::{
    collections::HashMap,
    io::Read,
    sync::Arc,
};

use jxl_bitstream::{Bitstream, Bundle};
use jxl_frame::{
    filter::Gabor,
    header::{Encoding, FrameType},
    Frame, data::{LfGlobal, HfGlobal, decode_pass_group, LfGroup, GlobalModular},
};
use jxl_grid::{SimpleGrid, CutGrid};
use jxl_image::{ImageHeader, ImageMetadata};
use jxl_modular::ChannelShift;

use crate::{
    blend,
    cut_grid,
    features,
    filter,
    vardct,
    IndexedFrame,
    Result,
    Error,
};

#[derive(Debug)]
pub struct ContextInner {
    image_header: Arc<ImageHeader>,
    pub(crate) frames: Vec<IndexedFrame>,
    pub(crate) keyframes: Vec<usize>,
    pub(crate) keyframe_in_progress: Option<usize>,
    pub(crate) refcounts: Vec<usize>,
    pub(crate) frame_deps: Vec<FrameDependence>,
    pub(crate) lf_frame: [usize; 4],
    pub(crate) reference: [usize; 4],
    pub(crate) loading_frame: Option<IndexedFrame>,
}

impl ContextInner {
    pub fn new(image_header: Arc<ImageHeader>) -> Self {
        Self {
            image_header,
            frames: Vec::new(),
            keyframes: Vec::new(),
            keyframe_in_progress: None,
            refcounts: Vec::new(),
            frame_deps: Vec::new(),
            lf_frame: [usize::MAX; 4],
            reference: [usize::MAX; 4],
            loading_frame: None,
        }
    }
}

impl ContextInner {
    #[inline]
    pub fn width(&self) -> u32 {
        self.image_header.size.width
    }

    #[inline]
    pub fn height(&self) -> u32 {
        self.image_header.size.height
    }

    #[inline]
    pub fn metadata(&self) -> &ImageMetadata {
        &self.image_header.metadata
    }

    #[inline]
    pub fn xyb_encoded(&self) -> bool {
        self.image_header.metadata.xyb_encoded
    }

    #[inline]
    pub fn loaded_keyframes(&self) -> usize {
        self.keyframes.len() + (self.keyframe_in_progress.is_some() as usize)
    }

    pub fn keyframe(&self, keyframe_idx: usize) -> Option<&IndexedFrame> {
        if keyframe_idx == self.keyframes.len() {
            if let Some(idx) = self.keyframe_in_progress {
                Some(&self.frames[idx])
            } else {
                let Some(frame) = &self.loading_frame else { return None; };
                frame.header().is_keyframe().then_some(frame)
            }
        } else if let Some(&idx) = self.keyframes.get(keyframe_idx) {
            Some(&self.frames[idx])
        } else {
            None
        }
    }

    pub fn preserve_current_frame(&mut self) {
        let Some(frame) = self.loading_frame.take() else { return; };

        let header = frame.header();
        let idx = self.frames.len();
        let is_last = header.is_last;

        self.refcounts.push(0);

        let lf = if header.flags.use_lf_frame() {
            let lf = self.lf_frame[header.lf_level as usize];
            self.refcounts[lf] += 1;
            lf
        } else {
            usize::MAX
        };
        for ref_idx in self.reference {
            if ref_idx != usize::MAX {
                self.refcounts[ref_idx] += 1;
            }
        }

        let deps = FrameDependence {
            lf,
            ref_slots: self.reference,
        };

        if !is_last && (header.duration == 0 || header.save_as_reference != 0) && header.frame_type != FrameType::LfFrame {
            let ref_idx = header.save_as_reference as usize;
            self.reference[ref_idx] = idx;
        }
        if header.lf_level != 0 {
            let lf_idx = header.lf_level as usize - 1;
            self.lf_frame[lf_idx] = idx;
        }

        if header.is_keyframe() {
            self.refcounts[idx] += 1;
            self.keyframes.push(idx);
            self.keyframe_in_progress = None;
        } else if header.frame_type.is_normal_frame() {
            self.keyframe_in_progress = Some(idx);
        }

        self.frames.push(frame);
        self.frame_deps.push(deps);
    }
}

impl ContextInner {
    pub fn load_single<R: Read>(
        &mut self,
        bitstream: &mut Bitstream<R>,
    ) -> Result<&IndexedFrame> {
        let image_header = &self.image_header;

        let frame = match &mut self.loading_frame {
            Some(frame) => frame,
            slot => {
                let mut bitstream = bitstream.rewindable();
                let frame = Frame::parse(&mut bitstream, image_header.clone())?;
                bitstream.commit();
                *slot = Some(IndexedFrame::new(frame, self.frames.len()));
                slot.as_mut().unwrap()
            },
        };

        let header = frame.header();
        tracing::debug!(
            width = header.color_sample_width(),
            height = header.color_sample_height(),
            frame_type = format_args!("{:?}", header.frame_type),
            encoding = format_args!("{:?}", header.encoding),
            jpeg_upsampling = format_args!("{:?}", header.do_ycbcr.then_some(header.jpeg_upsampling)),
            upsampling = header.upsampling,
            lf_level = header.lf_level,
            "Decoding {}x{} frame", header.color_sample_width(), header.color_sample_height()
        );

        // Check if LF frame exists
        if header.flags.use_lf_frame() && self.lf_frame[header.lf_level as usize] == usize::MAX {
            return Err(Error::UninitializedLfFrame(header.lf_level));
        }

        frame.read_all(bitstream)?;
        Ok(frame)
    }
}

impl ContextInner {
    pub fn render_frame<'a>(
        &'a self,
        frame: &'a IndexedFrame,
        reference_frames: ReferenceFrames<'a>,
        cache: &mut RenderCache,
        mut region: Option<(u32, u32, u32, u32)>,
    ) -> Result<Vec<SimpleGrid<f32>>> {
        let frame_header = frame.header();
        if let Some(region) = &mut region {
            frame.adjust_region(region);
        }

        let (mut fb, gmodular) = match frame_header.encoding {
            Encoding::Modular => self.render_modular(frame, cache, region),
            Encoding::VarDct => self.render_vardct(frame, reference_frames.lf, cache, region),
        }?;

        let [a, b, c] = &mut fb;
        if frame.header().do_ycbcr {
            filter::apply_jpeg_upsampling([a, b, c], frame_header.jpeg_upsampling);
        }
        if let Gabor::Enabled(weights) = frame_header.restoration_filter.gab {
            filter::apply_gabor_like([a, b, c], weights);
        }
        filter::apply_epf([a, b, c], &cache.lf_groups, frame_header);

        let [a, b, c] = fb;
        let mut ret = vec![a, b, c];
        self.append_extra_channels(frame, &mut ret, gmodular);

        self.render_features(frame, &mut ret, reference_frames.refs, cache)?;

        if !frame_header.save_before_ct {
            if frame_header.do_ycbcr {
                let [cb, y, cr, ..] = &mut *ret else { panic!() };
                jxl_color::ycbcr_to_rgb([cb, y, cr]);
            }
            self.convert_color(&mut ret);
        }

        Ok(if !frame_header.frame_type.is_normal_frame() {
            ret
        } else if frame_header.resets_canvas {
            let mut cropped = Vec::with_capacity(ret.len());
            let l = (-frame_header.x0) as usize;
            let t = (-frame_header.y0) as usize;
            let w = self.width() as usize;
            let h = self.height() as usize;
            for g in ret {
                if g.width() == w && g.height() == h {
                    cropped.push(g);
                    continue;
                }

                let mut new_grid = SimpleGrid::new(w, h);
                for (idx, v) in new_grid.buf_mut().iter_mut().enumerate() {
                    let y = idx / w;
                    let x = idx % w;
                    *v = *g.get(x + l, y + t).unwrap();
                }
                cropped.push(new_grid);
            }
            cropped
        } else {
            blend::blend(&self.image_header, reference_frames.refs, frame, &ret)
        })
    }

    fn append_extra_channels<'a>(
        &'a self,
        frame: &'a IndexedFrame,
        fb: &mut Vec<SimpleGrid<f32>>,
        gmodular: GlobalModular,
    ) {
        tracing::debug!("Attaching extra channels");

        let extra_channel_from = gmodular.extra_channel_from();
        let gmodular = &gmodular.modular;

        let channel_data = &gmodular.image().channel_data()[extra_channel_from..];

        let width = frame.header().color_sample_width() as usize;
        let height = frame.header().color_sample_height() as usize;

        for (g, ec_info) in channel_data.iter().zip(&self.image_header.metadata.ec_info) {
            let bit_depth = ec_info.bit_depth;

            let mut out = SimpleGrid::new(width, height);
            let buffer = out.buf_mut();

            let (gw, gh) = g.group_dim();
            let group_stride = g.groups_per_row();
            for (group_idx, g) in g.groups() {
                let base_x = (group_idx % group_stride) * gw;
                let base_y = (group_idx / group_stride) * gh;
                for (idx, &s) in g.buf().iter().enumerate() {
                    let y = base_y + idx / g.width();
                    if y >= height {
                        break;
                    }
                    let x = base_x + idx % g.width();
                    if x >= width {
                        continue;
                    }

                    buffer[y * width + x] = bit_depth.parse_integer_sample(s);
                }
            }

            fb.push(out);
        }
    }

    fn render_features<'a>(
        &'a self,
        frame: &'a IndexedFrame,
        grid: &mut [SimpleGrid<f32>],
        reference_grids: [Option<&[SimpleGrid<f32>]>; 4],
        cache: &mut RenderCache,
    ) -> Result<()> {
        let frame_header = frame.header();
        let lf_global = cache.lf_global.as_ref().unwrap();
        let base_correlations_xb = lf_global.vardct.as_ref().map(|x| {
            (
                x.lf_chan_corr.base_correlation_x,
                x.lf_chan_corr.base_correlation_b,
            )
        });

        for (idx, g) in grid.iter_mut().enumerate() {
            features::upsample(g, &self.image_header, frame_header, idx);
        }

        if let Some(patches) = &lf_global.patches {
            for patch in &patches.patches {
                let Some(ref_grid) = reference_grids[patch.ref_idx as usize] else {
                    return Err(Error::InvalidReference(patch.ref_idx));
                };
                blend::patch(&self.image_header, grid, ref_grid, patch);
            }
        }

        if let Some(splines) = &lf_global.splines {
            features::render_spline(frame_header, grid, splines, base_correlations_xb)?;
        }
        if let Some(noise) = &lf_global.noise {
            let (visible_frames_num, invisible_frames_num) =
                self.get_previous_frames_visibility(frame);

            features::render_noise(
                frame.header(),
                visible_frames_num,
                invisible_frames_num,
                base_correlations_xb,
                grid,
                noise,
            )?;
        }

        Ok(())
    }

    pub fn convert_color(&self, grid: &mut [SimpleGrid<f32>]) {
        let metadata = &self.image_header.metadata;
        if metadata.xyb_encoded {
            let [x, y, b, ..] = grid else { panic!() };
            jxl_color::xyb_to_linear_srgb(
                [x, y, b],
                &metadata.opsin_inverse_matrix,
                metadata.tone_mapping.intensity_target,
            );

            if metadata.colour_encoding.want_icc {
                // Don't convert tf, return linear sRGB as is
                return;
            }

            jxl_color::from_linear_srgb(
                grid,
                &metadata.colour_encoding,
                metadata.tone_mapping.intensity_target,
            );
        }
    }

    fn get_previous_frames_visibility<'a>(&'a self, frame: &'a IndexedFrame) -> (usize, usize) {
        let frame_idx = frame.index();
        let (is_keyframe, keyframe_idx) = match self.keyframes.binary_search(&frame_idx) {
            Ok(val) => (true, val),
            Err(val) => (false, val),
        };
        let prev_keyframes = &self.keyframes[..keyframe_idx];

        let visible_frames_num = keyframe_idx + is_keyframe as usize;

        let invisible_frames_num = if is_keyframe {
            0
        } else if prev_keyframes.is_empty() {
            1 + frame_idx
        } else {
            let last_visible_frame = prev_keyframes[keyframe_idx];
            frame_idx - last_visible_frame
        };

        (visible_frames_num, invisible_frames_num)
    }

    fn render_modular<'a>(
        &'a self,
        frame: &'a IndexedFrame,
        cache: &mut RenderCache,
        _region: Option<(u32, u32, u32, u32)>,
    ) -> Result<([SimpleGrid<f32>; 3], GlobalModular)> {
        let metadata = self.metadata();
        let xyb_encoded = self.xyb_encoded();
        let frame_header = frame.header();

        let lf_global = if let Some(x) = &cache.lf_global {
            x
        } else {
            let lf_global = frame.try_parse_lf_global().ok_or(Error::IncompleteFrame)??;
            cache.lf_global = Some(lf_global);
            cache.lf_global.as_ref().unwrap()
        };
        let mut gmodular = lf_global.gmodular.clone();

        let jpeg_upsampling = frame_header.jpeg_upsampling;
        let shifts_cbycr = [0, 1, 2].map(|idx| {
            ChannelShift::from_jpeg_upsampling(jpeg_upsampling, idx)
        });
        let channels = metadata.encoded_color_channels();

        let width = frame_header.color_sample_width() as usize;
        let height = frame_header.color_sample_height() as usize;
        let bit_depth = metadata.bit_depth;
        let mut fb_xyb = [
            SimpleGrid::new(width, height),
            SimpleGrid::new(width, height),
            SimpleGrid::new(width, height),
        ];

        let lf_groups = &mut cache.lf_groups;
        for idx in 0..frame_header.num_lf_groups() {
            let lf_group = lf_groups.entry(idx);
            let lf_group = match lf_group {
                std::collections::hash_map::Entry::Occupied(x) => x.into_mut(),
                std::collections::hash_map::Entry::Vacant(x) => {
                    let Some(lf_group) = frame.try_parse_lf_group(Some(lf_global), idx).transpose()? else { continue; };
                    &*x.insert(lf_group)
                },
            };
            gmodular.modular.copy_from_modular(lf_group.mlf_group.clone());
        }

        for pass_idx in 0..frame_header.passes.num_passes {
            for group_idx in 0..frame_header.num_groups() {
                let lf_group_idx = frame_header.lf_group_idx_from_group_idx(group_idx);
                let Some(lf_group) = lf_groups.get(&lf_group_idx) else { continue; };
                let Some(mut bitstream) = frame.pass_group_bitstream(pass_idx, group_idx).transpose()? else { continue; };

                let shift = frame.pass_shifts(pass_idx);
                decode_pass_group(
                    &mut bitstream,
                    frame_header,
                    None,
                    lf_group,
                    None,
                    pass_idx,
                    group_idx,
                    shift,
                    &mut gmodular,
                    None,
                )?;
            }
        }

        gmodular.modular.inverse_transform();
        let channel_data = gmodular.modular.image().channel_data();

        for ((g, shift), buffer) in channel_data.iter().zip(shifts_cbycr).zip(fb_xyb.iter_mut()) {
            let buffer = buffer.buf_mut();
            let (gw, gh) = g.group_dim();
            let group_stride = g.groups_per_row();
            for (group_idx, g) in g.groups() {
                let base_x = (group_idx % group_stride) * gw;
                let base_y = (group_idx / group_stride) * gh;
                for (idx, &s) in g.buf().iter().enumerate() {
                    let y = base_y + idx / g.width();
                    let y = y << shift.vshift();
                    if y >= height {
                        break;
                    }
                    let x = base_x + idx % g.width();
                    let x = x << shift.hshift();
                    if x >= width {
                        continue;
                    }

                    buffer[y * width + x] = if xyb_encoded {
                        s as f32
                    } else {
                        bit_depth.parse_integer_sample(s)
                    };
                }
            }
        }

        if channels == 1 {
            fb_xyb[1] = fb_xyb[0].clone();
            fb_xyb[2] = fb_xyb[0].clone();
        }
        if xyb_encoded {
            // Make Y'X'B' to X'Y'B'
            fb_xyb.swap(0, 1);
            let [x, y, b] = &mut fb_xyb;
            let x = x.buf_mut();
            let y = y.buf_mut();
            let b = b.buf_mut();
            for ((x, y), b) in x.iter_mut().zip(y).zip(b) {
                *b += *y;
                *x *= lf_global.lf_dequant.m_x_lf_unscaled();
                *y *= lf_global.lf_dequant.m_y_lf_unscaled();
                *b *= lf_global.lf_dequant.m_b_lf_unscaled();
            }
        }

        Ok((fb_xyb, gmodular))
    }

    fn render_vardct<'a>(
        &'a self,
        frame: &'a IndexedFrame,
        lf_frame: Option<&'a [SimpleGrid<f32>]>,
        cache: &mut RenderCache,
        _region: Option<(u32, u32, u32, u32)>,
    ) -> Result<([SimpleGrid<f32>; 3], GlobalModular)> {
        let span = tracing::span!(tracing::Level::TRACE, "RenderContext::render_vardct");
        let _guard = span.enter();

        let frame_header = frame.header();

        let lf_global = if let Some(x) = &cache.lf_global {
            x
        } else {
            let lf_global = frame.try_parse_lf_global().ok_or(Error::IncompleteFrame)??;
            cache.lf_global = Some(lf_global);
            cache.lf_global.as_ref().unwrap()
        };
        let mut gmodular = lf_global.gmodular.clone();
        let lf_global_vardct = lf_global.vardct.as_ref().unwrap();

        let hf_global = if let Some(x) = &cache.hf_global {
            x
        } else {
            let hf_global = frame.try_parse_hf_global(Some(lf_global)).ok_or(Error::IncompleteFrame)??;
            cache.hf_global = Some(hf_global);
            cache.hf_global.as_ref().unwrap()
        };

        let jpeg_upsampling = frame_header.jpeg_upsampling;
        let shifts_cbycr: [_; 3] = std::array::from_fn(|idx| {
            ChannelShift::from_jpeg_upsampling(jpeg_upsampling, idx)
        });
        let subsampled = jpeg_upsampling.into_iter().any(|x| x != 0);

        let width = frame_header.color_sample_width() as usize;
        let height = frame_header.color_sample_height() as usize;
        let (width_rounded, height_rounded) = {
            let mut bw = (width + 7) / 8;
            let mut bh = (height + 7) / 8;
            let h_upsample = jpeg_upsampling.into_iter().any(|j| j == 1 || j == 2);
            let v_upsample = jpeg_upsampling.into_iter().any(|j| j == 1 || j == 3);
            if h_upsample {
                bw = (bw + 1) / 2 * 2;
            }
            if v_upsample {
                bh = (bh + 1) / 2 * 2;
            }
            (bw * 8, bh * 8)
        };
        let mut fb_xyb = [
            SimpleGrid::new(width_rounded, height_rounded),
            SimpleGrid::new(width_rounded, height_rounded),
            SimpleGrid::new(width_rounded, height_rounded),
        ];

        let lf_groups = &mut cache.lf_groups;
        for idx in 0..frame_header.num_lf_groups() {
            let lf_group = lf_groups.entry(idx);
            let lf_group = match lf_group {
                std::collections::hash_map::Entry::Occupied(x) => x.into_mut(),
                std::collections::hash_map::Entry::Vacant(x) => {
                    let Some(lf_group) = frame.try_parse_lf_group(Some(lf_global), idx).transpose()? else { continue; };
                    &*x.insert(lf_group)
                },
            };
            gmodular.modular.copy_from_modular(lf_group.mlf_group.clone());
        }

        let mut lf_xyb_buf;
        let lf_xyb;
        if let Some(x) = lf_frame {
            lf_xyb = x;
        } else {
            lf_xyb_buf = [
                SimpleGrid::new(width_rounded / 8, height_rounded / 8),
                SimpleGrid::new(width_rounded / 8, height_rounded / 8),
                SimpleGrid::new(width_rounded / 8, height_rounded / 8),
            ];
            for idx in 0..frame_header.num_lf_groups() {
                let Some(lf_group) = lf_groups.get(&idx) else { continue; };

                let lf_group_x = idx % frame_header.lf_groups_per_row();
                let lf_group_y = idx / frame_header.lf_groups_per_row();
                let left = lf_group_x * frame_header.group_dim();
                let top = lf_group_y * frame_header.group_dim();

                let lf_coeff = lf_group.lf_coeff.as_ref().unwrap();
                let channel_data = lf_coeff.lf_quant.image().channel_data();

                let [lf_x, lf_y, lf_b] = &mut lf_xyb_buf;
                let lf_x = cut_grid::make_quant_cut_grid(lf_x, left as usize, top as usize, shifts_cbycr[0], &channel_data[1]);
                let lf_y = cut_grid::make_quant_cut_grid(lf_y, left as usize, top as usize, shifts_cbycr[1], &channel_data[0]);
                let lf_b = cut_grid::make_quant_cut_grid(lf_b, left as usize, top as usize, shifts_cbycr[2], &channel_data[2]);
                let mut lf = [lf_x, lf_y, lf_b];

                vardct::dequant_lf(
                    &mut lf,
                    &lf_global.lf_dequant,
                    &lf_global_vardct.quantizer,
                    lf_coeff.extra_precision,
                );
                if !subsampled {
                    vardct::chroma_from_luma_lf(
                        &mut lf,
                        &lf_global_vardct.lf_chan_corr,
                    );
                }
            }

            if !frame_header.flags.skip_adaptive_lf_smoothing() {
                vardct::adaptive_lf_smoothing(
                    &mut lf_xyb_buf,
                    &lf_global.lf_dequant,
                    &lf_global_vardct.quantizer,
                );
            }

            lf_xyb = &lf_xyb_buf;
        }

        let group_dim = frame_header.group_dim();
        for pass_idx in 0..frame_header.passes.num_passes {
            for group_idx in 0..frame_header.num_groups() {
                let lf_group_idx = frame_header.lf_group_idx_from_group_idx(group_idx);
                let Some(lf_group) = lf_groups.get(&lf_group_idx) else { continue; };
                let Some(mut bitstream) = frame.pass_group_bitstream(pass_idx, group_idx).transpose()? else { continue; };

                let group_x = group_idx % frame_header.groups_per_row();
                let group_y = group_idx / frame_header.groups_per_row();
                let left = group_x * group_dim;
                let top = group_y * group_dim;
                let group_width = group_dim.min(width_rounded as u32 - left);
                let group_height = group_dim.min(height_rounded as u32 - top);

                let [fb_x, fb_y, fb_b] = &mut fb_xyb;
                let mut grid_xyb = [(0usize, fb_x), (1, fb_y), (2, fb_b)].map(|(idx, fb)| {
                    let hshift = shifts_cbycr[idx].hshift();
                    let vshift = shifts_cbycr[idx].vshift();
                    let group_width = group_width >> hshift;
                    let group_height = group_height >> vshift;
                    let left = left >> hshift;
                    let top = top >> vshift;
                    let offset = top as usize * width_rounded + left as usize;
                    CutGrid::from_buf(&mut fb.buf_mut()[offset..], group_width as usize, group_height as usize, width_rounded)
                });

                let shift = frame.pass_shifts(pass_idx);
                decode_pass_group(
                    &mut bitstream,
                    frame_header,
                    Some(lf_global_vardct),
                    lf_group,
                    Some(hf_global),
                    pass_idx,
                    group_idx,
                    shift,
                    &mut gmodular,
                    Some(&mut grid_xyb),
                )?;
            }
        }

        gmodular.modular.inverse_transform();
        vardct::dequant_hf_varblock(
            &mut fb_xyb,
            &self.image_header,
            frame_header,
            lf_global,
            &*lf_groups,
            hf_global,
        );
        if !subsampled {
            vardct::chroma_from_luma_hf(
                &mut fb_xyb,
                frame_header,
                lf_global,
                &*lf_groups,
            );
        }
        vardct::transform_with_lf(lf_xyb, &mut fb_xyb, frame_header, &*lf_groups);

        let fb = if width == width_rounded && height == width_rounded {
            fb_xyb
        } else {
            fb_xyb.map(|g| {
                let mut new_g = SimpleGrid::new(width, height);
                for (new_row, row) in new_g.buf_mut().chunks_exact_mut(width).zip(g.buf().chunks_exact(width_rounded)) {
                    new_row.copy_from_slice(&row[..width]);
                }
                new_g
            })
        };
        Ok((fb, gmodular))
    }
}

#[derive(Debug, Copy, Clone)]
pub struct FrameDependence {
    pub(crate) lf: usize,
    pub(crate) ref_slots: [usize; 4],
}

impl FrameDependence {
    pub fn indices(&self) -> impl Iterator<Item = usize> + 'static {
        std::iter::once(self.lf).chain(self.ref_slots).filter(|&v| v != usize::MAX)
    }
}

#[derive(Debug, Default)]
pub struct ReferenceFrames<'state> {
    pub(crate) lf: Option<&'state [SimpleGrid<f32>]>,
    pub(crate) refs: [Option<&'state [SimpleGrid<f32>]>; 4],
}

#[derive(Debug)]
pub struct RenderCache {
    lf_global: Option<LfGlobal>,
    hf_global: Option<HfGlobal>,
    lf_groups: HashMap<u32, LfGroup>,
}

impl RenderCache {
    pub fn new(frame: &IndexedFrame) -> Self {
        let frame_header = frame.header();
        let jpeg_upsampling = frame_header.jpeg_upsampling;
        let shifts_cbycr: [_; 3] = std::array::from_fn(|idx| {
            ChannelShift::from_jpeg_upsampling(jpeg_upsampling, idx)
        });

        let lf_width = (frame_header.color_sample_width() + 7) / 8;
        let lf_height = (frame_header.color_sample_height() + 7) / 8;
        let mut whd = [(lf_width, lf_height); 3];
        for ((w, h), shift) in whd.iter_mut().zip(shifts_cbycr) {
            let (shift_w, shift_h) = shift.shift_size((lf_width, lf_height));
            *w = shift_w;
            *h = shift_h;
        }
        Self {
            lf_global: None,
            hf_global: None,
            lf_groups: HashMap::new(),
        }
    }
}
