use std::{
    collections::{HashMap, HashSet},
    io::Read,
};

use jxl_bitstream::{Bitstream, Bundle};
use jxl_frame::{
    filter::Gabor,
    header::{Encoding, FrameType},
    Frame,
    ProgressiveResult,
};
use jxl_grid::SimpleGrid;
use jxl_image::{ImageHeader, ImageMetadata};
use jxl_modular::ChannelShift;
use jxl_vardct::HfCoeff;

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
pub struct ContextInner<'a> {
    image_header: &'a ImageHeader,
    pub(crate) frames: Vec<IndexedFrame<'a>>,
    pub(crate) keyframes: Vec<usize>,
    pub(crate) keyframe_in_progress: Option<usize>,
    pub(crate) refcounts: Vec<usize>,
    pub(crate) frame_deps: Vec<FrameDependence>,
    pub(crate) lf_frame: [usize; 4],
    pub(crate) reference: [usize; 4],
    pub(crate) loading_frame: Option<IndexedFrame<'a>>,
}

impl<'a> ContextInner<'a> {
    pub fn new(image_header: &'a ImageHeader) -> Self {
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

impl<'a> ContextInner<'a> {
    #[inline]
    pub fn width(&self) -> u32 {
        self.image_header.size.width
    }

    #[inline]
    pub fn height(&self) -> u32 {
        self.image_header.size.height
    }

    #[inline]
    pub fn metadata(&self) -> &'a ImageMetadata {
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

    pub fn keyframe(&self, keyframe_idx: usize) -> Option<&IndexedFrame<'a>> {
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

impl ContextInner<'_> {
    pub fn load_cropped_single<R: Read>(
        &mut self,
        bitstream: &mut Bitstream<R>,
        progressive: bool,
        mut region: Option<(u32, u32, u32, u32)>,
    ) -> Result<(ProgressiveResult, &IndexedFrame)> {
        let image_header = self.image_header;

        let frame = match &mut self.loading_frame {
            Some(frame) => frame,
            slot => {
                let mut bitstream = bitstream.rewindable();
                let frame = Frame::parse(&mut bitstream, image_header)?;
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

        if let Some(region) = &mut region {
            frame.adjust_region(region);
        };
        let filter = if region.is_some() {
            Box::new(jxl_frame::crop_filter(region)) as Box<dyn FnMut(&_, &_, _) -> bool>
        } else {
            Box::new(|_: &_, _: &_, _| true)
        };

        let result = if header.frame_type == FrameType::RegularFrame {
            frame.load_with_filter(bitstream, progressive, filter)?
        } else {
            frame.load_all(bitstream)?;
            ProgressiveResult::FrameComplete
        };

        Ok((result, frame))
    }
}

impl<'f> ContextInner<'f> {
    pub fn render_frame<'a>(
        &'a self,
        frame: &'a IndexedFrame<'f>,
        reference_frames: ReferenceFrames<'a>,
        cache: &mut RenderCache,
        mut region: Option<(u32, u32, u32, u32)>,
    ) -> Result<Vec<SimpleGrid<f32>>> {
        let frame_header = frame.header();
        if let Some(region) = &mut region {
            frame.adjust_region(region);
        }

        let mut fb = match frame_header.encoding {
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
        filter::apply_epf([a, b, c], &frame.data().lf_group, frame_header);

        let [a, b, c] = fb;
        let mut ret = vec![a, b, c];
        self.append_extra_channels(frame, &mut ret);

        self.render_features(frame, &mut ret, reference_frames.refs)?;

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
            blend::blend(self.image_header, reference_frames.refs, frame, &ret)
        })
    }

    fn append_extra_channels<'a>(
        &'a self,
        frame: &'a IndexedFrame<'f>,
        fb: &mut Vec<SimpleGrid<f32>>,
    ) {
        tracing::debug!("Attaching extra channels");

        let frame_data = frame.data();
        let lf_global = frame_data.lf_global.as_ref().unwrap();
        let extra_channel_from = lf_global.gmodular.extra_channel_from();
        let gmodular = &lf_global.gmodular.modular;

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
        frame: &'a IndexedFrame<'f>,
        grid: &mut [SimpleGrid<f32>],
        reference_grids: [Option<&[SimpleGrid<f32>]>; 4],
    ) -> Result<()> {
        let frame_data = frame.data();
        let frame_header = frame.header();
        let lf_global = frame_data.lf_global.as_ref().unwrap();
        let base_correlations_xb = lf_global.vardct.as_ref().map(|x| {
            (
                x.lf_chan_corr.base_correlation_x,
                x.lf_chan_corr.base_correlation_b,
            )
        });

        for (idx, g) in grid.iter_mut().enumerate() {
            features::upsample(g, self.image_header, frame_header, idx);
        }

        if let Some(patches) = &lf_global.patches {
            for patch in &patches.patches {
                let Some(ref_grid) = reference_grids[patch.ref_idx as usize] else {
                    return Err(Error::InvalidReference(patch.ref_idx));
                };
                blend::patch(self.image_header, grid, ref_grid, patch);
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
        frame: &'a IndexedFrame<'f>,
        _cache: &mut RenderCache,
        _region: Option<(u32, u32, u32, u32)>,
    ) -> Result<[SimpleGrid<f32>; 3]> {
        let metadata = self.metadata();
        let xyb_encoded = self.xyb_encoded();
        let frame_header = frame.header();
        let frame_data = frame.data();
        let lf_global = frame_data.lf_global.as_ref().ok_or(Error::IncompleteFrame)?;
        let gmodular = &lf_global.gmodular.modular;
        let jpeg_upsampling = frame_header.jpeg_upsampling;
        let shifts_cbycr = [0, 1, 2].map(|idx| {
            ChannelShift::from_jpeg_upsampling(jpeg_upsampling, idx)
        });
        let channels = metadata.encoded_color_channels();

        let channel_data = &gmodular.image().channel_data()[..channels];

        let width = frame_header.color_sample_width() as usize;
        let height = frame_header.color_sample_height() as usize;
        let bit_depth = metadata.bit_depth;
        let mut fb_xyb = [
            SimpleGrid::new(width, height),
            SimpleGrid::new(width, height),
            SimpleGrid::new(width, height),
        ];

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

        Ok(fb_xyb)
    }

    fn render_vardct<'a>(
        &'a self,
        frame: &'a IndexedFrame<'f>,
        lf_frame: Option<&'a [SimpleGrid<f32>]>,
        cache: &mut RenderCache,
        region: Option<(u32, u32, u32, u32)>,
    ) -> Result<[SimpleGrid<f32>; 3]> {
        let span = tracing::span!(tracing::Level::TRACE, "RenderContext::render_vardct");
        let _guard = span.enter();

        let metadata = self.metadata();
        let frame_header = frame.header();
        let frame_data = frame.data();
        let lf_global = frame_data.lf_global.as_ref().ok_or(Error::IncompleteFrame)?;
        let lf_global_vardct = lf_global.vardct.as_ref().unwrap();
        let hf_global = frame_data.hf_global.as_ref().ok_or(Error::IncompleteFrame)?;
        let hf_global = hf_global.as_ref().expect("HfGlobal not found for VarDCT frame");
        let jpeg_upsampling = frame_header.jpeg_upsampling;
        let shifts_cbycr: [_; 3] = std::array::from_fn(|idx| {
            ChannelShift::from_jpeg_upsampling(jpeg_upsampling, idx)
        });
        let subsampled = jpeg_upsampling.into_iter().any(|x| x != 0);

        // Modular extra channels are already merged into GlobalModular,
        // so it's okay to drop PassGroup modular
        for (&(pass_idx, group_idx), group_pass) in &frame_data.group_pass {
            if let Some(region) = region {
                if !frame_header.is_group_collides_region(group_idx, region) {
                    continue;
                }
            }
            if !cache.coeff_merged.insert((pass_idx, group_idx)) {
                continue;
            }

            let hf_coeff = group_pass.hf_coeff.as_ref().unwrap();
            cache.group_coeffs
                .entry(group_idx as usize)
                .or_insert_with(HfCoeff::empty)
                .merge(hf_coeff);
        }
        let group_coeffs = &cache.group_coeffs;

        let quantizer = &lf_global_vardct.quantizer;
        let oim = &metadata.opsin_inverse_matrix;
        let dequant_matrices = &hf_global.dequant_matrices;
        let lf_chan_corr = &lf_global_vardct.lf_chan_corr;

        let width = frame_header.color_sample_width() as usize;
        let height = frame_header.color_sample_height() as usize;
        let width_rounded = ((width + 7) / 8) * 8;
        let height_rounded = ((height + 7) / 8) * 8;
        let mut fb_xyb = [
            SimpleGrid::new(width_rounded, height_rounded),
            SimpleGrid::new(width_rounded, height_rounded),
            SimpleGrid::new(width_rounded, height_rounded),
        ];

        let mut subgrids = {
            let [x, y, b] = &mut fb_xyb;
            let group_dim = frame_header.group_dim() as usize;
            [
                cut_grid::cut_with_block_info(x, group_coeffs, group_dim, shifts_cbycr[0]),
                cut_grid::cut_with_block_info(y, group_coeffs, group_dim, shifts_cbycr[1]),
                cut_grid::cut_with_block_info(b, group_coeffs, group_dim, shifts_cbycr[2]),
            ]
        };

        let lf_group_it = frame_data.lf_group
            .iter()
            .filter(|(&lf_group_idx, _)| {
                let Some(region) = region else { return true; };
                frame_header.is_lf_group_collides_region(lf_group_idx, region)
            });
        let mut hf_meta_map = HashMap::new();
        let mut lf_image_changed = false;
        for (&lf_group_idx, data) in lf_group_it {
            let group_x = lf_group_idx % frame_header.lf_groups_per_row();
            let group_y = lf_group_idx / frame_header.lf_groups_per_row();

            let lf_group_idx = lf_group_idx as usize;
            hf_meta_map.insert(lf_group_idx, data.hf_meta.as_ref().unwrap());

            if lf_frame.is_some() {
                continue;
            }
            if !cache.inserted_lf_groups.insert(lf_group_idx) {
                continue;
            }

            let group_dim = frame_header.group_dim();
            let lf_coeff = data.lf_coeff.as_ref().unwrap();
            let quant_channel_data = lf_coeff.lf_quant.image().channel_data();
            let [lf_x, lf_y, lf_b] = &mut cache.dequantized_lf;

            let left = (group_x * group_dim) as usize;
            let top = (group_y * group_dim) as usize;
            let lf_x = cut_grid::make_quant_cut_grid(lf_x, left, top, shifts_cbycr[0], &quant_channel_data[1]);
            let lf_y = cut_grid::make_quant_cut_grid(lf_y, left, top, shifts_cbycr[1], &quant_channel_data[0]);
            let lf_b = cut_grid::make_quant_cut_grid(lf_b, left, top, shifts_cbycr[2], &quant_channel_data[2]);
            let mut lf = [lf_x, lf_y, lf_b];

            vardct::dequant_lf(
                &mut lf,
                &lf_global.lf_dequant,
                quantizer,
                lf_coeff.extra_precision,
            );
            if !subsampled {
                vardct::chroma_from_luma_lf(
                    &mut lf,
                    &lf_global_vardct.lf_chan_corr,
                );
            }

            lf_image_changed = true;
        }

        if lf_image_changed && lf_frame.is_none() && !frame_header.flags.skip_adaptive_lf_smoothing() {
            let smoothed_lf = match &mut cache.smoothed_lf {
                Some(smoothed_lf) => smoothed_lf,
                x => {
                    let width = cache.dequantized_lf[0].width();
                    let height = cache.dequantized_lf[0].height();
                    *x = Some(std::array::from_fn(|_| SimpleGrid::new(width, height)));
                    x.as_mut().unwrap()
                },
            };
            vardct::adaptive_lf_smoothing(
                &cache.dequantized_lf,
                smoothed_lf,
                &lf_global.lf_dequant,
                quantizer,
            );
        }

        let dequantized_lf = if let Some(lf_frame) = lf_frame {
            lf_frame
        } else if let Some(smoothed_lf) = &cache.smoothed_lf {
            smoothed_lf
        } else {
            &cache.dequantized_lf
        };

        let group_dim = frame_header.group_dim() as usize;
        let groups_per_row = frame_header.groups_per_row() as usize;

        for (group_idx, hf_coeff) in group_coeffs {
            let mut x = subgrids[0].remove(group_idx).unwrap();
            let mut y = subgrids[1].remove(group_idx).unwrap();
            let mut b = subgrids[2].remove(group_idx).unwrap();
            let lf_group_id = frame_header.lf_group_idx_from_group_idx(*group_idx as u32) as usize;
            let hf_meta = hf_meta_map.get(&lf_group_id).unwrap();
            let x_from_y = &hf_meta.x_from_y;
            let b_from_y = &hf_meta.b_from_y;

            let group_row = group_idx / groups_per_row;
            let group_col = group_idx % groups_per_row;

            for coeff_data in hf_coeff.data() {
                let bx = coeff_data.bx;
                let by = coeff_data.by;
                let coord = (bx, by);
                let mut x = x.get_mut(&coord);
                let mut y = y.get_mut(&coord);
                let mut b = b.get_mut(&coord);
                let dct_select = coeff_data.dct_select;

                if let Some(x) = &mut x {
                    vardct::dequant_hf_varblock(coeff_data, x, 0, oim, quantizer, dequant_matrices, Some(frame_header.x_qm_scale));
                }
                if let Some(y) = &mut y {
                    vardct::dequant_hf_varblock(coeff_data, y, 1, oim, quantizer, dequant_matrices, None);
                }
                if let Some(b) = &mut b {
                    vardct::dequant_hf_varblock(coeff_data, b, 2, oim, quantizer, dequant_matrices, Some(frame_header.b_qm_scale));
                }

                let lf_left = (group_col * group_dim) / 8 + bx;
                let lf_top = (group_row * group_dim) / 8 + by;
                if !subsampled {
                    let lf_left = (lf_left % group_dim) * 8;
                    let lf_top = (lf_top % group_dim) * 8;
                    let mut xyb = [
                        &mut **x.as_mut().unwrap(),
                        &mut **y.as_mut().unwrap(),
                        &mut **b.as_mut().unwrap(),
                    ];
                    vardct::chroma_from_luma_hf(&mut xyb, lf_left, lf_top, x_from_y, b_from_y, lf_chan_corr);
                }

                for ((coeff, lf_dequant), shift) in [x, y, b].into_iter().zip(dequantized_lf.iter()).zip(shifts_cbycr) {
                    let Some(coeff) = coeff else { continue; };

                    let s_lf_left = lf_left >> shift.hshift();
                    let s_lf_top = lf_top >> shift.vshift();
                    if s_lf_left << shift.hshift() != lf_left || s_lf_top << shift.vshift() != lf_top {
                        continue;
                    }

                    let llf = vardct::llf_from_lf(lf_dequant, s_lf_left, s_lf_top, dct_select);
                    for y in 0..llf.height() {
                        for x in 0..llf.width() {
                            *coeff.get_mut(x, y) = *llf.get(x, y).unwrap();
                        }
                    }

                    vardct::transform(coeff, dct_select);
                }
            }
        }

        if width == width_rounded && height == width_rounded {
            Ok(fb_xyb)
        } else {
            Ok(fb_xyb.map(|g| {
                let mut new_g = SimpleGrid::new(width, height);
                for (new_row, row) in new_g.buf_mut().chunks_exact_mut(width).zip(g.buf().chunks_exact(width_rounded)) {
                    new_row.copy_from_slice(&row[..width]);
                }
                new_g
            }))
        }
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
    dequantized_lf: [SimpleGrid<f32>; 3],
    smoothed_lf: Option<[SimpleGrid<f32>; 3]>,
    inserted_lf_groups: HashSet<usize>,
    group_coeffs: HashMap<usize, HfCoeff>,
    coeff_merged: HashSet<(u32, u32)>,
}

impl RenderCache {
    pub fn new(frame: &IndexedFrame<'_>) -> Self {
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
        let dequantized_lf = whd.map(|(w, h)| SimpleGrid::new(w as usize, h as usize));
        Self {
            dequantized_lf,
            smoothed_lf: None,
            inserted_lf_groups: HashSet::new(),
            group_coeffs: HashMap::new(),
            coeff_merged: HashSet::new(),
        }
    }
}
