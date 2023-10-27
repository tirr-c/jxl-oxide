use std::{
    collections::HashMap,
    sync::Arc,
};

use jxl_bitstream::{Bitstream, Bundle};
use jxl_frame::{
    data::*,
    filter::{Gabor, EdgePreservingFilter},
    header::{Encoding, FrameType},
    Frame, FrameHeader,
};
use jxl_grid::{SimpleGrid, CutGrid};
use jxl_image::{ImageHeader, ImageMetadata};
use jxl_modular::ChannelShift;

use crate::{
    blend,
    features,
    filter,
    vardct,
    IndexedFrame,
    Result,
    Error, region::{Region, ImageWithRegion},
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
    pub(crate) fn load_frame_header(&mut self, bitstream: &mut Bitstream) -> Result<&mut IndexedFrame> {
        let image_header = &self.image_header;

        let bitstream_original = bitstream.clone();
        let frame = match Frame::parse(bitstream, image_header.clone()) {
            Ok(frame) => frame,
            Err(e) => {
                *bitstream = bitstream_original;
                return Err(e.into());
            },
        };

        let header = frame.header();
        // Check if LF frame exists
        if header.flags.use_lf_frame() && self.lf_frame[header.lf_level as usize] == usize::MAX {
            return Err(Error::UninitializedLfFrame(header.lf_level));
        }

        self.loading_frame = Some(IndexedFrame::new(frame, self.frames.len()));
        Ok(self.loading_frame.as_mut().unwrap())
    }
}

impl ContextInner {
    pub fn render_frame<'a>(
        &'a self,
        frame: &'a IndexedFrame,
        reference_frames: ReferenceFrames<'a>,
        cache: &mut RenderCache,
        frame_region: Region,
    ) -> Result<ImageWithRegion> {
        let frame_header = frame.header();
        let full_frame_region = Region::with_size(frame_header.color_sample_width(), frame_header.color_sample_height());
        let frame_region = if frame_header.lf_level != 0 {
            // Lower level frames might be padded, so apply padding to LF frames
            frame_region.pad(4 * frame_header.lf_level + 32)
        } else {
            frame_region
        };

        let color_upsample_factor = frame_header.upsampling.ilog2();
        let max_upsample_factor = frame_header.ec_upsampling.iter()
            .zip(self.image_header.metadata.ec_info.iter())
            .map(|(upsampling, ec_info)| upsampling.ilog2() + ec_info.dim_shift)
            .max()
            .unwrap_or(color_upsample_factor);

        let mut color_padded_region = if max_upsample_factor > 0 {
            // Additional upsampling pass is needed for every 3 levels of upsampling factor.
            let padded_region = frame_region.downsample(max_upsample_factor).pad(2 + (max_upsample_factor - 1) / 3);
            let upsample_diff = max_upsample_factor - color_upsample_factor;
            padded_region.upsample(upsample_diff)
        } else {
            frame_region
        };

        // TODO: actual region could be smaller.
        if let EdgePreservingFilter::Enabled { iters, .. } = frame_header.restoration_filter.epf {
            // EPF references adjacent samples.
            color_padded_region = if iters == 1 {
                color_padded_region.pad(2)
            } else if iters == 2 {
                color_padded_region.pad(5)
            } else {
                color_padded_region.pad(6)
            };
        }
        if frame_header.restoration_filter.gab.enabled() {
            // Gabor-like filter references adjacent samples.
            color_padded_region = color_padded_region.pad(1);
        }
        if frame_header.do_ycbcr {
            // Chroma upsampling references adjacent samples.
            color_padded_region = color_padded_region.pad(1).downsample(2).upsample(2);
        }
        if frame_header.restoration_filter.epf.enabled() {
            // EPF performs filtering in 8x8 blocks.
            color_padded_region = color_padded_region.container_aligned(8);
        }
        color_padded_region = color_padded_region.intersection(full_frame_region);

        let (mut fb, gmodular) = match frame_header.encoding {
            Encoding::Modular => self.render_modular(frame, cache, color_padded_region),
            Encoding::VarDct => self.render_vardct(frame, reference_frames.lf, cache, color_padded_region),
        }?;
        if fb.region().intersection(full_frame_region) != fb.region() {
            let mut new_fb = fb.clone_intersection(full_frame_region);
            std::mem::swap(&mut fb, &mut new_fb);
        }

        let [a, b, c] = fb.buffer_mut() else { panic!() };
        if frame.header().do_ycbcr {
            filter::apply_jpeg_upsampling([a, b, c], frame_header.jpeg_upsampling);
        }
        if let Gabor::Enabled(weights) = frame_header.restoration_filter.gab {
            filter::apply_gabor_like([a, b, c], weights);
        }
        filter::apply_epf(&mut fb, &cache.lf_groups, frame_header);

        self.upsample_color_channels(&mut fb, frame_header, frame_region);
        self.append_extra_channels(frame, &mut fb, gmodular, frame_region);

        self.render_features(frame, &mut fb, reference_frames.refs, cache)?;

        if !frame_header.save_before_ct {
            if frame_header.do_ycbcr {
                let [cb, y, cr, ..] = fb.buffer_mut() else { panic!() };
                jxl_color::ycbcr_to_rgb([cb, y, cr]);
            }
            self.convert_color(fb.buffer_mut());
        }

        Ok(if !frame_header.frame_type.is_normal_frame() || frame_header.resets_canvas {
            fb
        } else {
            blend::blend(&self.image_header, reference_frames.refs, frame, &fb)
        })
    }

    fn upsample_color_channels(&self, fb: &mut ImageWithRegion, frame_header: &FrameHeader, original_region: Region) {
        let upsample_factor = frame_header.upsampling.ilog2();
        let upsampled_region = fb.region().upsample(upsample_factor);
        let upsampled_buffer = if upsample_factor == 0 {
            if fb.region() == original_region {
                return;
            }
            fb.take_buffer()
        } else {
            let mut buffer = fb.take_buffer();
            tracing::trace_span!("Upsample color channels").in_scope(|| {
                for (idx, g) in buffer.iter_mut().enumerate() {
                    features::upsample(g, &self.image_header, frame_header, idx);
                }
            });
            buffer
        };

        let upsampled_fb = ImageWithRegion::from_buffer(
            upsampled_buffer,
            upsampled_region.left,
            upsampled_region.top,
        );
        tracing::trace_span!("Copy upsampled color channels").in_scope(|| {
            *fb = ImageWithRegion::from_region(upsampled_fb.channels(), original_region);
            for (channel_idx, output) in fb.buffer_mut().iter_mut().enumerate() {
                upsampled_fb.clone_region_channel(original_region, channel_idx, output);
            }
        });
    }

    fn append_extra_channels<'a>(
        &'a self,
        frame: &'a IndexedFrame,
        fb: &mut ImageWithRegion,
        gmodular: GlobalModular,
        original_region: Region,
    ) {
        let fb_region = fb.region();
        let frame_header = frame.header();

        let extra_channel_from = gmodular.extra_channel_from();
        let gmodular = &gmodular.modular;

        let channel_data = &gmodular.image().channel_data()[extra_channel_from..];

        if !channel_data.is_empty() {
            tracing::debug!("Attaching extra channels");
        }

        for (idx, g) in channel_data.iter().enumerate() {
            let upsampling = frame_header.ec_upsampling[idx];
            let ec_info = &self.image_header.metadata.ec_info[idx];

            let upsample_factor = upsampling.ilog2() + ec_info.dim_shift;
            let region = if upsample_factor > 0 {
                original_region.downsample(upsample_factor).pad(2 + (upsample_factor - 1) / 3)
            } else {
                original_region
            };
            let bit_depth = ec_info.bit_depth;

            let width = region.width as usize;
            let height = region.height as usize;
            let mut out = SimpleGrid::new(width, height);
            let buffer = out.buf_mut();

            let (gw, gh) = g.group_dim();
            let group_stride = g.groups_per_row();
            for (group_idx, g) in g.groups() {
                let base_x = (group_idx % group_stride) * gw;
                let base_y = (group_idx / group_stride) * gh;
                let group_region = Region {
                    left: base_x as i32,
                    top: base_y as i32,
                    width: gw as u32,
                    height: gh as u32,
                };
                let region_intersection = region.intersection(group_region);
                if region_intersection.is_empty() {
                    continue;
                }

                let group_x = region.left.abs_diff(region_intersection.left) as usize;
                let group_y = region.top.abs_diff(region_intersection.top) as usize;

                let begin_x = region_intersection.left.abs_diff(group_region.left) as usize;
                let begin_y = region_intersection.top.abs_diff(group_region.top) as usize;
                let end_x = begin_x + region_intersection.width as usize;
                let end_y = begin_y + region_intersection.height as usize;
                for (idx, &s) in g.buf().iter().enumerate() {
                    let x = idx % g.width();
                    let y = idx / g.width();
                    if y >= end_y {
                        break;
                    }
                    if y < begin_y || !(begin_x..end_x).contains(&x) {
                        continue;
                    }

                    buffer[(group_y + y - begin_y) * width + (group_x + x - begin_x)] = bit_depth.parse_integer_sample(s);
                }
            }

            let upsampled_region = region.upsample(upsample_factor);
            features::upsample(&mut out, &self.image_header, frame_header, idx + 3);
            let out = ImageWithRegion::from_buffer(
                vec![out],
                upsampled_region.left,
                upsampled_region.top,
            );
            let cropped = fb.add_channel();
            out.clone_region_channel(fb_region, 0, cropped);
        }
    }

    fn render_features<'a>(
        &'a self,
        frame: &'a IndexedFrame,
        grid: &mut ImageWithRegion,
        reference_grids: [Option<Reference>; 4],
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

        if let Some(patches) = &lf_global.patches {
            for patch in &patches.patches {
                let Some(ref_grid) = reference_grids[patch.ref_idx as usize] else {
                    return Err(Error::InvalidReference(patch.ref_idx));
                };
                blend::patch(&self.image_header, grid, ref_grid.image, patch);
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
            tracing::trace_span!("XYB to linear sRGB").in_scope(|| {
                jxl_color::xyb_to_linear_srgb(
                    [x, y, b],
                    &metadata.opsin_inverse_matrix,
                    metadata.tone_mapping.intensity_target,
                );
            });

            if metadata.colour_encoding.want_icc {
                // Don't convert tf, return linear sRGB as is
                return;
            }

            tracing::trace_span!("Linear sRGB to target colorspace").in_scope(|| {
                tracing::trace!(colour_encoding = ?metadata.colour_encoding);
                jxl_color::from_linear_srgb(
                    grid,
                    &metadata.colour_encoding,
                    metadata.tone_mapping.intensity_target,
                );
            });
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
        region: Region,
    ) -> Result<(ImageWithRegion, GlobalModular)> {
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
        let modular_region = compute_modular_region(frame_header, &gmodular, region);

        let jpeg_upsampling = frame_header.jpeg_upsampling;
        let shifts_cbycr = [0, 1, 2].map(|idx| {
            ChannelShift::from_jpeg_upsampling(jpeg_upsampling, idx)
        });
        let channels = metadata.encoded_color_channels();

        let stride = region.width as usize;
        let bit_depth = metadata.bit_depth;
        let mut fb_xyb = ImageWithRegion::from_region(channels, region);

        let lf_groups = &mut cache.lf_groups;
        load_lf_groups(frame, lf_global, lf_groups, modular_region.downsample(3), &mut gmodular)?;

        let group_dim = frame_header.group_dim();
        let groups_per_row = frame_header.groups_per_row();
        for pass_idx in 0..frame_header.passes.num_passes {
            for group_idx in 0..frame_header.num_groups() {
                let lf_group_idx = frame_header.lf_group_idx_from_group_idx(group_idx);
                let Some(lf_group) = lf_groups.get(&lf_group_idx) else { continue; };
                let Some(bitstream) = frame.pass_group_bitstream(pass_idx, group_idx).transpose()? else { continue; };
                let allow_partial = bitstream.partial;
                let mut bitstream = bitstream.bitstream;

                let group_x = group_idx % groups_per_row;
                let group_y = group_idx / groups_per_row;
                let left = group_x * group_dim;
                let top = group_y * group_dim;

                let group_region = Region {
                    left: left as i32,
                    top: top as i32,
                    width: group_dim,
                    height: group_dim,
                };
                if group_region.intersection(modular_region).is_empty() {
                    continue;
                }

                let shift = frame.pass_shifts(pass_idx);
                let result = decode_pass_group(
                    &mut bitstream,
                    PassGroupParams {
                        frame_header,
                        lf_group,
                        pass_idx,
                        group_idx,
                        shift,
                        gmodular: &mut gmodular,
                        vardct: None,
                        allow_partial,
                    },
                );
                if !allow_partial {
                    result?;
                }
            }
        }

        gmodular.modular.inverse_transform();
        let channel_data = gmodular.modular.image().channel_data();

        for ((g, shift), buffer) in channel_data.iter().zip(shifts_cbycr).zip(fb_xyb.buffer_mut()) {
            let buffer = buffer.buf_mut();
            let (gw, gh) = g.group_dim();
            let group_stride = g.groups_per_row();
            let region = region.downsample_separate(shift.hshift() as u32, shift.vshift() as u32);
            for (group_idx, g) in g.groups() {
                let base_x = (group_idx % group_stride) * gw;
                let base_y = (group_idx / group_stride) * gh;
                let group_region = Region {
                    left: base_x as i32,
                    top: base_y as i32,
                    width: gw as u32,
                    height: gh as u32,
                };
                let region_intersection = region.intersection(group_region);
                if region_intersection.is_empty() {
                    continue;
                }

                let group_x = region.left.abs_diff(region_intersection.left) as usize;
                let group_y = region.top.abs_diff(region_intersection.top) as usize;

                let begin_x = region_intersection.left.abs_diff(group_region.left) as usize;
                let begin_y = region_intersection.top.abs_diff(group_region.top) as usize;
                let end_x = begin_x + region_intersection.width as usize;
                let end_y = begin_y + region_intersection.height as usize;
                for (idx, &s) in g.buf().iter().enumerate() {
                    let x = idx % g.width();
                    let y = idx / g.width();
                    if y >= end_y {
                        break;
                    }
                    if y < begin_y || !(begin_x..end_x).contains(&x) {
                        continue;
                    }

                    buffer[(group_y + y - begin_y) * stride + (group_x + x - begin_x)] = if xyb_encoded {
                        s as f32
                    } else {
                        bit_depth.parse_integer_sample(s)
                    };
                }
            }
        }

        if channels == 1 {
            fb_xyb.add_channel();
            fb_xyb.add_channel();
            let fb_xyb = fb_xyb.buffer_mut();
            fb_xyb[1] = fb_xyb[0].clone();
            fb_xyb[2] = fb_xyb[0].clone();
        }
        if xyb_encoded {
            let fb_xyb = fb_xyb.buffer_mut();
            // Make Y'X'B' to X'Y'B'
            fb_xyb.swap(0, 1);
            let [x, y, b] = fb_xyb else { panic!() };
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
        lf_frame: Option<Reference<'a>>,
        cache: &mut RenderCache,
        region: Region,
    ) -> Result<(ImageWithRegion, GlobalModular)> {
        let span = tracing::span!(tracing::Level::TRACE, "Render VarDCT");
        let _guard = span.enter();

        let frame_header = frame.header();

        let jpeg_upsampling = frame_header.jpeg_upsampling;
        let shifts_cbycr: [_; 3] = std::array::from_fn(|idx| {
            ChannelShift::from_jpeg_upsampling(jpeg_upsampling, idx)
        });
        let subsampled = jpeg_upsampling.into_iter().any(|x| x != 0);

        let lf_global = if let Some(x) = &cache.lf_global {
            x
        } else {
            let lf_global = frame.try_parse_lf_global().ok_or(Error::IncompleteFrame)??;
            cache.lf_global = Some(lf_global);
            cache.lf_global.as_ref().unwrap()
        };
        let mut gmodular = lf_global.gmodular.clone();
        let lf_global_vardct = lf_global.vardct.as_ref().unwrap();

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

        let aligned_region = region.container_aligned(frame_header.group_dim());
        let aligned_lf_region = {
            // group_dim is multiple of 8
            let aligned_region_div8 = Region {
                left: aligned_region.left / 8,
                top: aligned_region.top / 8,
                width: aligned_region.width / 8,
                height: aligned_region.height / 8,
            };
            if frame_header.flags.skip_adaptive_lf_smoothing() {
                aligned_region_div8
            } else {
                aligned_region_div8.pad(1)
            }.container_aligned(frame_header.group_dim())
        };

        let modular_region = compute_modular_region(frame_header, &gmodular, aligned_region);
        let modular_lf_region = compute_modular_region(frame_header, &gmodular, aligned_lf_region)
            .intersection(Region::with_size(width_rounded as u32 / 8, height_rounded as u32 / 8));
        let aligned_region = aligned_region.intersection(Region::with_size(width_rounded as u32, height_rounded as u32));
        let aligned_lf_region = aligned_lf_region.intersection(Region::with_size(width_rounded as u32 / 8, height_rounded as u32 / 8));

        let mut fb_xyb = ImageWithRegion::from_region(3, aligned_region);
        let fb_stride = aligned_region.width as usize;

        let lf_groups = &mut cache.lf_groups;
        tracing::trace_span!("Load LF groups").in_scope(|| {
            load_lf_groups(frame, lf_global, lf_groups, modular_lf_region, &mut gmodular)
        })?;

        let group_dim = frame_header.group_dim();
        let (hf_cfl_data, mut lf_xyb) = tracing::trace_span!("Copy LFQuant").in_scope(|| {
            let mut hf_cfl_data = (!subsampled).then(|| {
                ImageWithRegion::from_region(2, aligned_lf_region.downsample(3))
            });

            let mut lf_xyb = ImageWithRegion::from_region(3, aligned_lf_region);

            if let Some(x) = lf_frame {
                x.image.clone_region_channel(aligned_lf_region, 0, &mut lf_xyb.buffer_mut()[0]);
                x.image.clone_region_channel(aligned_lf_region, 1, &mut lf_xyb.buffer_mut()[1]);
                x.image.clone_region_channel(aligned_lf_region, 2, &mut lf_xyb.buffer_mut()[2]);
            }

            let lf_groups_per_row = frame_header.lf_groups_per_row();
            for idx in 0..frame_header.num_lf_groups() {
                let Some(lf_group) = lf_groups.get(&idx) else { continue; };

                let lf_group_x = idx % lf_groups_per_row;
                let lf_group_y = idx / lf_groups_per_row;
                let left = lf_group_x * frame_header.group_dim();
                let top = lf_group_y * frame_header.group_dim();
                let lf_group_region = Region {
                    left: left as i32,
                    top: top as i32,
                    width: group_dim,
                    height: group_dim,
                };
                if aligned_lf_region.intersection(lf_group_region).is_empty() {
                    continue;
                }

                let left = left - aligned_lf_region.left as u32;
                let top = top - aligned_lf_region.top as u32;

                if lf_frame.is_none() {
                    let quantizer = &lf_global_vardct.quantizer;
                    let lf_coeff = lf_group.lf_coeff.as_ref().unwrap();
                    let channel_data = lf_coeff.lf_quant.image().channel_data();
                    let [lf_x, lf_y, lf_b] = lf_xyb.buffer_mut() else { panic!() };
                    vardct::copy_lf_dequant(
                        lf_x,
                        left as usize >> shifts_cbycr[0].hshift(),
                        top as usize >> shifts_cbycr[0].vshift(),
                        quantizer,
                        lf_global.lf_dequant.m_x_lf,
                        &channel_data[1],
                        lf_coeff.extra_precision,
                    );
                    vardct::copy_lf_dequant(
                        lf_y,
                        left as usize >> shifts_cbycr[1].hshift(),
                        top as usize >> shifts_cbycr[1].vshift(),
                        quantizer,
                        lf_global.lf_dequant.m_y_lf,
                        &channel_data[0],
                        lf_coeff.extra_precision,
                    );
                    vardct::copy_lf_dequant(
                        lf_b,
                        left as usize >> shifts_cbycr[2].hshift(),
                        top as usize >> shifts_cbycr[2].vshift(),
                        quantizer,
                        lf_global.lf_dequant.m_b_lf,
                        &channel_data[2],
                        lf_coeff.extra_precision,
                    );
                }

                let Some(hf_meta) = &lf_group.hf_meta else { continue; };

                if let Some(cfl) = &mut hf_cfl_data {
                    let corr = &lf_global_vardct.lf_chan_corr;
                    let [x_from_y, b_from_y] = cfl.buffer_mut() else { panic!() };
                    let group_x_from_y = &hf_meta.x_from_y;
                    let group_b_from_y = &hf_meta.b_from_y;
                    let left = left as usize / 8;
                    let top = top as usize / 8;
                    for y in 0..group_x_from_y.height() {
                        for x in 0..group_x_from_y.width() {
                            let v = *group_x_from_y.get(x, y).unwrap();
                            let kx = corr.base_correlation_x + (v as f32 / corr.colour_factor as f32);
                            *x_from_y.get_mut(left + x, top + y).unwrap() = kx;
                        }
                    }
                    for y in 0..group_b_from_y.height() {
                        for x in 0..group_b_from_y.width() {
                            let v = *group_b_from_y.get(x, y).unwrap();
                            let kb = corr.base_correlation_b + (v as f32 / corr.colour_factor as f32);
                            *b_from_y.get_mut(left + x, top + y).unwrap() = kb;
                        }
                    }
                }
            }

            (hf_cfl_data, lf_xyb)
        });

        if lf_frame.is_none() {
            if !subsampled {
                tracing::trace_span!("LF CfL").in_scope(|| {
                    vardct::chroma_from_luma_lf(
                        lf_xyb.buffer_mut(),
                        &lf_global_vardct.lf_chan_corr,
                    );
                });
            }

            if !frame_header.flags.skip_adaptive_lf_smoothing() {
                tracing::trace_span!("Adaptive LF smoothing").in_scope(|| {
                    vardct::adaptive_lf_smoothing(
                        lf_xyb.buffer_mut(),
                        &lf_global.lf_dequant,
                        &lf_global_vardct.quantizer,
                    );
                });
            }
        }

        let hf_global = if let Some(x) = &cache.hf_global {
            Some(x)
        } else {
            cache.hf_global = frame.try_parse_hf_global(Some(lf_global)).transpose()?;
            cache.hf_global.as_ref()
        };

        tracing::trace_span!("Decode pass groups").in_scope(|| -> Result<_> {
            let Some(hf_global) = hf_global else { return Ok(()); };
            let groups_per_row = frame_header.groups_per_row();
            for pass_idx in 0..frame_header.passes.num_passes {
                for group_idx in 0..frame_header.num_groups() {
                    let lf_group_idx = frame_header.lf_group_idx_from_group_idx(group_idx);
                    let Some(lf_group) = lf_groups.get(&lf_group_idx) else { continue; };
                    if lf_group.hf_meta.is_none() {
                        continue;
                    }
                    let Some(bitstream) = frame.pass_group_bitstream(pass_idx, group_idx).transpose()? else { continue; };
                    let allow_partial = bitstream.partial;
                    let mut bitstream = bitstream.bitstream;

                    let group_x = group_idx % groups_per_row;
                    let group_y = group_idx / groups_per_row;
                    let left = group_x * group_dim;
                    let top = group_y * group_dim;
                    let group_width = group_dim.min(width_rounded as u32 - left);
                    let group_height = group_dim.min(height_rounded as u32 - top);

                    let group_region = Region {
                        left: left as i32,
                        top: top as i32,
                        width: group_width,
                        height: group_height,
                    };
                    if group_region.intersection(modular_region).is_empty() {
                        continue;
                    }

                    let mut grid_xyb;
                    let vardct = if group_region.intersection(aligned_region).is_empty() {
                        None
                    } else {
                        let left = left - aligned_region.left as u32;
                        let top = top - aligned_region.top as u32;

                        let [fb_x, fb_y, fb_b] = fb_xyb.buffer_mut() else { panic!() };
                        grid_xyb = [(0usize, fb_x), (1, fb_y), (2, fb_b)].map(|(idx, fb)| {
                            let hshift = shifts_cbycr[idx].hshift();
                            let vshift = shifts_cbycr[idx].vshift();
                            let group_width = group_width >> hshift;
                            let group_height = group_height >> vshift;
                            let left = left >> hshift;
                            let top = top >> vshift;
                            let offset = top as usize * fb_stride + left as usize;
                            CutGrid::from_buf(&mut fb.buf_mut()[offset..], group_width as usize, group_height as usize, fb_stride)
                        });

                        Some(PassGroupParamsVardct {
                            lf_vardct: lf_global_vardct,
                            hf_global,
                            hf_coeff_output: &mut grid_xyb,
                        })
                    };

                    let shift = frame.pass_shifts(pass_idx);
                    let result = decode_pass_group(
                        &mut bitstream,
                        PassGroupParams {
                            frame_header,
                            lf_group,
                            pass_idx,
                            group_idx,
                            shift,
                            gmodular: &mut gmodular,
                            vardct,
                            allow_partial,
                        },
                    );
                    if !allow_partial {
                        result?;
                    }
                }
            }
            Ok(())
        })?;

        tracing::trace_span!("Extra channel inverse transform").in_scope(|| {
            gmodular.modular.inverse_transform();
        });

        tracing::trace_span!("Dequant HF").in_scope(|| {
            let Some(hf_global) = hf_global else { return; };
            vardct::dequant_hf_varblock(
                &mut fb_xyb,
                &self.image_header,
                frame_header,
                lf_global,
                &*lf_groups,
                hf_global,
            );
        });

        if let Some(cfl) = hf_cfl_data {
            tracing::trace_span!("HF CfL").in_scope(|| {
                if hf_global.is_none() {
                    return;
                }
                vardct::chroma_from_luma_hf(&mut fb_xyb, &cfl);
            });
        }

        tracing::trace_span!("Transform varblocks").in_scope(|| {
            vardct::transform_with_lf(
                &lf_xyb,
                &mut fb_xyb,
                frame_header,
                &*lf_groups,
            );
        });

        Ok((fb_xyb, gmodular))
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
    pub(crate) lf: Option<Reference<'state>>,
    pub(crate) refs: [Option<Reference<'state>>; 4],
}

#[derive(Debug, Copy, Clone)]
pub struct Reference<'state> {
    pub(crate) frame: &'state IndexedFrame,
    pub(crate) image: &'state ImageWithRegion,
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

fn load_lf_groups(
    frame: &IndexedFrame,
    lf_global: &LfGlobal,
    lf_groups: &mut HashMap<u32, LfGroup>,
    lf_region: Region,
    gmodular: &mut GlobalModular,
) -> Result<()> {
    let frame_header = frame.header();
    let lf_groups_per_row = frame_header.lf_groups_per_row();
    let group_dim = frame_header.group_dim();
    for idx in 0..frame_header.num_lf_groups() {
        let left = (idx % lf_groups_per_row) * group_dim;
        let top = (idx / lf_groups_per_row) * group_dim;
        let lf_group_region = Region {
            left: left as i32,
            top: top as i32,
            width: group_dim,
            height: group_dim,
        };
        if lf_region.intersection(lf_group_region).is_empty() {
            continue;
        }

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

    Ok(())
}

#[inline]
fn compute_modular_region(
    frame_header: &FrameHeader,
    gmodular: &GlobalModular,
    region: Region,
) -> Region {
    if gmodular.modular.has_palette() || gmodular.modular.has_squeeze() {
        Region::with_size(frame_header.color_sample_width(), frame_header.color_sample_height())
    } else {
        region
    }
}
