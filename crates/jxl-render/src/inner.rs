use std::sync::Arc;

use jxl_bitstream::{Bitstream, Bundle};
use jxl_frame::{
    data::*,
    filter::{Gabor, EdgePreservingFilter},
    header::{Encoding, FrameType},
    Frame, FrameHeader, FrameContext,
};
use jxl_grid::SimpleGrid;
use jxl_image::{ImageHeader, ImageMetadata};
use jxl_threadpool::JxlThreadPool;

use crate::{
    blend,
    features,
    filter,
    modular,
    vardct,
    IndexedFrame,
    Result,
    Error,
    region::{Region, ImageWithRegion},
    state::RenderCache,
};

#[derive(Debug)]
pub struct ContextInner {
    image_header: Arc<ImageHeader>,
    pool: JxlThreadPool,
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
        Self::with_threads(image_header, JxlThreadPool::none())
    }

    pub fn with_threads(image_header: Arc<ImageHeader>, pool: JxlThreadPool) -> Self {
        Self {
            image_header,
            pool,
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
    pub fn loaded_keyframes(&self) -> usize {
        self.keyframes.len()
    }

    pub fn keyframe(&self, keyframe_idx: usize) -> Option<&IndexedFrame> {
        if keyframe_idx == self.keyframes.len() {
            self.loading_frame()
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

    pub(crate) fn loading_frame(&self) -> Option<&IndexedFrame> {
        let search_from = self.keyframe_in_progress.or_else(|| self.keyframes.last().map(|x| x + 1)).unwrap_or(0);
        self.frames[search_from..].iter().chain(self.loading_frame.as_ref()).rev().find(|x| x.header().frame_type.is_progressive_frame())
    }
}

impl ContextInner {
    pub(crate) fn load_frame_header(&mut self, bitstream: &mut Bitstream) -> Result<&mut IndexedFrame> {
        let image_header = &self.image_header;

        let bitstream_original = bitstream.clone();
        let frame = match Frame::parse(
            bitstream,
            FrameContext { image_header: image_header.clone(), pool: self.pool.clone() },
        ) {
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
        let image_header = frame.image_header();
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
            .zip(image_header.metadata.ec_info.iter())
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
            Encoding::Modular => {
                let (grid, gmodular) = modular::render_modular(frame, cache, color_padded_region, &self.pool)?;
                (grid, Some(gmodular))
            },
            Encoding::VarDct => {
                let result = vardct::render_vardct(frame, reference_frames.lf, cache, color_padded_region, &self.pool);
                match (result, reference_frames.lf) {
                    (Ok((grid, gmodular)), _) => (grid, Some(gmodular)),
                    (Err(e), Some(lf)) if e.unexpected_eof() => {
                        (super::upsample_lf(lf.image, lf.frame, frame_region), None)
                    },
                    (Err(e), _) => return Err(e),
                }
            },
        };
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
        filter::apply_epf(&mut fb, &cache.lf_groups, frame_header, &self.pool);

        self.upsample_color_channels(&mut fb, frame_header, frame_region);
        if let Some(gmodular) = gmodular {
            self.append_extra_channels(frame, &mut fb, gmodular, frame_region);
        }

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
        let Some(gmodular) = gmodular.modular.into_image() else { return; };
        let mut channel_data = gmodular.into_image_channels();
        let channel_data = channel_data.drain(extra_channel_from..);

        for (idx, g) in channel_data.enumerate() {
            tracing::debug!(ec_idx = idx, "Attaching extra channels");

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
            modular::copy_modular_groups(&g, &mut out, region, bit_depth, false);

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
