//! This crate is the core of jxl-oxide that provides JPEG XL renderer.
use std::sync::Arc;

use jxl_bitstream::{Bitstream, Bundle};
use jxl_color::{
    ColorEncodingWithProfile, ColorManagementSystem, ColourEncoding, EnumColourEncoding,
};
use jxl_frame::{
    data::{LfGlobalVarDct, LfGroup},
    header::FrameType,
    Frame, FrameContext,
};
use jxl_grid::AllocTracker;
use jxl_image::{ImageHeader, ImageMetadata};

mod blend;
mod dct;
mod error;
mod features;
mod filter;
mod inner;
mod modular;
mod region;
mod state;
mod vardct;
pub use error::{Error, Result};
pub use features::render_spot_color;
use jxl_modular::{image::TransformedModularSubimage, MaConfig, Sample};
use jxl_threadpool::JxlThreadPool;
pub use region::{ImageWithRegion, Region};

use state::*;

/// Render context that tracks loaded and rendered frames.
pub struct RenderContext {
    image_header: Arc<ImageHeader>,
    pool: JxlThreadPool,
    tracker: Option<AllocTracker>,
    pub(crate) frames: Vec<Arc<IndexedFrame>>,
    pub(crate) renders_wide: Vec<Arc<FrameRenderHandle<i32>>>,
    pub(crate) renders_narrow: Vec<Arc<FrameRenderHandle<i16>>>,
    pub(crate) keyframes: Vec<usize>,
    pub(crate) keyframe_in_progress: Option<usize>,
    pub(crate) refcounts: Vec<usize>,
    pub(crate) frame_deps: Vec<FrameDependence>,
    pub(crate) lf_frame: [usize; 4],
    pub(crate) reference: [usize; 4],
    pub(crate) loading_frame: Option<IndexedFrame>,
    pub(crate) loading_render_cache_wide: Option<RenderCache<i32>>,
    pub(crate) loading_render_cache_narrow: Option<RenderCache<i16>>,
    pub(crate) loading_region: Option<Region>,
    requested_image_region: Region,
    embedded_icc: Vec<u8>,
    requested_color_encoding: ColorEncodingWithProfile,
    cms: Box<dyn ColorManagementSystem + Send + Sync>,
}

impl std::fmt::Debug for RenderContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RenderContext").finish_non_exhaustive()
    }
}

impl RenderContext {
    pub fn builder() -> RenderContextBuilder {
        RenderContextBuilder::default()
    }
}

#[derive(Debug, Default)]
pub struct RenderContextBuilder {
    embedded_icc: Vec<u8>,
    pool: Option<JxlThreadPool>,
    tracker: Option<AllocTracker>,
}

impl RenderContextBuilder {
    pub fn embedded_icc(mut self, icc: Vec<u8>) -> Self {
        self.embedded_icc = icc;
        self
    }

    pub fn pool(mut self, pool: JxlThreadPool) -> Self {
        self.pool = Some(pool);
        self
    }

    pub fn alloc_tracker(mut self, tracker: AllocTracker) -> Self {
        self.tracker = Some(tracker);
        self
    }

    pub fn build(self, image_header: Arc<ImageHeader>) -> RenderContext {
        let color_encoding = &image_header.metadata.colour_encoding;
        let requested_color_encoding = if let ColourEncoding::Enum(encoding) = color_encoding {
            ColorEncodingWithProfile::new(encoding.clone())
        } else if image_header.metadata.xyb_encoded {
            ColorEncodingWithProfile::new(EnumColourEncoding::srgb(
                jxl_color::RenderingIntent::Relative,
            ))
        } else {
            match ColorEncodingWithProfile::with_icc(&self.embedded_icc) {
                Ok(x) => x,
                Err(e) => {
                    tracing::warn!(%e, "Malformed embedded ICC profile");
                    ColorEncodingWithProfile::new(EnumColourEncoding::srgb(
                        jxl_color::RenderingIntent::Relative,
                    ))
                }
            }
        };

        let full_image_region = Region::with_size(
            image_header.width_with_orientation(),
            image_header.height_with_orientation(),
        );

        RenderContext {
            image_header,
            tracker: self.tracker,
            pool: self.pool.unwrap_or_else(JxlThreadPool::none),
            frames: Vec::new(),
            renders_wide: Vec::new(),
            renders_narrow: Vec::new(),
            keyframes: Vec::new(),
            keyframe_in_progress: None,
            refcounts: Vec::new(),
            frame_deps: Vec::new(),
            lf_frame: [usize::MAX; 4],
            reference: [usize::MAX; 4],
            loading_frame: None,
            loading_render_cache_wide: None,
            loading_render_cache_narrow: None,
            loading_region: None,
            requested_image_region: full_image_region,
            embedded_icc: self.embedded_icc,
            requested_color_encoding,
            cms: Box::new(jxl_color::NullCms),
        }
    }
}

impl RenderContext {
    #[inline]
    pub fn alloc_tracker(&self) -> Option<&AllocTracker> {
        self.tracker.as_ref()
    }
}

impl RenderContext {
    #[inline]
    pub fn set_cms(&mut self, cms: impl ColorManagementSystem + Send + Sync + 'static) {
        self.cms = Box::new(cms);
    }

    #[inline]
    pub fn request_color_encoding(&mut self, encoding: ColorEncodingWithProfile) {
        self.requested_color_encoding = encoding;
    }

    #[inline]
    pub fn requested_color_encoding(&self) -> &ColorEncodingWithProfile {
        &self.requested_color_encoding
    }

    #[inline]
    pub fn request_image_region(&mut self, image_region: Region) {
        self.requested_image_region = image_region;
        self.reset_cache();
    }
}

impl RenderContext {
    /// Returns the image width.
    #[inline]
    pub fn width(&self) -> u32 {
        self.image_header.size.width
    }

    /// Returns the image height.
    #[inline]
    pub fn height(&self) -> u32 {
        self.image_header.size.height
    }

    #[inline]
    pub fn embedded_icc(&self) -> Option<&[u8]> {
        (!self.embedded_icc.is_empty()).then_some(&self.embedded_icc)
    }

    /// Returns the number of loaded keyframes in the context.
    #[inline]
    pub fn loaded_keyframes(&self) -> usize {
        self.keyframes.len()
    }

    /// Returns the number of loaded frames in the context, including frames that are not shown
    /// directly.
    #[inline]
    pub fn loaded_frames(&self) -> usize {
        self.frames.len()
    }

    #[inline]
    fn metadata(&self) -> &ImageMetadata {
        &self.image_header.metadata
    }

    #[inline]
    fn narrow_modular(&self) -> bool {
        self.image_header.metadata.modular_16bit_buffers
    }

    fn preserve_current_frame(&mut self) {
        let Some(frame) = self.loading_frame.take() else {
            return;
        };

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

        if !is_last
            && (header.duration == 0 || header.save_as_reference != 0)
            && header.frame_type != FrameType::LfFrame
        {
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

        let frame = Arc::new(frame);
        let image_region = self.requested_image_region;
        if self.narrow_modular() {
            let render_op = self.render_op_narrow(Arc::clone(&frame), deps);
            let handle = if let Some(cache) = self.loading_render_cache_narrow.take() {
                FrameRenderHandle::from_cache(Arc::clone(&frame), image_region, cache, render_op)
            } else {
                FrameRenderHandle::new(Arc::clone(&frame), image_region, render_op)
            };
            self.renders_narrow.push(Arc::new(handle));
        } else {
            let render_op = self.render_op_wide(Arc::clone(&frame), deps);
            let handle = if let Some(cache) = self.loading_render_cache_wide.take() {
                FrameRenderHandle::from_cache(Arc::clone(&frame), image_region, cache, render_op)
            } else {
                FrameRenderHandle::new(Arc::clone(&frame), image_region, render_op)
            };
            self.renders_wide.push(Arc::new(handle));
        }

        self.frames.push(Arc::clone(&frame));
        self.frame_deps.push(deps);
    }

    fn loading_frame(&self) -> Option<&IndexedFrame> {
        let search_from = self
            .keyframe_in_progress
            .or_else(|| self.keyframes.last().map(|x| x + 1))
            .unwrap_or(0);
        self.frames[search_from..]
            .iter()
            .map(|r| &**r)
            .chain(self.loading_frame.as_ref())
            .rev()
            .find(|x| x.header().frame_type.is_progressive_frame())
    }
}

impl RenderContext {
    pub fn load_frame_header(&mut self, bitstream: &mut Bitstream) -> Result<&mut IndexedFrame> {
        if self.loading_frame.is_some() && !self.try_finalize_current_frame() {
            panic!("another frame is still loading");
        }

        let image_header = &self.image_header;

        let bitstream_original = bitstream.clone();
        let frame = match Frame::parse(
            bitstream,
            FrameContext {
                image_header: image_header.clone(),
                tracker: self.tracker.as_ref(),
                pool: self.pool.clone(),
            },
        ) {
            Ok(frame) => frame,
            Err(e) => {
                *bitstream = bitstream_original;
                return Err(e.into());
            }
        };

        let header = frame.header();
        // Check if LF frame exists
        if header.flags.use_lf_frame() && self.lf_frame[header.lf_level as usize] == usize::MAX {
            return Err(Error::UninitializedLfFrame(header.lf_level));
        }

        self.loading_frame = Some(IndexedFrame::new(frame, self.frames.len()));
        Ok(self.loading_frame.as_mut().unwrap())
    }

    pub fn current_loading_frame(&mut self) -> Option<&mut IndexedFrame> {
        self.try_finalize_current_frame();
        self.loading_frame.as_mut()
    }

    pub fn finalize_current_frame(&mut self) {
        if !self.try_finalize_current_frame() {
            panic!("frame is not fully loaded");
        }
    }

    fn try_finalize_current_frame(&mut self) -> bool {
        if let Some(loading_frame) = &self.loading_frame {
            if loading_frame.is_loading_done() {
                self.preserve_current_frame();
                return true;
            }
        }
        false
    }
}

impl RenderContext {
    /// Returns the frame with the keyframe index, or `None` if the keyframe does not exist.
    #[inline]
    pub fn keyframe(&self, keyframe_idx: usize) -> Option<&IndexedFrame> {
        if keyframe_idx == self.keyframes.len() {
            self.loading_frame()
        } else if let Some(&idx) = self.keyframes.get(keyframe_idx) {
            Some(&self.frames[idx])
        } else {
            None
        }
    }

    #[inline]
    pub fn frame(&self, frame_idx: usize) -> Option<&IndexedFrame> {
        if self.frames.len() == frame_idx {
            self.loading_frame.as_ref()
        } else {
            self.frames.get(frame_idx).map(|x| &**x)
        }
    }
}

impl RenderContext {
    fn render_op_narrow(&self, frame: Arc<IndexedFrame>, deps: FrameDependence) -> RenderOp<i16> {
        let reference_frames = ReferenceFrames {
            lf: (deps.lf != usize::MAX).then(|| Reference {
                frame: Arc::clone(&self.frames[deps.lf]),
                image: Arc::clone(&self.renders_narrow[deps.lf]),
            }),
            refs: deps.ref_slots.map(|r| {
                (r != usize::MAX).then(|| Reference {
                    frame: Arc::clone(&self.frames[r]),
                    image: Arc::clone(&self.renders_narrow[r]),
                })
            }),
        };
        self.render_op(frame, reference_frames)
    }

    fn render_op_wide(&self, frame: Arc<IndexedFrame>, deps: FrameDependence) -> RenderOp<i32> {
        let reference_frames = ReferenceFrames {
            lf: (deps.lf != usize::MAX).then(|| Reference {
                frame: Arc::clone(&self.frames[deps.lf]),
                image: Arc::clone(&self.renders_wide[deps.lf]),
            }),
            refs: deps.ref_slots.map(|r| {
                (r != usize::MAX).then(|| Reference {
                    frame: Arc::clone(&self.frames[r]),
                    image: Arc::clone(&self.renders_wide[r]),
                })
            }),
        };
        self.render_op(frame, reference_frames)
    }

    fn render_op<S: Sample>(
        &self,
        frame: Arc<IndexedFrame>,
        reference_frames: ReferenceFrames<S>,
    ) -> RenderOp<S> {
        let prev_frame_visibility = self.get_previous_frames_visibility(&frame);

        let pool = self.pool.clone();
        Arc::new(move |mut state, image_region| {
            let mut cache = match state {
                FrameRender::InProgress(cache) => cache,
                _ => {
                    state = FrameRender::InProgress(Box::new(RenderCache::new(&frame)));
                    let FrameRender::InProgress(cache) = state else {
                        unreachable!()
                    };
                    cache
                }
            };

            let result = inner::render_frame(
                &frame,
                reference_frames.clone(),
                &mut cache,
                image_region,
                pool.clone(),
                prev_frame_visibility,
            );
            match result {
                Ok(grid) => FrameRender::Done(grid),
                Err(Error::IncompleteFrame) => FrameRender::InProgress(cache),
                Err(e) if e.unexpected_eof() => FrameRender::InProgress(cache),
                Err(e) => FrameRender::Err(e),
            }
        })
    }

    fn get_previous_frames_visibility<'a>(&'a self, frame: &'a IndexedFrame) -> (usize, usize) {
        let frame_idx = frame.index();
        let (is_keyframe, keyframe_idx) = match self.keyframes.binary_search(&frame_idx) {
            Ok(val) => (true, val),
            // Handle partial rendering. If val != self.keyframes.len(), is_keyframe() should be
            // false, since if not the index should exist in self.keyframes.
            Err(val) => (frame.header().is_keyframe(), val),
        };
        let prev_keyframes = &self.keyframes[..keyframe_idx];

        let visible_frames_num = keyframe_idx + is_keyframe as usize;

        let invisible_frames_num = if is_keyframe {
            0
        } else if prev_keyframes.is_empty() {
            1 + frame_idx
        } else {
            let last_visible_frame = prev_keyframes[keyframe_idx - 1];
            frame_idx - last_visible_frame
        };

        (visible_frames_num, invisible_frames_num)
    }
}

fn image_region_to_frame(frame: &Frame, image_region: Region, ignore_lf_level: bool) -> Region {
    let image_header = frame.image_header();
    let frame_header = frame.header();
    let frame_region = if frame_header.frame_type == FrameType::ReferenceOnly {
        Region::with_size(frame_header.width, frame_header.height)
    } else {
        let image_width = image_header.width_with_orientation();
        let image_height = image_header.height_with_orientation();
        let (_, _, mut left, mut top) = image_header.metadata.apply_orientation(
            image_width,
            image_height,
            image_region.left,
            image_region.top,
            true,
        );
        let (_, _, mut right, mut bottom) = image_header.metadata.apply_orientation(
            image_width,
            image_height,
            image_region.left + image_region.width as i32 - 1,
            image_region.top + image_region.height as i32 - 1,
            true,
        );

        if left > right {
            std::mem::swap(&mut left, &mut right);
        }
        if top > bottom {
            std::mem::swap(&mut top, &mut bottom);
        }
        let width = right.abs_diff(left) + 1;
        let height = bottom.abs_diff(top) + 1;
        Region {
            left: left - frame_header.x0,
            top: top - frame_header.y0,
            width,
            height,
        }
    };

    if ignore_lf_level {
        frame_region
    } else {
        frame_region.downsample(frame_header.lf_level * 3)
    }
}

impl RenderContext {
    fn spawn_renderer(&self, index: usize) {
        let image_region = self.requested_image_region;
        if self.narrow_modular() {
            let render_handle = Arc::clone(&self.renders_narrow[index]);
            self.pool.spawn(move || {
                render_handle.run(image_region);
            });
        } else {
            let render_handle = Arc::clone(&self.renders_wide[index]);
            self.pool.spawn(move || {
                render_handle.run(image_region);
            });
        }
    }

    fn render_by_index(&self, index: usize) -> Result<ImageWithRegion> {
        let deps = self.frame_deps[index];
        let indices: Vec<_> = deps.indices().collect();
        if !indices.is_empty() {
            tracing::trace!(
                "Depends on {} {}",
                indices.len(),
                if indices.len() == 1 {
                    "frame"
                } else {
                    "frames"
                },
            );
            for dep in indices {
                self.spawn_renderer(dep);
            }
        }

        if self.narrow_modular() {
            self.renders_narrow[index].run_with_image()
        } else {
            self.renders_wide[index].run_with_image()
        }
    }

    /// Renders the first keyframe.
    ///
    /// The keyframe should be loaded in prior to rendering, with one of the loading methods.
    #[inline]
    pub fn render(&mut self) -> Result<ImageWithRegion> {
        self.render_keyframe(0)
    }

    /// Renders the keyframe.
    ///
    /// The keyframe should be loaded in prior to rendering, with one of the loading methods.
    pub fn render_keyframe(&self, keyframe_idx: usize) -> Result<ImageWithRegion> {
        let idx = *self
            .keyframes
            .get(keyframe_idx)
            .ok_or(Error::IncompleteFrame)?;
        let mut grid = self.render_by_index(idx)?;
        let frame = &*self.frames[idx];

        self.postprocess_keyframe(frame, &mut grid)?;
        Ok(grid)
    }

    pub fn render_loading_keyframe(&mut self) -> Result<(&IndexedFrame, ImageWithRegion)> {
        let mut current_frame_grid = None;
        if self.loading_frame().is_some() {
            let ret = self.render_loading_frame();
            match ret {
                Ok(grid) => current_frame_grid = Some(grid),
                Err(Error::IncompleteFrame) => {}
                Err(e) => return Err(e),
            }
        }

        let (frame, mut grid) = if let Some(grid) = current_frame_grid {
            let frame = self.loading_frame().unwrap();
            (frame, grid)
        } else if let Some(idx) = self.keyframe_in_progress {
            let grid = self.render_by_index(idx)?;
            let frame = &*self.frames[idx];
            (frame, grid)
        } else {
            return Err(Error::IncompleteFrame);
        };

        self.postprocess_keyframe(frame, &mut grid)?;
        Ok((frame, grid))
    }

    pub fn reset_cache(&mut self) {
        let image_region = self.requested_image_region;

        self.loading_region = None;
        self.loading_render_cache_wide = None;
        self.loading_render_cache_narrow = None;
        for (idx, frame) in self.frames.iter().enumerate() {
            if frame.header().frame_type == FrameType::ReferenceOnly {
                continue;
            }

            let deps = self.frame_deps[idx];
            if self.narrow_modular() {
                let render_op = self.render_op_narrow(Arc::clone(frame), deps);
                let handle = FrameRenderHandle::new(Arc::clone(frame), image_region, render_op);
                self.renders_narrow[idx] = Arc::new(handle);
            } else {
                let render_op = self.render_op_wide(Arc::clone(frame), deps);
                let handle = FrameRenderHandle::new(Arc::clone(frame), image_region, render_op);
                self.renders_wide[idx] = Arc::new(handle);
            }
        }
    }

    fn render_loading_frame(&mut self) -> Result<ImageWithRegion> {
        let frame = self.loading_frame().unwrap();
        if !frame.header().frame_type.is_progressive_frame() {
            return Err(Error::IncompleteFrame);
        }

        let image_region = self.requested_image_region;
        let frame_region = image_region_to_frame(frame, image_region, false);
        self.loading_region = Some(frame_region);

        let frame = self.loading_frame().unwrap();
        let header = frame.header();
        let lf_global_failed = if self.narrow_modular() {
            frame.try_parse_lf_global::<i16>().is_none()
        } else {
            frame.try_parse_lf_global::<i32>().is_none()
        };
        if lf_global_failed {
            return Err(Error::IncompleteFrame);
        }

        let lf_frame_idx = self.lf_frame[header.lf_level as usize];
        if header.flags.use_lf_frame() {
            self.spawn_renderer(lf_frame_idx);
        }
        for idx in self.reference {
            if idx != usize::MAX {
                self.spawn_renderer(idx);
            }
        }

        tracing::debug!(?image_region, ?frame_region, "Rendering loading frame");
        let image = if self.narrow_modular() {
            let mut cache = self.loading_render_cache_narrow.take().unwrap_or_else(|| {
                let frame = self.loading_frame().unwrap();
                RenderCache::new(frame)
            });

            let reference_frames = ReferenceFrames {
                lf: (lf_frame_idx != usize::MAX).then(|| Reference {
                    frame: Arc::clone(&self.frames[lf_frame_idx]),
                    image: Arc::clone(&self.renders_narrow[lf_frame_idx]),
                }),
                refs: self.reference.map(|r| {
                    (r != usize::MAX).then(|| Reference {
                        frame: Arc::clone(&self.frames[r]),
                        image: Arc::clone(&self.renders_narrow[r]),
                    })
                }),
            };

            let frame = self.loading_frame().unwrap();
            let image_result = inner::render_frame(
                frame,
                reference_frames,
                &mut cache,
                image_region,
                self.pool.clone(),
                self.get_previous_frames_visibility(frame),
            );
            match image_result {
                Ok(image) => image,
                Err(e) => {
                    self.loading_render_cache_narrow = Some(cache);
                    return Err(e);
                }
            }
        } else {
            let mut cache = self.loading_render_cache_wide.take().unwrap_or_else(|| {
                let frame = self.loading_frame().unwrap();
                RenderCache::new(frame)
            });

            let reference_frames = ReferenceFrames {
                lf: (lf_frame_idx != usize::MAX).then(|| Reference {
                    frame: Arc::clone(&self.frames[lf_frame_idx]),
                    image: Arc::clone(&self.renders_wide[lf_frame_idx]),
                }),
                refs: self.reference.map(|r| {
                    (r != usize::MAX).then(|| Reference {
                        frame: Arc::clone(&self.frames[r]),
                        image: Arc::clone(&self.renders_wide[r]),
                    })
                }),
            };

            let frame = self.loading_frame().unwrap();
            let image_result = inner::render_frame(
                frame,
                reference_frames,
                &mut cache,
                image_region,
                self.pool.clone(),
                self.get_previous_frames_visibility(frame),
            );
            match image_result {
                Ok(image) => image,
                Err(e) => {
                    self.loading_render_cache_wide = Some(cache);
                    return Err(e);
                }
            }
        };

        let frame = self.loading_frame().unwrap();
        if frame.header().lf_level > 0 {
            let frame_region = image_region_to_frame(frame, image_region, true);
            Ok(upsample_lf(&image, frame, frame_region)?)
        } else {
            Ok(image)
        }
    }

    fn postprocess_keyframe(&self, frame: &IndexedFrame, grid: &mut ImageWithRegion) -> Result<()> {
        let image_region = self.requested_image_region;
        let frame_region = image_region_to_frame(frame, image_region, frame.header().lf_level > 0);
        if grid.region() != frame_region {
            let mut new_grid = ImageWithRegion::from_region_and_tracker(
                grid.channels(),
                frame_region,
                grid.ct_done(),
                self.tracker.as_ref(),
            )?;
            for (ch, g) in new_grid.buffer_mut().iter_mut().enumerate() {
                grid.clone_region_channel(frame_region, ch, g);
            }
            *grid = new_grid;
        }

        let metadata = self.metadata();
        let frame_header = frame.header();

        tracing::trace_span!("Transform to requested color encoding").in_scope(
            || -> Result<()> {
                let header_color_encoding = &metadata.colour_encoding;
                let frame_color_encoding = if !grid.ct_done() && metadata.xyb_encoded {
                    ColorEncodingWithProfile::new(EnumColourEncoding::xyb(
                        jxl_color::RenderingIntent::Perceptual,
                    ))
                } else if let ColourEncoding::Enum(encoding) = header_color_encoding {
                    ColorEncodingWithProfile::new(encoding.clone())
                } else {
                    ColorEncodingWithProfile::with_icc(&self.embedded_icc)?
                };
                tracing::trace!(?frame_color_encoding);
                tracing::trace!(requested_color_encoding = ?self.requested_color_encoding);
                tracing::trace!(do_ycbcr = frame_header.do_ycbcr);

                if !grid.ct_done() && frame_header.do_ycbcr {
                    let [cb, y, cr, ..] = grid.buffer_mut() else {
                        panic!()
                    };
                    jxl_color::ycbcr_to_rgb([cb, y, cr]);
                }

                let transform = jxl_color::ColorTransform::new(
                    &frame_color_encoding,
                    &self.requested_color_encoding,
                    &metadata.opsin_inverse_matrix,
                    &metadata.tone_mapping,
                )?;

                let (color_channels, extra_channels) = grid.buffer_mut().split_at_mut(3);
                let mut channels = color_channels
                    .iter_mut()
                    .map(|x| x.buf_mut())
                    .collect::<Vec<_>>();
                let mut has_black = false;
                for (grid, ec_info) in extra_channels.iter_mut().zip(&metadata.ec_info) {
                    if ec_info.is_black() {
                        channels.push(grid.buf_mut());
                        has_black = true;
                        break;
                    }
                }

                if has_black {
                    // 0 means full ink; invert samples
                    for grid in channels.iter_mut() {
                        for v in &mut **grid {
                            *v = 1.0 - *v;
                        }
                    }
                }

                let output_channels =
                    transform.run_with_threads(&mut channels, &*self.cms, &self.pool)?;
                if output_channels < 3 {
                    grid.remove_channels(output_channels..3);
                }
                grid.set_ct_done(true);
                Ok(())
            },
        )?;

        Ok(())
    }
}

fn load_lf_groups<S: Sample>(
    frame: &IndexedFrame,
    lf_global_vardct: Option<&LfGlobalVarDct>,
    lf_groups: &mut std::collections::HashMap<u32, LfGroup<S>>,
    global_ma_config: Option<&MaConfig>,
    mlf_groups: Vec<TransformedModularSubimage<S>>,
    lf_region: Region,
    pool: &JxlThreadPool,
) -> Result<()> {
    #[derive(Default)]
    struct LfGroupJob<'modular, S: Sample> {
        idx: u32,
        lf_group: Option<LfGroup<S>>,
        modular: Option<TransformedModularSubimage<'modular, S>>,
    }

    let frame_header = frame.header();
    let lf_groups_per_row = frame_header.lf_groups_per_row();
    let group_dim = frame_header.group_dim();
    let num_lf_groups = frame_header.num_lf_groups();

    let mlf_groups = mlf_groups
        .into_iter()
        .map(Some)
        .chain(std::iter::repeat_with(|| None));
    let mut lf_groups_out = (0..num_lf_groups)
        .map(|idx| LfGroupJob {
            idx,
            lf_group: None,
            modular: None,
        })
        .zip(mlf_groups)
        .filter_map(|(job, modular)| {
            let loaded = lf_groups.get(&job.idx).map(|g| !g.partial).unwrap_or(false);
            if loaded {
                return None;
            }

            let idx = job.idx;
            let left = (idx % lf_groups_per_row) * group_dim;
            let top = (idx / lf_groups_per_row) * group_dim;
            let lf_group_region = Region {
                left: left as i32,
                top: top as i32,
                width: group_dim,
                height: group_dim,
            };
            if lf_region.intersection(lf_group_region).is_empty() {
                return None;
            }

            Some(LfGroupJob { modular, ..job })
        })
        .collect::<Vec<_>>();

    let result = std::sync::RwLock::new(Result::Ok(()));
    pool.for_each_mut_slice(
        &mut lf_groups_out,
        |LfGroupJob {
             idx,
             lf_group,
             modular,
         }| {
            match frame.try_parse_lf_group(lf_global_vardct, global_ma_config, modular.take(), *idx)
            {
                Some(Ok(g)) => {
                    *lf_group = Some(g);
                }
                Some(Err(e)) => {
                    *result.write().unwrap() = Err(e.into());
                }
                None => {}
            }
        },
    );

    for LfGroupJob { idx, lf_group, .. } in lf_groups_out.into_iter() {
        if let Some(group) = lf_group {
            lf_groups.insert(idx, group);
        }
    }
    result.into_inner().unwrap()
}

fn upsample_lf(
    image: &ImageWithRegion,
    frame: &IndexedFrame,
    frame_region: Region,
) -> Result<ImageWithRegion> {
    let factor = frame.header().lf_level * 3;
    let step = 1usize << factor;
    let new_region = image.region().upsample(factor);
    let mut new_image = ImageWithRegion::from_region_and_tracker(
        image.channels(),
        new_region,
        image.ct_done(),
        frame.alloc_tracker(),
    )?;
    for (original, target) in image.buffer().iter().zip(new_image.buffer_mut()) {
        let height = original.height();
        let width = original.width();
        let stride = target.width();

        let original = original.buf();
        let target = target.buf_mut();
        for y in 0..height {
            let original = &original[y * width..];
            let target = &mut target[y * step * stride..];
            for (x, &value) in original[..width].iter().enumerate() {
                target[x * step..][..step].fill(value);
            }
            for row in 1..step {
                target.copy_within(..stride, stride * row);
            }
        }
    }
    new_image.clone_intersection(frame_region)
}

/// Frame with its index in the image.
#[derive(Debug)]
pub struct IndexedFrame {
    f: Frame,
    idx: usize,
}

impl IndexedFrame {
    fn new(frame: Frame, index: usize) -> Self {
        IndexedFrame {
            f: frame,
            idx: index,
        }
    }

    /// Returns the frame index.
    pub fn index(&self) -> usize {
        self.idx
    }
}

impl std::ops::Deref for IndexedFrame {
    type Target = Frame;

    fn deref(&self) -> &Self::Target {
        &self.f
    }
}

impl std::ops::DerefMut for IndexedFrame {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.f
    }
}

#[derive(Debug, Copy, Clone)]
struct FrameDependence {
    pub(crate) lf: usize,
    pub(crate) ref_slots: [usize; 4],
}

impl FrameDependence {
    pub fn indices(&self) -> impl Iterator<Item = usize> + 'static {
        std::iter::once(self.lf)
            .chain(self.ref_slots)
            .filter(|&v| v != usize::MAX)
    }
}

#[derive(Debug, Clone, Default)]
struct ReferenceFrames<S: Sample> {
    pub(crate) lf: Option<Reference<S>>,
    pub(crate) refs: [Option<Reference<S>>; 4],
}

#[derive(Debug, Clone)]
struct Reference<S: Sample> {
    pub(crate) frame: Arc<IndexedFrame>,
    pub(crate) image: Arc<FrameRenderHandle<S>>,
}
