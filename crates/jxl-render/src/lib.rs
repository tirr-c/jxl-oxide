//! This crate is the core of jxl-oxide that provides JPEG XL renderer.
use std::sync::{Arc, Mutex};

use jxl_bitstream::Bitstream;
use jxl_color::{
    ColorEncodingWithProfile, ColorManagementSystem, ColorTransform, ColourEncoding, ColourSpace,
    EnumColourEncoding, RenderingIntent, TransferFunction,
};
use jxl_frame::{Frame, FrameContext, header::FrameType};
use jxl_grid::AllocTracker;
use jxl_image::{ImageHeader, ImageMetadata};
use jxl_modular::Sample;
use jxl_oxide_common::Bundle;
use jxl_threadpool::JxlThreadPool;

mod blend;
mod error;
mod features;
mod filter;
mod image;
mod modular;
mod region;
mod render;
mod state;
mod util;
mod vardct;

pub use error::{Error, Result};
pub use features::render_spot_color;
pub use image::{ImageBuffer, ImageWithRegion};
pub use region::Region;
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
    cached_transform: Mutex<Option<ColorTransform>>,
    force_wide_buffers: bool,
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
    force_wide_buffers: bool,
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

    pub fn force_wide_buffers(mut self, force_wide_buffers: bool) -> Self {
        self.force_wide_buffers = force_wide_buffers;
        self
    }

    pub fn build(self, image_header: Arc<ImageHeader>) -> Result<RenderContext> {
        let color_encoding = &image_header.metadata.colour_encoding;
        let requested_color_encoding = match color_encoding {
            ColourEncoding::Enum(encoding) => ColorEncodingWithProfile::new(encoding.clone()),
            ColourEncoding::IccProfile(color_space) => {
                let header_is_gray = *color_space == ColourSpace::Grey;

                let parsed_icc = match ColorEncodingWithProfile::with_icc(&self.embedded_icc) {
                    Ok(parsed_icc) => {
                        let icc_is_gray = parsed_icc.is_grayscale();
                        if header_is_gray != icc_is_gray {
                            tracing::error!(
                                header_is_gray,
                                icc_is_gray,
                                "Color channel mismatch between header and ICC profile"
                            );
                            return Err(jxl_bitstream::Error::ValidationFailed(
                                "Color channel mismatch between header and ICC profile",
                            )
                            .into());
                        }

                        let is_supported_icc = parsed_icc.icc_profile().is_empty();
                        if !is_supported_icc {
                            tracing::trace!(
                                "Failed to convert embedded ICC into enum color encoding"
                            );
                        }

                        let allow_parsed_icc =
                            !image_header.metadata.xyb_encoded || is_supported_icc;
                        allow_parsed_icc.then_some(parsed_icc)
                    }
                    Err(e) => {
                        tracing::warn!(%e, "Malformed embedded ICC profile");
                        None
                    }
                };

                if let Some(profile) = parsed_icc {
                    profile
                } else if header_is_gray {
                    ColorEncodingWithProfile::new(EnumColourEncoding::gray_srgb(
                        RenderingIntent::Relative,
                    ))
                } else {
                    ColorEncodingWithProfile::new(EnumColourEncoding::srgb(
                        RenderingIntent::Relative,
                    ))
                }
            }
        };

        tracing::debug!(
            default_color_encoding = ?requested_color_encoding,
            "Setting default output color encoding"
        );

        let full_image_region = Region::with_size(
            image_header.width_with_orientation(),
            image_header.height_with_orientation(),
        );

        Ok(RenderContext {
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
            cached_transform: Mutex::new(None),
            force_wide_buffers: self.force_wide_buffers,
        })
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
        *self.cached_transform.get_mut().unwrap() = None;
        self.cms = Box::new(cms);
    }

    pub fn suggested_hdr_tf(&self) -> Option<TransferFunction> {
        let tf = match &self.image_header.metadata.colour_encoding {
            ColourEncoding::Enum(e) => e.tf,
            ColourEncoding::IccProfile(_) => {
                let icc = self.embedded_icc().unwrap();
                jxl_color::icc::icc_tf(icc)?
            }
        };

        match tf {
            TransferFunction::Pq | TransferFunction::Hlg => Some(tf),
            _ => None,
        }
    }

    #[inline]
    pub fn request_color_encoding(&mut self, encoding: ColorEncodingWithProfile) {
        *self.cached_transform.get_mut().unwrap() = None;
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

    #[inline]
    pub fn image_region(&self) -> Region {
        self.requested_image_region
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
        !self.force_wide_buffers && self.image_header.metadata.modular_16bit_buffers
    }

    fn preserve_current_frame(&mut self) {
        let Some(frame) = self.loading_frame.take() else {
            return;
        };

        let header = frame.header();
        let idx = self.frames.len();

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

        if header.can_reference() {
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
            let refs = reference_frames.refs.clone();

            let render_op = self.render_op::<i16>(Arc::clone(&frame), reference_frames);
            let handle = if let Some(cache) = self.loading_render_cache_narrow.take() {
                FrameRenderHandle::from_cache(
                    Arc::clone(&frame),
                    image_region,
                    cache,
                    render_op,
                    refs,
                )
            } else {
                FrameRenderHandle::new(Arc::clone(&frame), image_region, render_op, refs)
            };
            self.renders_narrow.push(Arc::new(handle));
        } else {
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
            let refs = reference_frames.refs.clone();

            let render_op = self.render_op::<i32>(Arc::clone(&frame), reference_frames);
            let handle = if let Some(cache) = self.loading_render_cache_wide.take() {
                FrameRenderHandle::from_cache(
                    Arc::clone(&frame),
                    image_region,
                    cache,
                    render_op,
                    refs,
                )
            } else {
                FrameRenderHandle::new(Arc::clone(&frame), image_region, render_op, refs)
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
        if let Some(loading_frame) = &self.loading_frame
            && loading_frame.is_loading_done()
        {
            self.preserve_current_frame();
            return true;
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
    fn do_render<S: Sample>(
        frame: &IndexedFrame,
        reference_frames: &ReferenceFrames<S>,
        mut state: FrameRender<S>,
        image_region: Region,
        prev_frame_visibility: (usize, usize),
        pool: &JxlThreadPool,
    ) -> FrameRender<S> {
        if let Some(lf) = &reference_frames.lf {
            tracing::trace!(idx = lf.frame.idx, "Spawn LF frame renderer");
            let lf_handle = Arc::clone(&lf.image);
            pool.spawn(move || {
                lf_handle.run(image_region);
            });
        }
        for grid in reference_frames.refs.iter().flatten() {
            tracing::trace!(idx = grid.frame.idx, "Spawn reference frame renderer");
            let ref_handle = Arc::clone(&grid.image);
            pool.spawn(move || {
                ref_handle.run(image_region);
            });
        }

        let mut cache = match state {
            FrameRender::InProgress(cache) => cache,
            _ => {
                state = FrameRender::InProgress(Box::new(RenderCache::new(frame)));
                let FrameRender::InProgress(cache) = state else {
                    unreachable!()
                };
                cache
            }
        };

        let result = render::render_frame(
            frame,
            reference_frames.clone(),
            &mut cache,
            image_region,
            pool.clone(),
            prev_frame_visibility,
        );
        match result {
            Ok(grid) => FrameRender::Done(grid),
            Err(e) if e.unexpected_eof() || matches!(e, Error::IncompleteFrame) => {
                if frame.is_loading_done() {
                    FrameRender::Err(e)
                } else {
                    FrameRender::InProgress(cache)
                }
            }
            Err(e) => FrameRender::Err(e),
        }
    }

    fn render_op<S: Sample>(
        &self,
        frame: Arc<IndexedFrame>,
        reference_frames: ReferenceFrames<S>,
    ) -> RenderOp<S> {
        let prev_frame_visibility = self.get_previous_frames_visibility(&frame);

        let pool = self.pool.clone();
        Arc::new(move |state, image_region| {
            Self::do_render(
                &frame,
                &reference_frames,
                state,
                image_region,
                prev_frame_visibility,
                &pool,
            )
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

impl RenderContext {
    fn spawn_renderer(&self, index: usize) {
        if !self.pool.is_multithreaded() {
            // Frame rendering will run immediately, this is not we want.
            return;
        }

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

    fn render_by_index(&self, index: usize) -> Result<Arc<ImageWithRegion>> {
        if self.narrow_modular() {
            Arc::clone(&self.renders_narrow[index])
                .run_with_image()?
                .blend(None, &self.pool)
        } else {
            Arc::clone(&self.renders_wide[index])
                .run_with_image()?
                .blend(None, &self.pool)
        }
    }

    /// Renders the first keyframe.
    ///
    /// The keyframe should be loaded in prior to rendering, with one of the loading methods.
    #[inline]
    pub fn render(&mut self) -> Result<Arc<ImageWithRegion>> {
        self.render_keyframe(0)
    }

    /// Renders the keyframe.
    ///
    /// The keyframe should be loaded in prior to rendering, with one of the loading methods.
    pub fn render_keyframe(&self, keyframe_idx: usize) -> Result<Arc<ImageWithRegion>> {
        let idx = *self
            .keyframes
            .get(keyframe_idx)
            .ok_or(Error::IncompleteFrame)?;
        let grid = self.render_by_index(idx)?;
        let frame = &*self.frames[idx];

        self.postprocess_keyframe(frame, grid)
    }

    pub fn render_loading_keyframe(&mut self) -> Result<(&IndexedFrame, Arc<ImageWithRegion>)> {
        let mut current_frame_grid = None;
        if self.loading_frame().is_some() {
            let ret = self.render_loading_frame();
            match ret {
                Ok(grid) => current_frame_grid = Some(grid),
                Err(Error::IncompleteFrame) => {}
                Err(e) => return Err(e),
            }
        }

        let (frame, grid) = if let Some(grid) = current_frame_grid {
            let frame = self.loading_frame().unwrap();
            (frame, Arc::new(grid))
        } else if let Some(idx) = self.keyframe_in_progress {
            let grid = self.render_by_index(idx)?;
            let frame = &*self.frames[idx];
            (frame, grid)
        } else {
            return Err(Error::IncompleteFrame);
        };

        let grid = self.postprocess_keyframe(frame, grid)?;
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
                let refs = reference_frames.refs.clone();

                let render_op = self.render_op::<i16>(Arc::clone(frame), reference_frames);
                let handle =
                    FrameRenderHandle::new(Arc::clone(frame), image_region, render_op, refs);
                self.renders_narrow[idx] = Arc::new(handle);
            } else {
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
                let refs = reference_frames.refs.clone();

                let render_op = self.render_op::<i32>(Arc::clone(frame), reference_frames);
                let handle =
                    FrameRenderHandle::new(Arc::clone(frame), image_region, render_op, refs);
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
        let frame_region = util::image_region_to_frame(frame, image_region, false);
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
            let state = if let Some(cache) = self.loading_render_cache_narrow.take() {
                FrameRender::InProgress(Box::new(cache))
            } else {
                FrameRender::None
            };

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
            let prev_frame_visibility = self.get_previous_frames_visibility(frame);

            let state = Self::do_render(
                frame,
                &reference_frames,
                state,
                image_region,
                prev_frame_visibility,
                &self.pool,
            );

            match state {
                FrameRender::InProgress(cache) => {
                    self.loading_render_cache_narrow = Some(*cache);
                    return Err(Error::IncompleteFrame);
                }
                FrameRender::Err(e) => {
                    return Err(e);
                }
                FrameRender::Done(mut image) => {
                    let skip_composition =
                        image::composite_preprocess(frame, &mut image, &self.pool)?;

                    if !skip_composition {
                        let oriented_image_region = util::apply_orientation_to_image_region(
                            &self.image_header,
                            image_region,
                        );
                        image::composite(
                            frame,
                            &mut image,
                            reference_frames.refs,
                            oriented_image_region,
                            &self.pool,
                        )?;
                    }

                    image
                }
                _ => todo!(),
            }
        } else {
            let state = if let Some(cache) = self.loading_render_cache_wide.take() {
                FrameRender::InProgress(Box::new(cache))
            } else {
                FrameRender::None
            };

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
            let prev_frame_visibility = self.get_previous_frames_visibility(frame);

            let state = Self::do_render(
                frame,
                &reference_frames,
                state,
                image_region,
                prev_frame_visibility,
                &self.pool,
            );

            match state {
                FrameRender::InProgress(cache) => {
                    self.loading_render_cache_wide = Some(*cache);
                    return Err(Error::IncompleteFrame);
                }
                FrameRender::Err(e) => {
                    return Err(e);
                }
                FrameRender::Done(mut image) => {
                    let skip_composition =
                        image::composite_preprocess(frame, &mut image, &self.pool)?;

                    if !skip_composition {
                        let oriented_image_region = util::apply_orientation_to_image_region(
                            &self.image_header,
                            image_region,
                        );
                        image::composite(
                            frame,
                            &mut image,
                            reference_frames.refs,
                            oriented_image_region,
                            &self.pool,
                        )?;
                    }

                    image
                }
                _ => todo!(),
            }
        };

        let frame = self.loading_frame().unwrap();
        if frame.header().lf_level > 0 {
            let mut render = image.upsample_lf(frame.header().lf_level)?;
            render.fill_opaque_alpha(&self.image_header.metadata.ec_info);
            Ok(render)
        } else {
            Ok(image)
        }
    }

    fn cache_color_transform(&self) -> Result<()> {
        let mut cached_transform = self.cached_transform.lock().unwrap();
        if cached_transform.is_some() {
            return Ok(());
        }

        let metadata = self.metadata();
        let header_color_encoding = &metadata.colour_encoding;

        let frame_color_encoding = if metadata.xyb_encoded {
            ColorEncodingWithProfile::new(EnumColourEncoding::xyb(RenderingIntent::Perceptual))
        } else if let ColourEncoding::Enum(encoding) = header_color_encoding {
            ColorEncodingWithProfile::new(encoding.clone())
        } else {
            ColorEncodingWithProfile::with_icc(&self.embedded_icc)?
        };
        tracing::trace!(?frame_color_encoding);
        tracing::trace!(requested_color_encoding = ?self.requested_color_encoding);

        let mut transform = ColorTransform::builder();
        transform.set_srgb_icc(!self.cms.supports_linear_tf());
        transform.from_pq(self.suggested_hdr_tf() == Some(TransferFunction::Pq));
        let transform = transform.build(
            &frame_color_encoding,
            &self.requested_color_encoding,
            &metadata.opsin_inverse_matrix,
            &metadata.tone_mapping,
            &*self.cms,
        )?;

        *cached_transform = Some(transform);
        Ok(())
    }

    fn postprocess_keyframe(
        &self,
        frame: &IndexedFrame,
        grid: Arc<ImageWithRegion>,
    ) -> Result<Arc<ImageWithRegion>> {
        let frame_header = frame.header();
        let metadata = self.metadata();

        tracing::trace_span!("Transform to requested color encoding").in_scope(|| -> Result<_> {
            if grid.ct_done() {
                return Ok(grid);
            }

            self.cache_color_transform()?;

            tracing::trace!(do_ycbcr = frame_header.do_ycbcr);

            let transform = self.cached_transform.lock().unwrap();
            let transform = transform.as_ref().unwrap();
            if transform.is_noop() && !frame_header.do_ycbcr {
                return Ok(grid);
            }

            let mut grid = grid.try_clone()?;

            if frame_header.do_ycbcr {
                grid.convert_modular_color(self.image_header.metadata.bit_depth)?;
                let [cb, y, cr] = grid.as_color_floats_mut();
                jxl_color::ycbcr_to_rgb([cb.buf_mut(), y.buf_mut(), cr.buf_mut()]);
            }
            if transform.is_noop() {
                let output_channels = transform.output_channels();
                grid.remove_color_channels(output_channels);
                return Ok(Arc::new(grid));
            }

            let encoded_color_channels = frame_header.encoded_color_channels();
            if encoded_color_channels < 3 {
                grid.clone_gray()?;
            }

            grid.convert_modular_color(self.image_header.metadata.bit_depth)?;
            let (color_channels, extra_channels) = grid.buffer_mut().split_at_mut(3);
            let mut channels = Vec::new();
            for grid in color_channels {
                channels.push(grid.as_float_mut().unwrap().buf_mut());
            }

            let mut has_black = false;
            for (grid, ec_info) in extra_channels.iter_mut().zip(&metadata.ec_info) {
                if ec_info.is_black() {
                    channels.push(grid.convert_to_float_modular(ec_info.bit_depth)?.buf_mut());
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

            let output_channels = transform.run_with_threads(&mut channels, &self.pool)?;
            if output_channels < 3 {
                grid.remove_color_channels(output_channels);
            }
            grid.set_ct_done(true);
            Ok(Arc::new(grid))
        })
    }
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
