//! This crate is the core of jxl-oxide that provides JPEG XL renderer.
use std::io::Read;
use std::sync::Arc;

use jxl_bitstream::Bitstream;
use jxl_frame::{Frame, data::TocGroupKind};
use jxl_grid::SimpleGrid;
use jxl_image::{ImageHeader, ExtraChannelType};

mod blend;
mod cut_grid;
mod dct;
mod error;
mod features;
mod filter;
mod inner;
mod vardct;
pub use error::{Error, Result};

use inner::*;

/// Render context that tracks loaded and rendered frames.
#[derive(Debug)]
pub struct RenderContext {
    inner: ContextInner,
    state: RenderState,
}

impl RenderContext {
    /// Creates a new render context.
    pub fn new(image_header: Arc<ImageHeader>) -> Self {
        Self {
            inner: ContextInner::new(image_header),
            state: RenderState::new(),
        }
    }
}

impl RenderContext {
    /// Returns the image width.
    #[inline]
    pub fn width(&self) -> u32 {
        self.inner.width()
    }

    /// Returns the image height.
    #[inline]
    pub fn height(&self) -> u32 {
        self.inner.height()
    }

    /// Returns the number of loaded keyframes in the context.
    #[inline]
    pub fn loaded_keyframes(&self) -> usize {
        self.inner.loaded_keyframes()
    }
}

impl RenderContext {
    /// Load all frames in the bitstream.
    pub fn load_all_frames<R: Read>(
        &mut self,
        bitstream: &mut Bitstream<R>,
    ) -> Result<()> {
        loop {
            let frame = self.inner.load_single(bitstream)?;

            let is_last = frame.header().is_last;
            let toc = frame.toc();
            let bookmark = toc.bookmark() + (toc.total_byte_size() * 8);
            self.inner.preserve_current_frame();
            self.state.preserve_current_frame();
            if is_last {
                break;
            }

            bitstream.skip_to_bookmark(bookmark)?;
        }

        Ok(())
    }

    /// Load a single keyframe from the bitstream.
    pub fn load_until_keyframe<R: Read>(
        &mut self,
        bitstream: &mut Bitstream<R>,
    ) -> Result<()> {
        loop {
            let frame = self.inner.load_single(bitstream)?;

            let is_keyframe = frame.header().is_keyframe();
            let toc = frame.toc();
            let bookmark = toc.bookmark() + (toc.total_byte_size() * 8);
            self.inner.preserve_current_frame();
            self.state.preserve_current_frame();
            if is_keyframe {
                break;
            }

            bitstream.skip_to_bookmark(bookmark)?;
        }

        Ok(())
    }
}

impl RenderContext {
    /// Returns the frame with the keyframe index, or `None` if the keyframe does not exist.
    #[inline]
    pub fn keyframe(&self, keyframe_idx: usize) -> Option<&IndexedFrame> {
        self.inner.keyframe(keyframe_idx)
    }
}

impl RenderContext {
    fn render_by_index(&mut self, index: usize) -> Result<()> {
        let span = tracing::span!(tracing::Level::TRACE, "RenderContext::render_by_index", index);
        let _guard = span.enter();

        if matches!(&self.state.renders[index], FrameRender::Done(_)) {
            return Ok(());
        }

        let deps = self.inner.frame_deps[index];
        for dep in deps.indices() {
            self.render_by_index(dep)?;
        }

        tracing::debug!(index, "Rendering frame");
        let frame = &self.inner.frames[index];
        let (prev, state) = self.state.renders.split_at_mut(index);
        let state = &mut state[0];
        let reference_frames = ReferenceFrames {
            lf: (deps.lf != usize::MAX).then(|| prev[deps.lf].as_grid().unwrap()),
            refs: deps.ref_slots.map(|r| (r != usize::MAX).then(|| prev[r].as_grid().unwrap())),
        };

        let cache = match state {
            FrameRender::Done(_) => return Ok(()),
            FrameRender::InProgress(cache) => cache,
            FrameRender::None => {
                *state = FrameRender::InProgress(Box::new(RenderCache::new(frame)));
                let FrameRender::InProgress(cache) = state else { unreachable!() };
                cache
            },
        };

        let grid = self.inner.render_frame(frame, reference_frames, cache, None)?;
        *state = FrameRender::Done(grid);

        let mut unref = |idx: usize| {
            tracing::debug!("Dereference frame #{idx}");
            let new_refcount = self.inner.refcounts[idx].saturating_sub(1);
            if new_refcount == 0 {
                tracing::debug!("Frame #{idx} is not referenced, dropping framebuffer");
                self.state.renders[idx] = FrameRender::None;
            }
        };

        if deps.lf != usize::MAX {
            unref(deps.lf);
        }
        for ref_idx in deps.ref_slots {
            if ref_idx != usize::MAX {
                unref(ref_idx);
            }
        }

        Ok(())
    }

    /// Renders the first keyframe.
    ///
    /// The keyframe should be loaded in prior to rendering, with one of the loading methods.
    #[inline]
    pub fn render(
        &mut self,
    ) -> Result<Vec<SimpleGrid<f32>>> {
        self.render_keyframe(0)
    }

    /// Renders the keyframe.
    ///
    /// The keyframe should be loaded in prior to rendering, with one of the loading methods.
    pub fn render_keyframe(
        &mut self,
        keyframe_idx: usize,
    ) -> Result<Vec<SimpleGrid<f32>>> {
        let (frame, mut grid) = if let Some(&idx) = self.inner.keyframes.get(keyframe_idx) {
            self.render_by_index(idx)?;
            let FrameRender::Done(grid) = &self.state.renders[idx] else { panic!(); };
            let frame = &self.inner.frames[idx];
            (frame, grid.clone())
        } else {
            let mut current_frame_grid = None;
            if let Some(frame) = &self.inner.loading_frame {
                if frame.header().frame_type.is_normal_frame() {
                    let ret = self.render_loading_frame();
                    match ret {
                        Ok(grid) => current_frame_grid = Some(grid),
                        Err(Error::IncompleteFrame) => {},
                        Err(e) => return Err(e),
                    }
                }
            }

            if let Some(grid) = current_frame_grid {
                let frame = self.inner.loading_frame.as_ref().unwrap();
                (frame, grid)
            } else if let Some(idx) = self.inner.keyframe_in_progress {
                self.render_by_index(idx)?;
                let FrameRender::Done(grid) = &self.state.renders[idx] else { panic!(); };
                let frame = &self.inner.frames[idx];
                (frame, grid.clone())
            } else {
                return Err(Error::IncompleteFrame);
            }
        };

        let frame_header = frame.header();

        if frame_header.save_before_ct {
            if frame_header.do_ycbcr {
                let [cb, y, cr, ..] = &mut *grid else { panic!() };
                jxl_color::ycbcr_to_rgb([cb, y, cr]);
            }
            self.inner.convert_color(&mut grid);
        }

        let channels = if self.inner.metadata().grayscale() { 1 } else { 3 };
        grid.drain(channels..3);
        Ok(grid)
    }

    fn render_loading_frame(&mut self) -> Result<Vec<SimpleGrid<f32>>> {
        let frame = self.inner.loading_frame.as_ref().unwrap();
        let header = frame.header();
        if frame.data(TocGroupKind::LfGlobal).is_none() {
            return Err(Error::IncompleteFrame);
        }

        let lf_frame = if header.flags.use_lf_frame() {
            let lf_frame_idx = self.inner.lf_frame[header.lf_level as usize];
            self.render_by_index(lf_frame_idx)?;
            Some(self.state.renders[lf_frame_idx].as_grid().unwrap())
        } else {
            None
        };

        let frame = self.inner.loading_frame.as_ref().unwrap();
        if self.state.loading_render_cache.is_none() {
            self.state.loading_render_cache = Some(RenderCache::new(frame));
        }
        let Some(cache) = &mut self.state.loading_render_cache else { unreachable!() };

        let reference_frames = ReferenceFrames {
            lf: lf_frame,
            refs: [None; 4],
        };

        self.inner.render_frame(frame, reference_frames, cache, None)
    }
}

#[derive(Debug)]
struct RenderState {
    renders: Vec<FrameRender>,
    loading_render_cache: Option<RenderCache>,
}

impl RenderState {
    fn new() -> Self {
        Self {
            renders: Vec::new(),
            loading_render_cache: None,
        }
    }
}

impl RenderState {
    fn preserve_current_frame(&mut self) {
        if let Some(cache) = self.loading_render_cache.take() {
            self.renders.push(FrameRender::InProgress(Box::new(cache)));
        } else {
            self.renders.push(FrameRender::None);
        }
    }
}

#[derive(Debug)]
enum FrameRender {
    None,
    InProgress(Box<RenderCache>),
    Done(Vec<SimpleGrid<f32>>),
}

impl FrameRender {
    fn as_grid(&self) -> Option<&[SimpleGrid<f32>]> {
        if let Self::Done(grid) = self {
            Some(grid)
        } else {
            None
        }
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
        IndexedFrame { f: frame, idx: index }
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

impl<'a> std::ops::DerefMut for IndexedFrame {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.f
    }
}

/// Renders a spot color channel onto color_channels
pub fn render_spot_colour(
    color_channels: &mut [SimpleGrid<f32>],
    ec_grid: &SimpleGrid<f32>,
    ec_ty: &ExtraChannelType,
) -> Result<()> {
    let ExtraChannelType::SpotColour { red, green, blue, solidity } = ec_ty else {
        return Err(Error::NotSupported("not a spot colour ec"));
    };
    if color_channels.len() != 3 {
        return Ok(())
    }

    let spot_colors = [red, green, blue];
    let s = ec_grid.buf();

    (0..3).for_each(|c| {
        let channel = color_channels[c].buf_mut();
        let color = spot_colors[c];
        assert_eq!(channel.len(), s.len());

        (0..channel.len()).for_each(|i| {
            let mix = s[i] * solidity;
            channel[i] = mix * color + (1.0 - mix) * channel[i];
        });
    });
    Ok(())
}
