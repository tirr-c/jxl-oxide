use std::io::Read;

use jxl_bitstream::Bitstream;
use jxl_frame::{Frame, ProgressiveResult};
use jxl_grid::SimpleGrid;
use jxl_image::ImageHeader;

mod blend;
mod cut_grid;
mod dct;
mod error;
mod fb;
mod features;
mod filter;
mod inner;
mod vardct;
pub use error::{Error, Result};
pub use fb::FrameBuffer;

use inner::*;

#[derive(Debug)]
pub struct RenderContext<'a> {
    inner: ContextInner<'a>,
    state: RenderState,
    icc: Vec<u8>,
}

impl<'a> RenderContext<'a> {
    pub fn new(image_header: &'a ImageHeader) -> Self {
        Self {
            inner: ContextInner::new(image_header),
            state: RenderState::new(),
            icc: Vec::new(),
        }
    }
}

impl RenderContext<'_> {
    #[inline]
    pub fn width(&self) -> u32 {
        self.inner.width()
    }

    #[inline]
    pub fn height(&self) -> u32 {
        self.inner.height()
    }

    #[inline]
    pub fn loaded_keyframes(&self) -> usize {
        self.inner.loaded_keyframes()
    }
}

impl RenderContext<'_> {
    pub fn read_icc_if_exists<R: Read>(&mut self, bitstream: &mut Bitstream<R>) -> Result<&[u8]> {
        if self.inner.metadata().colour_encoding.want_icc {
            tracing::debug!("Image has ICC profile");
            let icc = jxl_color::icc::read_icc(bitstream)?;
            self.icc = jxl_color::icc::decode_icc(&icc)?;
        }

        Ok(&self.icc)
    }

    pub fn icc(&self) -> &[u8] {
        &self.icc
    }

    pub fn load_cropped<R: Read>(
        &mut self,
        bitstream: &mut Bitstream<R>,
        progressive: bool,
        region: Option<(u32, u32, u32, u32)>,
    ) -> Result<ProgressiveResult> {
        loop {
            let (result, frame) = self.inner.load_cropped_single(bitstream, progressive, region)?;
            if result != ProgressiveResult::FrameComplete {
                return Ok(result);
            }

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

        Ok(ProgressiveResult::FrameComplete)
    }

    pub fn load_until_keyframe<R: Read>(
        &mut self,
        bitstream: &mut Bitstream<R>,
        progressive: bool,
        region: Option<(u32, u32, u32, u32)>,
    ) -> Result<ProgressiveResult> {
        loop {
            let (result, frame) = self.inner.load_cropped_single(bitstream, progressive, region)?;
            if result != ProgressiveResult::FrameComplete {
                return Ok(result);
            }

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

        Ok(ProgressiveResult::FrameComplete)
    }

    pub fn load_all_frames<R: Read>(
        &mut self,
        bitstream: &mut Bitstream<R>,
        progressive: bool,
    ) -> Result<ProgressiveResult> {
        self.load_cropped(bitstream, progressive, None)
    }
}

impl<'a> RenderContext<'a> {
    #[inline]
    pub fn keyframe(&self, keyframe_idx: usize) -> Option<&IndexedFrame<'a>> {
        self.inner.keyframe(keyframe_idx)
    }
}

impl RenderContext<'_> {
    fn render_by_index(&mut self, index: usize, region: Option<(u32, u32, u32, u32)>) -> Result<()> {
        let span = tracing::span!(tracing::Level::TRACE, "RenderContext::render_by_index", index);
        let _guard = span.enter();

        if matches!(&self.state.renders[index], FrameRender::Done(_)) {
            return Ok(());
        }

        let deps = self.inner.frame_deps[index];
        for dep in deps.indices() {
            self.render_by_index(dep, None)?;
        }

        tracing::debug!(index, region = format_args!("{:?}", region), "Rendering frame");
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

        let grid = self.inner.render_frame(frame, reference_frames, cache, region)?;
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

    #[inline]
    pub fn render_cropped(
        &mut self,
        region: Option<(u32, u32, u32, u32)>,
    ) -> Result<Vec<SimpleGrid<f32>>> {
        self.render_keyframe_cropped(0, region)
    }

    pub fn render_keyframe_cropped(
        &mut self,
        keyframe_idx: usize,
        region: Option<(u32, u32, u32, u32)>,
    ) -> Result<Vec<SimpleGrid<f32>>> {
        let (frame, grid) = if let Some(&idx) = self.inner.keyframes.get(keyframe_idx) {
            self.render_by_index(idx, region)?;
            let FrameRender::Done(grid) = &self.state.renders[idx] else { panic!(); };
            let frame = &self.inner.frames[idx];
            (frame, grid.clone())
        } else {
            let mut current_frame_grid = None;
            if let Some(frame) = &self.inner.loading_frame {
                if frame.header().frame_type.is_normal_frame() {
                    let ret = self.render_loading_frame(region);
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
                self.render_by_index(idx, region)?;
                let FrameRender::Done(grid) = &self.state.renders[idx] else { panic!(); };
                let frame = &self.inner.frames[idx];
                (frame, grid.clone())
            } else {
                return Err(Error::IncompleteFrame);
            }
        };

        let frame_header = frame.header();

        let mut cropped = if let Some((l, t, w, h)) = region {
            let mut cropped = Vec::with_capacity(grid.len());
            for g in grid {
                let mut new_grid = SimpleGrid::new(w as usize, h as usize);
                for (idx, v) in new_grid.buf_mut().iter_mut().enumerate() {
                    let y = idx / w as usize;
                    let x = idx % w as usize;
                    *v = *g.get(x + l as usize, y + t as usize).unwrap();
                }
                cropped.push(new_grid);
            }
            cropped
        } else {
            grid
        };

        if frame_header.save_before_ct {
            if frame_header.do_ycbcr {
                let [cb, y, cr, ..] = &mut *cropped else { panic!() };
                jxl_color::ycbcr_to_rgb([cb, y, cr]);
            }
            self.inner.convert_color(&mut cropped);
        }

        let channels = if self.inner.metadata().grayscale() { 1 } else { 3 };
        cropped.drain(channels..3);
        Ok(cropped)
    }

    fn render_loading_frame(&mut self, region: Option<(u32, u32, u32, u32)>) -> Result<Vec<SimpleGrid<f32>>> {
        let frame = self.inner.loading_frame.as_ref().unwrap();
        let header = frame.header();
        if frame.data().lf_global.is_none() {
            return Err(Error::IncompleteFrame);
        }

        let lf_frame = if header.flags.use_lf_frame() {
            let lf_frame_idx = self.inner.lf_frame[header.lf_level as usize];
            self.render_by_index(lf_frame_idx, None)?;
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

        self.inner.render_frame(frame, reference_frames, cache, region)
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

#[derive(Debug)]
pub struct IndexedFrame<'a> {
    f: Frame<'a>,
    idx: usize,
}

impl<'a> IndexedFrame<'a> {
    pub fn new(frame: Frame<'a>, idx: usize) -> Self {
        IndexedFrame { f: frame, idx }
    }

    pub fn idx(&self) -> usize {
        self.idx
    }
}

impl<'a> std::ops::Deref for IndexedFrame<'a> {
    type Target = Frame<'a>;

    fn deref(&self) -> &Self::Target {
        &self.f
    }
}

impl<'a> std::ops::DerefMut for IndexedFrame<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.f
    }
}
