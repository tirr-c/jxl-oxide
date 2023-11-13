use std::{collections::HashMap, sync::{Arc, Mutex, Condvar, MutexGuard}};

use jxl_frame::data::{LfGlobal, HfGlobal, LfGroup};
use jxl_modular::ChannelShift;

use crate::{region::ImageWithRegion, Region, Result, Error, IndexedFrame};

pub type RenderOp = Arc<dyn Fn(FrameRender, Option<Region>) -> FrameRender + Send + Sync + 'static>;

#[derive(Debug)]
pub struct RenderCache {
    pub(crate) lf_global: Option<LfGlobal>,
    pub(crate) hf_global: Option<HfGlobal>,
    pub(crate) lf_groups: HashMap<u32, LfGroup>,
}

impl RenderCache {
    pub fn new(frame: &crate::IndexedFrame) -> Self {
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

#[derive(Debug, Default)]
pub enum FrameRender {
    #[default]
    None,
    Rendering,
    InProgress(Box<RenderCache>),
    Done(ImageWithRegion),
    Err(crate::Error),
}

impl FrameRender {
    pub fn as_grid(&self) -> Option<&ImageWithRegion> {
        if let Self::Done(grid) = self {
            Some(grid)
        } else {
            None
        }
    }
}

pub struct FrameRenderHandle {
    frame: Arc<IndexedFrame>,
    render: Mutex<FrameRender>,
    condvar: Condvar,
    render_op: RenderOp,
}

impl std::fmt::Debug for FrameRenderHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FrameRenderHandle").field("render", &self.render).field("condvar", &self.condvar).finish_non_exhaustive()
    }
}

impl FrameRenderHandle {
    #[inline]
    pub fn new(frame: Arc<IndexedFrame>, render_op: RenderOp) -> Self {
        Self {
            frame,
            render: Mutex::new(FrameRender::None),
            condvar: Condvar::new(),
            render_op,
        }
    }

    #[inline]
    pub fn from_cache(frame: Arc<IndexedFrame>, cache: RenderCache, render_op: RenderOp) -> Self {
        let render = FrameRender::InProgress(Box::new(cache));
        Self {
            frame,
            render: Mutex::new(render),
            condvar: Condvar::new(),
            render_op,
        }
    }

    pub fn run_with_image(&self, image_region: Option<Region>) -> Result<ImageWithRegion> {
        let _guard = tracing::trace_span!("Run with image", index = self.frame.idx).entered();

        let frame_region = crate::image_region_to_frame(&self.frame, image_region, false);
        let guard = if let Some(state) = self.start_render(frame_region)? {
            let render_result = (self.render_op)(state, image_region);
            match render_result {
                FrameRender::InProgress(_) => {
                    drop(self.done_render(render_result));
                    return Err(Error::IncompleteFrame);
                },
                FrameRender::Err(e) => {
                    drop(self.done_render(FrameRender::None));
                    return Err(e);
                },
                _ => {},
            }
            self.done_render(render_result)
        } else {
            tracing::trace!("Another thread has started rendering");
            self.wait_until_render()?
        };
        let grid = guard.as_grid().unwrap().clone();
        Ok(grid)
    }

    pub fn run(&self, image_region: Option<Region>) {
        let _guard = tracing::trace_span!("Run", index = self.frame.idx).entered();

        let frame_region = crate::image_region_to_frame(&self.frame, image_region, false);
        if let Some(state) = self.start_render_silent(frame_region) {
            let render_result = (self.render_op)(state, image_region);
            drop(self.done_render(render_result));
        } else {
            tracing::trace!("Another thread has started rendering");
        }
    }

    fn start_render(&self, frame_region: Region) -> Result<Option<FrameRender>> {
        let mut guard = self.render.lock().unwrap();
        let render = std::mem::replace(&mut *guard, FrameRender::Rendering);
        match render {
            FrameRender::Done(ref image) if image.region().contains(frame_region) => {
                *guard = render;
                Ok(None)
            },
            FrameRender::Rendering => Ok(None),
            FrameRender::Err(e) => {
                *guard = FrameRender::None;
                Err(e)
            },
            render => Ok(Some(render)),
        }
    }

    fn start_render_silent(&self, frame_region: Region) -> Option<FrameRender> {
        let mut guard = self.render.lock().unwrap();
        let render = std::mem::replace(&mut *guard, FrameRender::Rendering);
        match render {
            FrameRender::Done(ref image) if image.region().contains(frame_region) => {
                *guard = render;
                None
            },
            FrameRender::Rendering => None,
            FrameRender::Err(_) => None,
            render => Some(render),
        }
    }

    fn wait_until_render(&self) -> Result<MutexGuard<'_, FrameRender>> {
        let mut guard = self.render.lock().unwrap();
        loop {
            let render = std::mem::replace(&mut *guard, FrameRender::None);
            match render {
                FrameRender::Rendering => {
                    tracing::trace!(index = self.frame.idx, "Waiting...");
                    guard = self.condvar.wait(guard).unwrap();
                },
                FrameRender::Done(_) => {
                    *guard = render;
                    return Ok(guard);
                },
                FrameRender::None | FrameRender::InProgress(_) => return Err(Error::IncompleteFrame),
                FrameRender::Err(e) => return Err(e),
            }
        }
    }

    fn done_render(&self, render: FrameRender) -> MutexGuard<'_, FrameRender> {
        assert!(!matches!(render, FrameRender::Rendering));
        let mut guard = self.render.lock().unwrap();
        *guard = render;
        self.condvar.notify_all();
        guard
    }
}
