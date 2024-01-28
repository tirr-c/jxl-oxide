use std::{
    collections::HashMap,
    sync::{Arc, Condvar, Mutex, MutexGuard},
};

use jxl_frame::data::{HfGlobal, LfGlobal, LfGroup};
use jxl_modular::{ChannelShift, Sample};
use jxl_threadpool::JxlThreadPool;

use crate::{region::ImageWithRegion, Error, IndexedFrame, Reference, Region, Result};

pub type RenderOp<S> =
    Arc<dyn Fn(FrameRender<S>, Region) -> FrameRender<S> + Send + Sync + 'static>;

#[derive(Debug)]
pub struct RenderCache<S: Sample> {
    pub(crate) lf_global: Option<LfGlobal<S>>,
    pub(crate) hf_global: Option<HfGlobal>,
    pub(crate) lf_groups: HashMap<u32, LfGroup<S>>,
}

impl<S: Sample> RenderCache<S> {
    pub fn new(frame: &crate::IndexedFrame) -> Self {
        let frame_header = frame.header();
        let jpeg_upsampling = frame_header.jpeg_upsampling;
        let shifts_cbycr: [_; 3] =
            std::array::from_fn(|idx| ChannelShift::from_jpeg_upsampling(jpeg_upsampling, idx));

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

#[derive(Default)]
pub enum FrameRender<S: Sample> {
    #[default]
    None,
    Rendering,
    InProgress(Box<RenderCache<S>>),
    Done(ImageWithRegion),
    Err(crate::Error),
}

impl<S: Sample> std::fmt::Debug for FrameRender<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "None"),
            Self::Rendering => write!(f, "Rendering"),
            Self::InProgress(_) => write!(f, "InProgress(_)"),
            Self::Done(_) => write!(f, "Done(_)"),
            Self::Err(e) => f.debug_tuple("Err").field(e).finish(),
        }
    }
}

impl<S: Sample> FrameRender<S> {
    fn as_grid_mut(&mut self) -> Option<&mut ImageWithRegion> {
        if let Self::Done(grid) = self {
            Some(grid)
        } else {
            None
        }
    }
}

pub struct FrameRenderHandle<S: Sample> {
    frame: Arc<IndexedFrame>,
    image_region: Region,
    render: Mutex<FrameRender<S>>,
    condvar: Condvar,
    render_op: RenderOp<S>,
    refs: [Option<Reference<S>>; 4],
}

impl<S: Sample> std::fmt::Debug for FrameRenderHandle<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FrameRenderHandle")
            .field("render", &self.render)
            .field("condvar", &self.condvar)
            .finish_non_exhaustive()
    }
}

impl<S: Sample> FrameRenderHandle<S> {
    #[inline]
    pub fn new(
        frame: Arc<IndexedFrame>,
        image_region: Region,
        render_op: RenderOp<S>,
        refs: [Option<Reference<S>>; 4],
    ) -> Self {
        Self {
            frame,
            image_region,
            render: Mutex::new(FrameRender::None),
            condvar: Condvar::new(),
            render_op,
            refs,
        }
    }

    #[inline]
    pub fn from_cache(
        frame: Arc<IndexedFrame>,
        image_region: Region,
        cache: RenderCache<S>,
        render_op: RenderOp<S>,
        refs: [Option<Reference<S>>; 4],
    ) -> Self {
        let render = FrameRender::InProgress(Box::new(cache));
        Self {
            frame,
            image_region,
            render: Mutex::new(render),
            condvar: Condvar::new(),
            render_op,
            refs,
        }
    }

    pub fn run_with_image(self: Arc<Self>) -> Result<RenderedImage<S>> {
        let _guard = tracing::trace_span!("Run with image", index = self.frame.idx).entered();

        let render = if let Some(state) = self.start_render()? {
            let render_result = (self.render_op)(state, self.image_region);
            match render_result {
                FrameRender::InProgress(_) => {
                    drop(self.done_render(render_result));
                    return Err(Error::IncompleteFrame);
                }
                FrameRender::Err(e) => {
                    drop(self.done_render(FrameRender::None));
                    return Err(e);
                }
                _ => {}
            }
            self.done_render(render_result)
        } else {
            tracing::trace!("Another thread has started rendering");
            self.wait_until_render()?
        };
        drop(render);

        Ok(RenderedImage { image: self })
    }

    pub fn run(&self, image_region: Region) {
        let _guard = tracing::trace_span!("Run", index = self.frame.idx).entered();

        if let Some(state) = self.start_render_silent() {
            let render_result = (self.render_op)(state, image_region);
            drop(self.done_render(render_result));
        } else {
            tracing::trace!("Another thread has started rendering");
        }
    }

    fn start_render(&self) -> Result<Option<FrameRender<S>>> {
        let mut render_ref = self.render.lock().unwrap();
        let render = std::mem::replace(&mut *render_ref, FrameRender::Rendering);
        match render {
            FrameRender::Done(_) => {
                *render_ref = render;
                Ok(None)
            }
            FrameRender::Rendering => Ok(None),
            FrameRender::Err(e) => {
                *render_ref = FrameRender::None;
                Err(e)
            }
            render => Ok(Some(render)),
        }
    }

    fn start_render_silent(&self) -> Option<FrameRender<S>> {
        let mut render_ref = self.render.lock().unwrap();
        let render = std::mem::replace(&mut *render_ref, FrameRender::Rendering);
        match render {
            FrameRender::Done(_) => {
                *render_ref = render;
                None
            }
            FrameRender::Rendering => None,
            FrameRender::Err(_) => None,
            render => Some(render),
        }
    }

    fn wait_until_render(&self) -> Result<MutexGuard<'_, FrameRender<S>>> {
        let mut render_ref = self.render.lock().unwrap();
        loop {
            let render = std::mem::replace(&mut *render_ref, FrameRender::None);
            match render {
                FrameRender::Rendering => {
                    tracing::trace!(index = self.frame.idx, "Waiting...");
                    render_ref = self.condvar.wait(render_ref).unwrap();
                }
                FrameRender::Done(_) => {
                    *render_ref = render;
                    return Ok(render_ref);
                }
                FrameRender::None | FrameRender::InProgress(_) => {
                    return Err(Error::IncompleteFrame)
                }
                FrameRender::Err(e) => return Err(e),
            }
        }
    }

    fn done_render(&self, render: FrameRender<S>) -> MutexGuard<'_, FrameRender<S>> {
        assert!(!matches!(render, FrameRender::Rendering));
        let mut guard = self.render.lock().unwrap();
        *guard = render;
        self.condvar.notify_all();
        guard
    }
}

pub struct RenderedImage<S: Sample> {
    image: Arc<FrameRenderHandle<S>>,
}

impl<S: Sample> RenderedImage<S> {
    pub fn blend(
        &self,
        blend_cache: &mut HashMap<usize, ImageWithRegion>,
        pool: &JxlThreadPool,
    ) -> Result<ImageWithRegion> {
        let idx = self.image.frame.idx;
        if let Some(grid) = blend_cache.get(&idx) {
            return grid.try_clone();
        }

        let image_header = self.image.frame.image_header();
        let frame_header = self.image.frame.header();

        let mut grid = self.image.render.lock().unwrap();
        let grid = grid.as_grid_mut().unwrap();
        if !frame_header.frame_type.is_normal_frame() || frame_header.resets_canvas {
            return grid.try_clone().map_err(From::from);
        }

        if !grid.ct_done() {
            let ct_done = crate::inner::convert_color_for_record(
                image_header,
                frame_header.do_ycbcr,
                grid.buffer_mut(),
                pool,
            );
            grid.set_ct_done(ct_done);
        }

        let out = crate::blend::blend(
            image_header,
            self.image.refs.clone(),
            &self.image.frame,
            grid,
            blend_cache,
            pool,
        )?;
        Ok(out)
    }
}
