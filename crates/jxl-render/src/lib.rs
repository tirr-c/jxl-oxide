//! This crate is the core of jxl-oxide that provides JPEG XL renderer.
use std::sync::Arc;

use jxl_bitstream::Bitstream;
use jxl_frame::{Frame, header::FrameType, data::{LfGroup, LfGlobalVarDct}};
use jxl_image::ImageHeader;

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
use jxl_modular::{image::TransformedModularSubimage, MaConfig};
use jxl_threadpool::JxlThreadPool;
pub use region::Region;

use inner::*;
use region::ImageWithRegion;
use state::*;

/// Render context that tracks loaded and rendered frames.
#[derive(Debug)]
pub struct RenderContext {
    inner: ContextInner,
    pool: JxlThreadPool,
}

impl RenderContext {
    /// Creates a new render context.
    pub fn new(image_header: Arc<ImageHeader>) -> Self {
        Self {
            inner: ContextInner::new(image_header),
            pool: JxlThreadPool::none(),
        }
    }

    /// Creates a new render context with custom thread pool.
    pub fn with_threads(image_header: Arc<ImageHeader>, pool: JxlThreadPool) -> Self {
        Self {
            inner: ContextInner::with_threads(image_header, pool.clone()),
            pool,
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

    #[inline]
    pub fn loaded_frames(&self) -> usize {
        self.inner.frames.len()
    }
}

impl RenderContext {
    pub fn load_frame_header(&mut self, bitstream: &mut Bitstream) -> Result<&mut IndexedFrame> {
        if self.inner.loading_frame.is_some() && !self.try_finalize_current_frame() {
            panic!("another frame is still loading");
        }

        self.inner.load_frame_header(bitstream)
    }

    pub fn current_loading_frame(&mut self) -> Option<&mut IndexedFrame> {
        self.try_finalize_current_frame();
        self.inner.loading_frame.as_mut()
    }

    pub fn finalize_current_frame(&mut self) {
        if !self.try_finalize_current_frame() {
            panic!("frame is not fully loaded");
        }
    }

    fn try_finalize_current_frame(&mut self) -> bool {
        if let Some(loading_frame) = &self.inner.loading_frame {
            if loading_frame.is_loading_done() {
                self.inner.preserve_current_frame();
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
        self.inner.keyframe(keyframe_idx)
    }

    #[inline]
    pub fn frame(&self, frame_idx: usize) -> Option<&IndexedFrame> {
        if self.inner.frames.len() == frame_idx {
            self.inner.loading_frame.as_ref()
        } else {
            self.inner.frames.get(frame_idx).map(|x| &**x)
        }
    }
}

fn image_region_to_frame(frame: &Frame, image_region: Option<Region>, ignore_lf_level: bool) -> Region {
    let image_header = frame.image_header();
    let frame_header = frame.header();
    let frame_region = if frame_header.frame_type == FrameType::ReferenceOnly {
        Region::with_size(frame_header.width, frame_header.height)
    } else if let Some(image_region) = image_region {
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
    } else {
        Region::with_size(image_header.size.width, image_header.size.height)
            .translate(-frame_header.x0, -frame_header.y0)
    };

    if ignore_lf_level {
        frame_region
    } else {
        frame_region.downsample(frame_header.lf_level * 3)
    }
}

impl RenderContext {
    fn spawn_renderer(&self, index: usize, image_region: Option<Region>) {
        let render_handle = Arc::clone(&self.inner.renders[index]);
        self.pool.spawn(move || {
            render_handle.run(image_region);
        });
    }

    fn render_by_index(&self, index: usize, image_region: Option<Region>) -> Result<ImageWithRegion> {
        let deps = self.inner.frame_deps[index];
        let indices: Vec<_> = deps.indices().collect();
        if !indices.is_empty() {
            tracing::trace!(
                "Depends on {} {}",
                indices.len(),
                if indices.len() == 1 { "frame" } else { "frames" },
            );
            for dep in indices {
                self.spawn_renderer(dep, image_region);
            }
        }

        self.inner.renders[index].run_with_image(image_region)
    }

    /// Renders the first keyframe.
    ///
    /// The keyframe should be loaded in prior to rendering, with one of the loading methods.
    #[inline]
    pub fn render(
        &mut self,
        image_region: Option<Region>,
    ) -> Result<ImageWithRegion> {
        self.render_keyframe(0, image_region)
    }

    /// Renders the keyframe.
    ///
    /// The keyframe should be loaded in prior to rendering, with one of the loading methods.
    pub fn render_keyframe(
        &self,
        keyframe_idx: usize,
        image_region: Option<Region>,
    ) -> Result<ImageWithRegion> {
        let idx = *self.inner.keyframes.get(keyframe_idx).ok_or(Error::IncompleteFrame)?;
        let mut grid = self.render_by_index(idx, image_region)?;
        let frame = &*self.inner.frames[idx];

        let image_header = frame.image_header();
        let frame_header = frame.header();

        if frame_header.save_before_ct {
            if frame_header.do_ycbcr {
                let [cb, y, cr, ..] = grid.buffer_mut() else { panic!() };
                jxl_color::ycbcr_to_rgb([cb, y, cr]);
            }
            ContextInner::convert_color(image_header, grid.buffer_mut());
        }

        let channels = if self.inner.metadata().grayscale() { 1 } else { 3 };
        grid.remove_channels(channels..3);
        Ok(grid)
    }

    pub fn render_loading_keyframe(
        &mut self,
        image_region: Option<Region>,
    ) -> Result<(&IndexedFrame, ImageWithRegion)> {
        let mut current_frame_grid = None;
        if self.inner.loading_frame().is_some() {
            let ret = self.render_loading_frame(image_region);
            match ret {
                Ok(grid) => current_frame_grid = Some(grid),
                Err(Error::IncompleteFrame) => {},
                Err(e) => return Err(e),
            }
        }

        let (frame, mut grid) = if let Some(grid) = current_frame_grid {
            let frame = self.inner.loading_frame().unwrap();
            (frame, grid)
        } else if let Some(idx) = self.inner.keyframe_in_progress {
            let grid = self.render_by_index(idx, image_region)?;
            let frame = &*self.inner.frames[idx];
            (frame, grid)
        } else {
            return Err(Error::IncompleteFrame);
        };

        let image_header = frame.image_header();
        let frame_header = frame.header();

        if frame_header.save_before_ct {
            if frame_header.do_ycbcr {
                let [cb, y, cr, ..] = grid.buffer_mut() else { panic!() };
                jxl_color::ycbcr_to_rgb([cb, y, cr]);
            }
            ContextInner::convert_color(image_header, grid.buffer_mut());
        }

        let channels = if self.inner.metadata().grayscale() { 1 } else { 3 };
        grid.remove_channels(channels..3);
        Ok((frame, grid))
    }

    fn render_loading_frame(&mut self, image_region: Option<Region>) -> Result<ImageWithRegion> {
        let frame = self.inner.loading_frame().unwrap();
        if !frame.header().frame_type.is_progressive_frame() {
            return Err(Error::IncompleteFrame);
        }

        let header = frame.header();
        let frame_region = image_region_to_frame(frame, image_region, false);
        if frame.try_parse_lf_global().is_none() {
            return Err(Error::IncompleteFrame);
        }

        let lf_frame_idx = self.inner.lf_frame[header.lf_level as usize];
        if header.flags.use_lf_frame() {
            self.spawn_renderer(lf_frame_idx, image_region);
        }
        for idx in self.inner.reference {
            if idx != usize::MAX {
                self.spawn_renderer(idx, image_region);
            }
        }

        tracing::debug!(?image_region, ?frame_region, "Rendering loading frame");
        let mut cache = self.inner.loading_render_cache
            .take()
            .unwrap_or_else(|| {
                let frame = self.inner.loading_frame().unwrap();
                RenderCache::new(frame)
            });

        let reference_frames = ReferenceFrames {
            lf: (lf_frame_idx != usize::MAX).then(|| Reference {
                frame: Arc::clone(&self.inner.frames[lf_frame_idx]),
                image: Arc::clone(&self.inner.renders[lf_frame_idx]),
            }),
            refs: self.inner.reference.map(|r| (r != usize::MAX).then(|| Reference {
                frame: Arc::clone(&self.inner.frames[r]),
                image: Arc::clone(&self.inner.renders[r]),
            })),
        };

        let frame = self.inner.loading_frame().unwrap();
        let image_result = ContextInner::render_frame(
            frame,
            reference_frames,
            &mut cache,
            image_region,
            self.pool.clone(),
            self.inner.get_previous_frames_visibility(frame),
        );
        let image = match image_result {
            Ok(image) => image,
            Err(e) => {
                self.inner.loading_render_cache = Some(cache);
                return Err(e);
            },
        };

        if frame.header().lf_level > 0 {
            let frame_region = image_region_to_frame(frame, image_region, true);
            Ok(upsample_lf(&image, frame, frame_region))
        } else {
            Ok(image)
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

impl std::ops::DerefMut for IndexedFrame {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.f
    }
}

fn load_lf_groups(
    frame: &IndexedFrame,
    lf_global_vardct: Option<&LfGlobalVarDct>,
    lf_groups: &mut std::collections::HashMap<u32, LfGroup>,
    global_ma_config: Option<&MaConfig>,
    mlf_groups: Vec<TransformedModularSubimage>,
    lf_region: Region,
    pool: &JxlThreadPool,
) -> Result<()> {
    let frame_header = frame.header();
    let lf_groups_per_row = frame_header.lf_groups_per_row();
    let group_dim = frame_header.group_dim();
    let num_lf_groups = frame_header.num_lf_groups();

    let result = std::sync::RwLock::new(Result::Ok(()));
    let mut lf_groups_out = Vec::with_capacity(num_lf_groups as usize);
    lf_groups_out.resize_with(num_lf_groups as usize, || (None, false));
    for (&idx, lf_group) in lf_groups.iter() {
        lf_groups_out[idx as usize].1 = !lf_group.partial;
    }

    pool.scope(|scope| {
        let mut modular_it = mlf_groups.into_iter();
        for (idx, (lf_group_out, loaded)) in lf_groups_out.iter_mut().enumerate() {
            let modular = modular_it.next();
            if *loaded {
                continue;
            }

            let idx = idx as u32;
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

            let result = &result;
            scope.spawn(move |_| {
                match frame.try_parse_lf_group(lf_global_vardct, global_ma_config, modular, idx) {
                    Some(Ok(g)) => {
                        *lf_group_out = Some(g);
                    },
                    Some(Err(e)) => {
                        *result.write().unwrap() = Err(e.into());
                    },
                    None => {
                    },
                }
            });
        }
    });

    for (idx, (group, _)) in lf_groups_out.into_iter().enumerate() {
        if let Some(group) = group {
            lf_groups.insert(idx as u32, group);
        }
    }
    result.into_inner().unwrap()
}

fn upsample_lf(image: &ImageWithRegion, frame: &IndexedFrame, frame_region: Region) -> ImageWithRegion {
    let factor = frame.header().lf_level * 3;
    let step = 1usize << factor;
    let new_region = image.region().upsample(factor);
    let mut new_image = ImageWithRegion::from_region(image.channels(), new_region);
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
