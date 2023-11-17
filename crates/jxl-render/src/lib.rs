//! This crate is the core of jxl-oxide that provides JPEG XL renderer.
use std::sync::Arc;

use jxl_bitstream::{Bitstream, Bundle};
use jxl_frame::{Frame, header::FrameType, data::{LfGroup, LfGlobalVarDct}, FrameContext};
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
use jxl_modular::{image::TransformedModularSubimage, MaConfig};
use jxl_threadpool::JxlThreadPool;
pub use region::Region;

use region::ImageWithRegion;
use state::*;

/// Render context that tracks loaded and rendered frames.
#[derive(Debug)]
pub struct RenderContext {
    image_header: Arc<ImageHeader>,
    pool: JxlThreadPool,
    pub(crate) frames: Vec<Arc<IndexedFrame>>,
    pub(crate) renders: Vec<Arc<FrameRenderHandle>>,
    pub(crate) keyframes: Vec<usize>,
    pub(crate) keyframe_in_progress: Option<usize>,
    pub(crate) refcounts: Vec<usize>,
    pub(crate) frame_deps: Vec<FrameDependence>,
    pub(crate) lf_frame: [usize; 4],
    pub(crate) reference: [usize; 4],
    pub(crate) loading_frame: Option<IndexedFrame>,
    pub(crate) loading_render_cache: Option<RenderCache>,
    pub(crate) loading_region: Option<Region>,
}

impl RenderContext {
    /// Creates a new render context without any multithreading.
    pub fn new(image_header: Arc<ImageHeader>) -> Self {
        Self::with_threads(image_header, JxlThreadPool::none())
    }

    /// Creates a new render context with custom thread pool.
    pub fn with_threads(image_header: Arc<ImageHeader>, pool: JxlThreadPool) -> Self {
        Self {
            image_header,
            pool,
            frames: Vec::new(),
            renders: Vec::new(),
            keyframes: Vec::new(),
            keyframe_in_progress: None,
            refcounts: Vec::new(),
            frame_deps: Vec::new(),
            lf_frame: [usize::MAX; 4],
            reference: [usize::MAX; 4],
            loading_frame: None,
            loading_render_cache: None,
            loading_region: None,
        }
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

    fn preserve_current_frame(&mut self) {
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

        let frame = Arc::new(frame);
        let render_op = self.render_op(Arc::clone(&frame), deps);
        let handle = if let Some(cache) = self.loading_render_cache.take() {
            let region = self.loading_region.take().unwrap();
            FrameRenderHandle::from_cache(Arc::clone(&frame), region, cache, render_op)
        } else {
            FrameRenderHandle::new(Arc::clone(&frame), render_op)
        };

        self.frames.push(Arc::clone(&frame));
        self.frame_deps.push(deps);
        self.renders.push(Arc::new(handle));
    }

    fn loading_frame(&self) -> Option<&IndexedFrame> {
        let search_from = self.keyframe_in_progress.or_else(|| self.keyframes.last().map(|x| x + 1)).unwrap_or(0);
        self.frames[search_from..]
            .iter()
            .map(|r| &**r)
            .chain(self.loading_frame.as_ref()).rev().find(|x| x.header().frame_type.is_progressive_frame())
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
    fn render_op(&self, frame: Arc<IndexedFrame>, deps: FrameDependence) -> RenderOp {
        let prev_frame_visibility = self.get_previous_frames_visibility(&frame);
        let reference_frames = ReferenceFrames {
            lf: (deps.lf != usize::MAX).then(|| Reference {
                frame: Arc::clone(&self.frames[deps.lf]),
                image: Arc::clone(&self.renders[deps.lf]),
            }),
            refs: deps.ref_slots.map(|r| (r != usize::MAX).then(|| Reference {
                frame: Arc::clone(&self.frames[r]),
                image: Arc::clone(&self.renders[r]),
            })),
        };

        let pool = self.pool.clone();
        Arc::new(move |mut state, image_region| {
            let mut cache = match state {
                FrameRender::InProgress(cache) => cache,
                _ => {
                    state = FrameRender::InProgress(Box::new(RenderCache::new(&frame)));
                    let FrameRender::InProgress(cache) = state else { unreachable!() };
                    cache
                },
            };

            tracing::debug!(index = frame.idx, ?image_region, "Rendering frame");

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
        let render_handle = Arc::clone(&self.renders[index]);
        self.pool.spawn(move || {
            render_handle.run(image_region);
        });
    }

    fn render_by_index(&self, index: usize, image_region: Option<Region>) -> Result<ImageWithRegion> {
        let deps = self.frame_deps[index];
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

        self.renders[index].run_with_image(image_region)
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
        let idx = *self.keyframes.get(keyframe_idx).ok_or(Error::IncompleteFrame)?;
        let mut grid = self.render_by_index(idx, image_region)?;
        let frame = &*self.frames[idx];

        self.postprocess_keyframe(frame, &mut grid, image_region);
        Ok(grid)
    }

    pub fn render_loading_keyframe(
        &mut self,
        image_region: Option<Region>,
    ) -> Result<(&IndexedFrame, ImageWithRegion)> {
        let mut current_frame_grid = None;
        if self.loading_frame().is_some() {
            let ret = self.render_loading_frame(image_region);
            match ret {
                Ok(grid) => current_frame_grid = Some(grid),
                Err(Error::IncompleteFrame) => {},
                Err(e) => return Err(e),
            }
        }

        let (frame, mut grid) = if let Some(grid) = current_frame_grid {
            let frame = self.loading_frame().unwrap();
            (frame, grid)
        } else if let Some(idx) = self.keyframe_in_progress {
            let grid = self.render_by_index(idx, image_region)?;
            let frame = &*self.frames[idx];
            (frame, grid)
        } else {
            return Err(Error::IncompleteFrame);
        };

        self.postprocess_keyframe(frame, &mut grid, image_region);
        Ok((frame, grid))
    }

    fn render_loading_frame(&mut self, image_region: Option<Region>) -> Result<ImageWithRegion> {
        let frame = self.loading_frame().unwrap();
        if !frame.header().frame_type.is_progressive_frame() {
            return Err(Error::IncompleteFrame);
        }

        let frame_region = image_region_to_frame(frame, image_region, false);
        self.loading_region = Some(frame_region);

        let frame = self.loading_frame().unwrap();
        let header = frame.header();
        if frame.try_parse_lf_global().is_none() {
            return Err(Error::IncompleteFrame);
        }

        let lf_frame_idx = self.lf_frame[header.lf_level as usize];
        if header.flags.use_lf_frame() {
            self.spawn_renderer(lf_frame_idx, image_region);
        }
        for idx in self.reference {
            if idx != usize::MAX {
                self.spawn_renderer(idx, image_region);
            }
        }

        tracing::debug!(?image_region, ?frame_region, "Rendering loading frame");
        let mut cache = self.loading_render_cache
            .take()
            .unwrap_or_else(|| {
                let frame = self.loading_frame().unwrap();
                RenderCache::new(frame)
            });

        let reference_frames = ReferenceFrames {
            lf: (lf_frame_idx != usize::MAX).then(|| Reference {
                frame: Arc::clone(&self.frames[lf_frame_idx]),
                image: Arc::clone(&self.renders[lf_frame_idx]),
            }),
            refs: self.reference.map(|r| (r != usize::MAX).then(|| Reference {
                frame: Arc::clone(&self.frames[r]),
                image: Arc::clone(&self.renders[r]),
            })),
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
        let image = match image_result {
            Ok(image) => image,
            Err(e) => {
                self.loading_render_cache = Some(cache);
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

    fn postprocess_keyframe(&self, frame: &IndexedFrame, grid: &mut ImageWithRegion, image_region: Option<Region>) {
        let frame_region = image_region_to_frame(frame, image_region, frame.header().lf_level > 0);
        if grid.region() != frame_region {
            let mut new_grid = ImageWithRegion::from_region(grid.channels(), frame_region);
            for (ch, g) in new_grid.buffer_mut().iter_mut().enumerate() {
                grid.clone_region_channel(frame_region, ch, g);
            }
            *grid = new_grid;
        }

        let image_header = frame.image_header();
        let frame_header = frame.header();

        if frame_header.save_before_ct {
            if frame_header.do_ycbcr {
                let [cb, y, cr, ..] = grid.buffer_mut() else { panic!() };
                jxl_color::ycbcr_to_rgb([cb, y, cr]);
            }
            inner::convert_color(image_header, grid.buffer_mut());
        }

        let channels = if self.metadata().grayscale() { 1 } else { 3 };
        grid.remove_channels(channels..3);
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

#[derive(Debug, Copy, Clone)]
struct FrameDependence {
    pub(crate) lf: usize,
    pub(crate) ref_slots: [usize; 4],
}

impl FrameDependence {
    pub fn indices(&self) -> impl Iterator<Item = usize> + 'static {
        std::iter::once(self.lf).chain(self.ref_slots).filter(|&v| v != usize::MAX)
    }
}

#[derive(Debug, Clone, Default)]
struct ReferenceFrames {
    pub(crate) lf: Option<Reference>,
    pub(crate) refs: [Option<Reference>; 4],
}

#[derive(Debug, Clone)]
struct Reference {
    pub(crate) frame: Arc<IndexedFrame>,
    pub(crate) image: Arc<FrameRenderHandle>,
}
