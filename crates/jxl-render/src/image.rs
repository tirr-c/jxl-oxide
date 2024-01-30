use std::sync::Arc;

use jxl_frame::header::FrameType;
use jxl_grid::{AllocTracker, CutGrid, SimpleGrid};
use jxl_modular::Sample;
use jxl_threadpool::JxlThreadPool;

use crate::{util, FrameRender, FrameRenderHandle, Region, Result};

#[derive(Debug)]
pub struct ImageWithRegion {
    region: Region,
    buffer: Vec<SimpleGrid<f32>>,
    ct_done: bool,
    blend_done: bool,
    tracker: Option<AllocTracker>,
}

impl ImageWithRegion {
    pub(crate) fn from_region_and_tracker(
        channels: usize,
        region: Region,
        ct_done: bool,
        tracker: Option<&AllocTracker>,
    ) -> Result<Self> {
        let width = region.width as usize;
        let height = region.height as usize;
        let buffer =
            std::iter::repeat_with(|| SimpleGrid::with_alloc_tracker(width, height, tracker))
                .take(channels)
                .collect::<std::result::Result<_, _>>()?;
        let tracker = tracker.cloned();

        Ok(Self {
            region,
            buffer,
            ct_done,
            blend_done: false,
            tracker,
        })
    }

    pub(crate) fn from_buffer(
        buffer: Vec<SimpleGrid<f32>>,
        left: i32,
        top: i32,
        ct_done: bool,
    ) -> Self {
        let channels = buffer.len();
        if channels == 0 {
            Self {
                region: Region {
                    top,
                    left,
                    width: 0,
                    height: 0,
                },
                buffer,
                ct_done,
                blend_done: false,
                tracker: None,
            }
        } else {
            let width = buffer[0].width();
            let height = buffer[0].height();
            if buffer
                .iter()
                .any(|g| g.width() != width || g.height() != height)
            {
                panic!("Buffer size is not uniform");
            }
            let tracker = buffer[0].tracker();
            Self {
                region: Region {
                    top,
                    left,
                    width: width as u32,
                    height: height as u32,
                },
                buffer,
                ct_done,
                blend_done: false,
                tracker,
            }
        }
    }

    pub(crate) fn try_clone(&self) -> Result<Self> {
        Ok(Self {
            region: self.region,
            buffer: self
                .buffer
                .iter()
                .map(|x| x.try_clone())
                .collect::<std::result::Result<_, _>>()?,
            ct_done: self.ct_done,
            blend_done: false,
            tracker: self.tracker.clone(),
        })
    }

    #[inline]
    pub(crate) fn alloc_tracker(&self) -> Option<&AllocTracker> {
        self.tracker.as_ref()
    }

    #[inline]
    pub fn channels(&self) -> usize {
        self.buffer.len()
    }

    #[inline]
    pub fn region(&self) -> Region {
        self.region
    }

    #[inline]
    pub fn buffer(&self) -> &[SimpleGrid<f32>] {
        &self.buffer
    }

    #[inline]
    pub fn buffer_mut(&mut self) -> &mut [SimpleGrid<f32>] {
        &mut self.buffer
    }

    #[inline]
    pub fn take_buffer(&mut self) -> Vec<SimpleGrid<f32>> {
        std::mem::take(&mut self.buffer)
    }

    #[inline]
    pub(crate) fn ct_done(&self) -> bool {
        self.ct_done
    }

    #[inline]
    pub(crate) fn set_ct_done(&mut self, ct_done: bool) {
        self.ct_done = ct_done;
    }

    #[inline]
    pub(crate) fn set_blend_done(&mut self, blend_done: bool) {
        self.blend_done = blend_done;
    }

    #[inline]
    pub(crate) fn add_channel(&mut self) -> Result<&mut SimpleGrid<f32>> {
        self.buffer.push(SimpleGrid::with_alloc_tracker(
            self.region.width as usize,
            self.region.height as usize,
            self.tracker.as_ref(),
        )?);
        Ok(self.buffer.last_mut().unwrap())
    }

    #[inline]
    pub(crate) fn push_channel(&mut self, g: SimpleGrid<f32>) {
        assert_eq!(self.region.width as usize, g.width());
        assert_eq!(self.region.height as usize, g.height());
        self.buffer.push(g);
    }

    #[inline]
    pub(crate) fn remove_channels(&mut self, range: impl std::ops::RangeBounds<usize>) {
        self.buffer.drain(range);
    }

    pub(crate) fn clone_intersection(&self, new_region: Region) -> Result<Self> {
        let intersection = self.region.intersection(new_region);
        let begin_x = intersection.left.abs_diff(self.region.left) as usize;
        let begin_y = intersection.top.abs_diff(self.region.top) as usize;
        let mut out = Self::from_region_and_tracker(
            self.channels(),
            intersection,
            self.ct_done,
            self.tracker.as_ref(),
        )?;

        for (input, output) in self.buffer.iter().zip(out.buffer.iter_mut()) {
            let input_stride = input.width();
            let input_buf = input.buf();
            let output_stride = output.width();
            let output_buf = output.buf_mut();
            for (input_row, output_row) in input_buf
                .chunks_exact(input_stride)
                .skip(begin_y)
                .zip(output_buf.chunks_exact_mut(output_stride))
            {
                let input_row = &input_row[begin_x..][..output_stride];
                output_row.copy_from_slice(input_row);
            }
        }

        Ok(out)
    }

    pub(crate) fn clone_region_channel(
        &self,
        out_region: Region,
        channel_idx: usize,
        output: &mut SimpleGrid<f32>,
    ) {
        if output.width() != out_region.width as usize
            || output.height() != out_region.height as usize
        {
            panic!(
                "region mismatch, out_region={:?}, grid size={:?}",
                out_region,
                (output.width(), output.height())
            );
        }
        let input = self
            .buffer
            .get(channel_idx)
            .expect("channel index out of range");

        let intersection = self.region.intersection(out_region);
        if intersection.is_empty() {
            return;
        }

        let input_begin_x = intersection.left.abs_diff(self.region.left) as usize;
        let input_begin_y = intersection.top.abs_diff(self.region.top) as usize;
        let output_begin_x = intersection
            .left
            .abs_diff(out_region.left)
            .clamp(0, out_region.width) as usize;
        let output_begin_y = intersection
            .top
            .abs_diff(out_region.top)
            .clamp(0, out_region.height) as usize;

        let input_stride = input.width();
        let input_buf = input.buf();
        let output_stride = output.width();
        let output_buf = output.buf_mut();
        for (input_row, output_row) in input_buf
            .chunks_exact(input_stride)
            .skip(input_begin_y)
            .zip(
                output_buf
                    .chunks_exact_mut(output_stride)
                    .skip(output_begin_y),
            )
        {
            let input_row = &input_row[input_begin_x..][..intersection.width as usize];
            let output_row = &mut output_row[output_begin_x..][..intersection.width as usize];
            output_row.copy_from_slice(input_row);
        }
    }
}

impl ImageWithRegion {
    pub(crate) fn groups_with_group_id(
        &mut self,
        frame_header: &jxl_frame::FrameHeader,
    ) -> Vec<(u32, [CutGrid<'_, f32>; 3])> {
        let shifts_cbycr: [_; 3] = std::array::from_fn(|idx| {
            jxl_modular::ChannelShift::from_jpeg_upsampling(frame_header.jpeg_upsampling, idx)
        });

        let [fb_x, fb_y, fb_b, ..] = self.buffer.as_mut_slice() else {
            panic!();
        };

        let group_dim = frame_header.group_dim();
        let base_group_x = self.region.left as u32 / group_dim;
        let base_group_y = self.region.top as u32 / group_dim;
        let width = self.region.width;
        let height = self.region.height;
        let frame_groups_per_row = frame_header.groups_per_row();
        let groups_per_row = (width + group_dim - 1) / group_dim;

        let [fb_x, fb_y, fb_b] = [(0usize, fb_x), (1, fb_y), (2, fb_b)].map(|(idx, fb)| {
            let fb_width = fb.width();
            let shifted = shifts_cbycr[idx].shift_size((width, height));
            let fb = CutGrid::from_buf(
                fb.buf_mut(),
                shifted.0 as usize,
                shifted.1 as usize,
                fb_width,
            );

            let hshift = shifts_cbycr[idx].hshift();
            let vshift = shifts_cbycr[idx].vshift();
            let group_dim = group_dim as usize;
            fb.into_groups(group_dim >> hshift, group_dim >> vshift)
        });

        fb_x.into_iter()
            .zip(fb_y)
            .zip(fb_b)
            .enumerate()
            .map(|(idx, ((fb_x, fb_y), fb_b))| {
                let idx = idx as u32;
                let group_x = base_group_x + (idx % groups_per_row);
                let group_y = base_group_y + (idx / groups_per_row);
                let group_idx = group_y * frame_groups_per_row + group_x;
                (group_idx, [fb_x, fb_y, fb_b])
            })
            .collect()
    }
}

pub struct RenderedImage<S: Sample> {
    image: Arc<FrameRenderHandle<S>>,
}

impl<S: Sample> RenderedImage<S> {
    pub(crate) fn new(image: Arc<FrameRenderHandle<S>>) -> Self {
        Self { image }
    }
}

impl<S: Sample> RenderedImage<S> {
    pub(crate) fn blend<'r>(
        &'r self,
        oriented_image_region: Option<Region>,
        pool: &JxlThreadPool,
    ) -> Result<BlendResult<'r, S>> {
        let image_header = self.image.frame.image_header();
        let frame_header = self.image.frame.header();
        let image_region = self.image.image_region;
        let oriented_image_region = oriented_image_region
            .unwrap_or_else(|| util::apply_orientation_to_image_region(image_header, image_region));
        let frame_region = oriented_image_region
            .translate(-frame_header.x0, -frame_header.y0)
            .downsample(frame_header.lf_level * 3);
        let frame_region = util::pad_lf_region(frame_header, frame_region);
        let frame_region = util::pad_color_region(image_header, frame_header, frame_region);
        let frame_region = frame_region.upsample(frame_header.upsampling.ilog2());
        let frame_region = if frame_header.frame_type.is_normal_frame() {
            let full_image_region_in_frame =
                Region::with_size(image_header.size.width, image_header.size.height)
                    .translate(-frame_header.x0, -frame_header.y0);
            frame_region.intersection(full_image_region_in_frame)
        } else {
            frame_region
        };

        let mut grid_lock = self.image.render.lock().unwrap();
        let grid = grid_lock.as_grid_mut().unwrap();
        if grid.blend_done {
            return Ok(BlendResult(grid_lock));
        }

        if !frame_header.frame_type.is_normal_frame() || frame_header.resets_canvas {
            grid.blend_done = true;
            if frame_header.frame_type == FrameType::ReferenceOnly
                || grid.region().contains(frame_region)
            {
                return Ok(BlendResult(grid_lock));
            }

            let mut out = ImageWithRegion::from_region_and_tracker(
                grid.channels(),
                frame_region,
                grid.ct_done(),
                grid.alloc_tracker(),
            )?;
            for (idx, channel) in out.buffer_mut().iter_mut().enumerate() {
                grid.clone_region_channel(frame_region, idx, channel);
            }
            out.blend_done = true;
            *grid = out;
            return Ok(BlendResult(grid_lock));
        }

        if !grid.ct_done() {
            let ct_done = util::convert_color_for_record(
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
            frame_region,
            pool,
        )?;
        *grid = out;
        Ok(BlendResult(grid_lock))
    }
}

pub struct BlendResult<'r, S: Sample>(std::sync::MutexGuard<'r, FrameRender<S>>);

impl<S: Sample> std::ops::Deref for BlendResult<'_, S> {
    type Target = ImageWithRegion;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.0.as_grid().unwrap()
    }
}

impl<S: Sample> std::ops::DerefMut for BlendResult<'_, S> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.as_grid_mut().unwrap()
    }
}
