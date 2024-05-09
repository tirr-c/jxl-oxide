use std::sync::Arc;

use jxl_frame::{data::GlobalModular, FrameHeader};
use jxl_grid::{AlignedGrid, AllocTracker, MutableSubgrid};
use jxl_image::{BitDepth, ImageHeader};
use jxl_modular::{ChannelShift, Sample};
use jxl_threadpool::JxlThreadPool;
use jxl_vardct::LfChannelDequantization;

use crate::{util, FrameRender, FrameRenderHandle, Region, Result};

#[derive(Debug)]
pub enum ImageBuffer {
    F32(AlignedGrid<f32>),
    I32(AlignedGrid<i32>),
    I16(AlignedGrid<i16>),
}

impl ImageBuffer {
    #[inline]
    pub fn from_modular_channel<S: Sample>(g: AlignedGrid<S>) -> Self {
        let g = match S::try_into_grid_i16(g) {
            Ok(g) => return Self::I16(g),
            Err(g) => g,
        };
        match S::try_into_grid_i32(g) {
            Ok(g) => Self::I32(g),
            Err(_) => panic!(),
        }
    }

    #[inline]
    pub fn zeroed_f32(width: usize, height: usize, tracker: Option<&AllocTracker>) -> Result<Self> {
        let grid = AlignedGrid::with_alloc_tracker(width, height, tracker)?;
        Ok(Self::F32(grid))
    }

    #[inline]
    pub fn try_clone(&self) -> Result<Self> {
        match self {
            Self::F32(g) => g.try_clone().map(Self::F32),
            Self::I32(g) => g.try_clone().map(Self::I32),
            Self::I16(g) => g.try_clone().map(Self::I16),
        }
        .map_err(From::from)
    }

    #[inline]
    pub fn width(&self) -> usize {
        match self {
            Self::F32(g) => g.width(),
            Self::I32(g) => g.width(),
            Self::I16(g) => g.width(),
        }
    }

    #[inline]
    pub fn height(&self) -> usize {
        match self {
            Self::F32(g) => g.height(),
            Self::I32(g) => g.height(),
            Self::I16(g) => g.height(),
        }
    }

    #[inline]
    pub fn tracker(&self) -> Option<AllocTracker> {
        match self {
            Self::F32(g) => g.tracker(),
            Self::I32(g) => g.tracker(),
            Self::I16(g) => g.tracker(),
        }
    }

    #[inline]
    pub fn as_float(&self) -> Option<&AlignedGrid<f32>> {
        if let Self::F32(g) = self {
            Some(g)
        } else {
            None
        }
    }

    #[inline]
    pub fn as_float_mut(&mut self) -> Option<&mut AlignedGrid<f32>> {
        if let Self::F32(g) = self {
            Some(g)
        } else {
            None
        }
    }

    pub fn convert_to_float_modular(
        &mut self,
        bit_depth: BitDepth,
    ) -> Result<&mut AlignedGrid<f32>> {
        Ok(match self {
            Self::F32(g) => g,
            Self::I32(g) => {
                let mut out =
                    AlignedGrid::with_alloc_tracker(g.width(), g.height(), g.tracker().as_ref())?;
                for (o, &i) in out.buf_mut().iter_mut().zip(g.buf()) {
                    *o = bit_depth.parse_integer_sample(i);
                }

                *self = Self::F32(out);
                self.as_float_mut().unwrap()
            }
            Self::I16(g) => {
                let mut out =
                    AlignedGrid::with_alloc_tracker(g.width(), g.height(), g.tracker().as_ref())?;
                for (o, &i) in out.buf_mut().iter_mut().zip(g.buf()) {
                    *o = bit_depth.parse_integer_sample(i as i32);
                }

                *self = Self::F32(out);
                self.as_float_mut().unwrap()
            }
        })
    }

    pub fn cast_to_float(&mut self) -> Result<&mut AlignedGrid<f32>> {
        Ok(match self {
            Self::F32(g) => g,
            Self::I32(g) => {
                let mut out =
                    AlignedGrid::with_alloc_tracker(g.width(), g.height(), g.tracker().as_ref())?;
                for (o, &i) in out.buf_mut().iter_mut().zip(g.buf()) {
                    *o = i as f32;
                }

                *self = Self::F32(out);
                self.as_float_mut().unwrap()
            }
            Self::I16(g) => {
                let mut out =
                    AlignedGrid::with_alloc_tracker(g.width(), g.height(), g.tracker().as_ref())?;
                for (o, &i) in out.buf_mut().iter_mut().zip(g.buf()) {
                    *o = i as f32;
                }

                *self = Self::F32(out);
                self.as_float_mut().unwrap()
            }
        })
    }

    pub fn convert_to_float_modular_xyb<'g>(
        yxb: [&'g mut Self; 3],
        lf_dequant: &LfChannelDequantization,
    ) -> Result<[&'g mut AlignedGrid<f32>; 3]> {
        match yxb {
            [Self::F32(_), Self::F32(_), Self::F32(_)] => {
                panic!("channels are already converted");
            }
            [Self::I32(y), Self::I32(_), Self::I32(b)] => {
                for (b, &y) in b.buf_mut().iter_mut().zip(y.buf()) {
                    *b += y;
                }
            }
            [Self::I16(y), Self::I16(_), Self::I16(b)] => {
                for (b, &y) in b.buf_mut().iter_mut().zip(y.buf()) {
                    *b += y;
                }
            }
            _ => panic!(),
        }

        let [y, x, b] = yxb;
        let y = y.cast_to_float()?;
        let x = x.cast_to_float()?;
        let b = b.cast_to_float()?;
        let buf_y = y.buf_mut();
        let buf_x = x.buf_mut();
        let buf_b = b.buf_mut();
        let m_x_lf = lf_dequant.m_x_lf_unscaled();
        let m_y_lf = lf_dequant.m_y_lf_unscaled();
        let m_b_lf = lf_dequant.m_b_lf_unscaled();

        for ((y, x), b) in buf_y.iter_mut().zip(buf_x).zip(buf_b) {
            let py = *y;
            let px = *x;
            *y = px * m_x_lf;
            *x = py * m_y_lf;
            *b *= m_b_lf;
        }

        Ok([y, x, b])
    }

    pub(crate) fn upsample_nn(&self, factor: u32) -> Result<ImageBuffer> {
        #[inline]
        fn inner<S: Copy>(
            original: &[S],
            target: &mut [S],
            width: usize,
            height: usize,
            factor: u32,
        ) {
            assert_eq!(original.len(), width * height);
            assert_eq!(target.len(), original.len() << (factor * 2));
            let step = 1usize << factor;
            let stride = width << factor;
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

        if factor == 0 {
            return self.try_clone();
        }

        let tracker = self.tracker();
        let width = self.width();
        let height = self.height();
        Ok(match self {
            Self::F32(g) => {
                let up_width = width << factor;
                let up_height = height << factor;
                let mut out =
                    AlignedGrid::with_alloc_tracker(up_width, up_height, tracker.as_ref())?;

                let original = g.buf();
                let target = out.buf_mut();
                inner(original, target, width, height, factor);
                Self::F32(out)
            }
            Self::I32(g) => {
                let up_width = width << factor;
                let up_height = height << factor;
                let mut out =
                    AlignedGrid::with_alloc_tracker(up_width, up_height, tracker.as_ref())?;

                let original = g.buf();
                let target = out.buf_mut();
                inner(original, target, width, height, factor);
                Self::I32(out)
            }
            Self::I16(g) => {
                let up_width = width << factor;
                let up_height = height << factor;
                let mut out =
                    AlignedGrid::with_alloc_tracker(up_width, up_height, tracker.as_ref())?;

                let original = g.buf();
                let target = out.buf_mut();
                inner(original, target, width, height, factor);
                Self::I16(out)
            }
        })
    }
}

#[derive(Debug)]
pub struct ImageWithRegion {
    buffer: Vec<ImageBuffer>,
    regions: Vec<(Region, ChannelShift)>,
    ct_done: bool,
    blend_done: bool,
    tracker: Option<AllocTracker>,
}

impl ImageWithRegion {
    pub(crate) fn new(tracker: Option<&AllocTracker>) -> Self {
        Self {
            buffer: Vec::new(),
            regions: Vec::new(),
            ct_done: false,
            blend_done: false,
            tracker: tracker.cloned(),
        }
    }

    pub(crate) fn try_clone(&self) -> Result<Self> {
        Ok(Self {
            buffer: self
                .buffer
                .iter()
                .map(|x| x.try_clone())
                .collect::<std::result::Result<_, _>>()?,
            regions: self.regions.clone(),
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
    pub fn buffer(&self) -> &[ImageBuffer] {
        &self.buffer
    }

    #[inline]
    pub fn buffer_mut(&mut self) -> &mut [ImageBuffer] {
        &mut self.buffer
    }

    #[inline]
    pub fn take_buffer(&mut self) -> Vec<ImageBuffer> {
        std::mem::take(&mut self.buffer)
    }

    #[inline]
    pub fn regions_and_shifts(&self) -> &[(Region, ChannelShift)] {
        &self.regions
    }

    #[inline]
    pub fn append_channel(&mut self, buffer: ImageBuffer, region: Region) {
        assert_eq!(buffer.width(), region.width as usize);
        assert_eq!(buffer.height(), region.height as usize);
        self.buffer.push(buffer);
        self.regions.push((region, ChannelShift::from_shift(0)));
    }

    #[inline]
    pub fn append_channel_shifted(
        &mut self,
        buffer: ImageBuffer,
        original_region: Region,
        shift: ChannelShift,
    ) {
        let (width, height) = shift.shift_size((original_region.width, original_region.height));
        assert_eq!(buffer.width(), width as usize);
        assert_eq!(buffer.height(), height as usize);
        self.buffer.push(buffer);
        self.regions.push((original_region, shift));
    }

    pub fn extend_from_gmodular<S: Sample>(&mut self, gmodular: GlobalModular<S>) {
        let Some(image) = gmodular.modular.into_image() else {
            return;
        };
        for g in image.into_image_channels() {
            let width = g.width();
            let height = g.height();
            let region = Region::with_size(width as u32, height as u32);
            let buffer = ImageBuffer::from_modular_channel(g);
            self.append_channel(buffer, region);
        }
    }

    pub(crate) fn clone_gray(&mut self) -> Result<()> {
        let gray = self.buffer[0].try_clone()?;
        self.buffer.insert(1, gray.try_clone()?);
        self.buffer.insert(2, gray);
        Ok(())
    }

    pub(crate) fn convert_modular_color(&mut self, bit_depth: BitDepth) -> Result<()> {
        let [a, b, c, ..] = &mut *self.buffer else {
            panic!()
        };
        a.convert_to_float_modular(bit_depth)?;
        b.convert_to_float_modular(bit_depth)?;
        c.convert_to_float_modular(bit_depth)?;
        Ok(())
    }

    pub(crate) fn convert_modular_xyb(
        &mut self,
        lf_dequant: &LfChannelDequantization,
    ) -> Result<()> {
        let [y, x, b, ..] = &mut *self.buffer else {
            panic!()
        };
        ImageBuffer::convert_to_float_modular_xyb([y, x, b], lf_dequant)?;
        Ok(())
    }

    pub(crate) fn upsample_lf(&self, lf_level: u32) -> Result<Self> {
        let factor = lf_level * 3;
        let mut out = Self::new(self.tracker.as_ref());
        for (&(region, shift), buffer) in self.regions.iter().zip(&self.buffer) {
            let up_region = region.upsample(factor);
            let buffer = buffer.upsample_nn(factor)?;
            out.append_channel_shifted(buffer, up_region, shift);
        }
        Ok(out)
    }

    pub(crate) fn upsample_nonseparable(
        &mut self,
        image_header: &ImageHeader,
        frame_header: &FrameHeader,
    ) -> Result<()> {
        if frame_header.upsampling != 1 && self.buffer[0].as_float().is_none() {
            debug_assert!(!image_header.metadata.xyb_encoded);
        }

        for (idx, ((region, _), g)) in self.regions.iter_mut().zip(&mut self.buffer).enumerate() {
            let g = if let Some(ec_idx) = idx.checked_sub(3) {
                if image_header.metadata.ec_info[ec_idx].dim_shift == 0
                    && frame_header.ec_upsampling[ec_idx] == 1
                {
                    continue;
                }
                g.convert_to_float_modular(image_header.metadata.ec_info[ec_idx].bit_depth)?
            } else {
                if frame_header.upsampling == 1 {
                    continue;
                }
                g.convert_to_float_modular(image_header.metadata.bit_depth)?
            };
            crate::features::upsample(g, region, image_header, frame_header, idx)?;
        }

        Ok(())
    }

    #[inline]
    pub(crate) fn remove_color_channels(&mut self, count: usize) {
        self.buffer.drain(count..3);
        self.regions.drain(count..3);
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
}

impl ImageWithRegion {
    pub(crate) fn as_color_floats(&self) -> [&AlignedGrid<f32>; 3] {
        let [a, b, c, ..] = &*self.buffer else {
            panic!()
        };
        let a = a.as_float().unwrap();
        let b = b.as_float().unwrap();
        let c = c.as_float().unwrap();
        [a, b, c]
    }

    pub(crate) fn as_color_floats_mut(&mut self) -> [&mut AlignedGrid<f32>; 3] {
        let [a, b, c, ..] = &mut *self.buffer else {
            panic!()
        };
        let a = a.as_float_mut().unwrap();
        let b = b.as_float_mut().unwrap();
        let c = c.as_float_mut().unwrap();
        [a, b, c]
    }

    pub(crate) fn color_groups_with_group_id(
        &mut self,
        frame_header: &jxl_frame::FrameHeader,
    ) -> Vec<(u32, [MutableSubgrid<'_, f32>; 3])> {
        let [fb_x, fb_y, fb_b, ..] = &mut *self.buffer else {
            panic!();
        };

        let group_dim = frame_header.group_dim();
        let base_group_x = self.regions[0].0.left as u32 / group_dim;
        let base_group_y = self.regions[0].0.top as u32 / group_dim;
        let width = self.regions[0].0.width;
        let frame_groups_per_row = frame_header.groups_per_row();
        let groups_per_row = (width + group_dim - 1) / group_dim;

        let [fb_x, fb_y, fb_b] = [(0usize, fb_x), (1, fb_y), (2, fb_b)].map(|(idx, fb)| {
            let fb = fb.as_float_mut().unwrap().as_subgrid_mut();
            let hshift = self.regions[idx].1.hshift();
            let vshift = self.regions[idx].1.vshift();
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
            return Ok(BlendResult(grid_lock));
        }

        if !grid.ct_done() {
            let bit_depth = self.image.frame.header().bit_depth;
            grid.convert_modular_color(bit_depth)?;
            let ct_done = util::convert_color_for_record(
                image_header,
                frame_header.do_ycbcr,
                grid.as_color_floats_mut(),
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
