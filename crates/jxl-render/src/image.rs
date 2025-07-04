use std::sync::Arc;

use jxl_frame::{FrameHeader, data::GlobalModular};
use jxl_grid::{AlignedGrid, AllocTracker, MutableSubgrid};
use jxl_image::{BitDepth, ImageHeader};
use jxl_modular::{ChannelShift, Sample};
use jxl_threadpool::JxlThreadPool;
use jxl_vardct::LfChannelDequantization;

use crate::{FrameRender, FrameRenderHandle, IndexedFrame, Region, Result, util};

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
                    *b = b.saturating_add(y);
                }
            }
            [Self::I16(y), Self::I16(_), Self::I16(b)] => {
                for (b, &y) in b.buf_mut().iter_mut().zip(y.buf()) {
                    *b = b.saturating_add(y);
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
    color_channels: usize,
    ct_done: bool,
    blend_done: bool,
    tracker: Option<AllocTracker>,
}

impl ImageWithRegion {
    pub(crate) fn new(color_channels: usize, tracker: Option<&AllocTracker>) -> Self {
        Self {
            buffer: Vec::new(),
            regions: Vec::new(),
            color_channels,
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
            color_channels: self.color_channels,
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

    #[inline]
    pub fn replace_channel(&mut self, index: usize, buffer: ImageBuffer, region: Region) {
        assert_eq!(buffer.width(), region.width as usize);
        assert_eq!(buffer.height(), region.height as usize);
        self.buffer[index] = buffer;
        self.regions[index] = (region, ChannelShift::from_shift(0));
    }

    #[inline]
    pub fn replace_channel_shifted(
        &mut self,
        index: usize,
        buffer: ImageBuffer,
        original_region: Region,
        shift: ChannelShift,
    ) {
        let (width, height) = shift.shift_size((original_region.width, original_region.height));
        assert_eq!(buffer.width(), width as usize);
        assert_eq!(buffer.height(), height as usize);
        self.buffer[index] = buffer;
        self.regions[index] = (original_region, shift);
    }

    #[inline]
    pub(crate) fn swap_channel_f32(
        &mut self,
        index: usize,
        buffer: &mut AlignedGrid<f32>,
        region: Region,
    ) {
        assert_eq!(buffer.width(), region.width as usize);
        assert_eq!(buffer.height(), region.height as usize);
        let ImageBuffer::F32(original_buffer) = &mut self.buffer[index] else {
            panic!("original buffer is not F32");
        };
        std::mem::swap(original_buffer, buffer);
        self.regions[index] = (region, ChannelShift::from_shift(0));
    }

    pub fn extend_from_gmodular<S: Sample>(&mut self, gmodular: GlobalModular<S>) {
        let Some(image) = gmodular.modular.into_image() else {
            return;
        };
        for (g, info) in image.into_image_channels_with_info() {
            let (width, height) = info.original_size();
            let shift = info.shift();

            let original_region = Region::with_size(width, height);
            let buffer = ImageBuffer::from_modular_channel(g);
            self.append_channel_shifted(buffer, original_region, shift);
        }
    }

    pub(crate) fn clone_gray(&mut self) -> Result<()> {
        assert_eq!(self.color_channels, 1);

        let gray = self.buffer[0].try_clone()?;
        let region = self.regions[0];
        self.buffer.insert(1, gray.try_clone()?);
        self.regions.insert(1, region);
        self.buffer.insert(2, gray);
        self.regions.insert(2, region);

        self.color_channels = 3;
        Ok(())
    }

    pub(crate) fn convert_modular_color(&mut self, bit_depth: BitDepth) -> Result<()> {
        assert!(self.buffer.len() >= self.color_channels);
        for g in self.buffer.iter_mut().take(self.color_channels) {
            g.convert_to_float_modular(bit_depth)?;
        }
        Ok(())
    }

    pub(crate) fn convert_modular_xyb(
        &mut self,
        lf_dequant: &LfChannelDequantization,
    ) -> Result<()> {
        assert_eq!(self.color_channels, 3);
        let [y, x, b, ..] = &mut *self.buffer else {
            panic!()
        };
        ImageBuffer::convert_to_float_modular_xyb([y, x, b], lf_dequant)?;
        Ok(())
    }

    pub(crate) fn upsample_lf(&self, lf_level: u32) -> Result<Self> {
        let factor = lf_level * 3;
        let mut out = Self::new(self.color_channels, self.tracker.as_ref());
        for (&(region, shift), buffer) in self.regions.iter().zip(&self.buffer) {
            let up_region = region.upsample(factor);
            let buffer = buffer.upsample_nn(factor)?;
            out.append_channel_shifted(buffer, up_region, shift);
        }
        Ok(out)
    }

    pub(crate) fn upsample_jpeg(
        &mut self,
        valid_region: Region,
        bit_depth: BitDepth,
    ) -> Result<()> {
        assert_eq!(self.color_channels, 3);
        self.convert_modular_color(bit_depth)?;

        for (g, (region, shift)) in self.buffer.iter_mut().zip(&mut self.regions).take(3) {
            let downsampled_image_region = region.downsample_with_shift(*shift);
            let downsampled_valid_region = valid_region.downsample_with_shift(*shift);
            let left = downsampled_valid_region
                .left
                .abs_diff(downsampled_image_region.left);
            let top = downsampled_valid_region
                .top
                .abs_diff(downsampled_image_region.top);
            let width = downsampled_valid_region.width;
            let height = downsampled_valid_region.height;

            let subgrid = g.as_float().unwrap().as_subgrid().subgrid(
                left as usize..(left + width) as usize,
                top as usize..(top + height) as usize,
            );
            let out = crate::filter::apply_jpeg_upsampling_single(
                subgrid,
                *shift,
                valid_region,
                self.tracker.as_ref(),
            )?;

            *g = ImageBuffer::F32(out);
            *region = valid_region;
            *shift = ChannelShift::from_shift(0);
        }

        Ok(())
    }

    pub(crate) fn upsample_nonseparable(
        &mut self,
        image_header: &ImageHeader,
        frame_header: &FrameHeader,
        upsampled_valid_region: Region,
        ec_to_color_only: bool,
    ) -> Result<()> {
        if frame_header.upsampling != 1 && self.buffer[0].as_float().is_none() {
            debug_assert!(!image_header.metadata.xyb_encoded);
        }

        let color_channels = self.color_channels;
        let color_shift = frame_header.upsampling.trailing_zeros();

        for (idx, ((region, shift), g)) in self.regions.iter_mut().zip(&mut self.buffer).enumerate()
        {
            let tracker = g.tracker();
            let ChannelShift::Shifts(upsampling_factor) = *shift else {
                panic!("invalid channel shift for upsampling: {shift:?}");
            };
            let bit_depth = if let Some(ec_idx) = idx.checked_sub(color_channels) {
                image_header.metadata.ec_info[ec_idx].bit_depth
            } else {
                image_header.metadata.bit_depth
            };

            let target_factor = if ec_to_color_only { color_shift } else { 0 };
            if upsampling_factor == target_factor {
                continue;
            }
            let grid = g.convert_to_float_modular(bit_depth)?;

            let downsampled_image_region = region.downsample(upsampling_factor);
            let downsampled_valid_region = upsampled_valid_region.downsample(upsampling_factor);
            let left = downsampled_valid_region
                .left
                .abs_diff(downsampled_image_region.left);
            let top = downsampled_valid_region
                .top
                .abs_diff(downsampled_image_region.top);
            let width = downsampled_valid_region.width;
            let height = downsampled_valid_region.height;

            let subgrid = grid.as_subgrid().subgrid(
                left as usize..(left + width) as usize,
                top as usize..(top + height) as usize,
            );
            *region = downsampled_valid_region;
            let out = tracing::trace_span!(
                "Non-separable upsampling",
                channel_idx = idx,
                upsampling_factor,
                target_factor
            )
            .in_scope(|| {
                crate::features::upsample(
                    subgrid,
                    region,
                    image_header,
                    upsampling_factor - target_factor,
                    tracker.as_ref(),
                )
            })?;
            if let Some(out) = out {
                *g = ImageBuffer::F32(out);
            }
            *shift = ChannelShift::from_shift(target_factor);
        }

        Ok(())
    }

    pub(crate) fn prepare_color_upsampling(&mut self, frame_header: &FrameHeader) {
        let upsampling_factor = frame_header.upsampling.trailing_zeros();
        for (region, shift) in &mut self.regions {
            match shift {
                ChannelShift::Raw(..=-1, _) | ChannelShift::Raw(_, ..=-1) => continue,
                ChannelShift::Raw(h, v) => {
                    *h = h.wrapping_add_unsigned(upsampling_factor);
                    *v = v.wrapping_add_unsigned(upsampling_factor);
                }
                ChannelShift::Shifts(shift) => {
                    *shift += upsampling_factor;
                }
                ChannelShift::JpegUpsampling {
                    has_h_subsample: false,
                    h_subsample: false,
                    has_v_subsample: false,
                    v_subsample: false,
                } => {
                    *shift = ChannelShift::Shifts(upsampling_factor);
                }
                ChannelShift::JpegUpsampling { .. } => {
                    panic!("unexpected chroma subsampling {shift:?}");
                }
            }
            *region = region.upsample(upsampling_factor);
        }
    }

    #[inline]
    pub(crate) fn remove_color_channels(&mut self, count: usize) {
        assert!(self.color_channels >= count);
        self.buffer.drain(count..self.color_channels);
        self.regions.drain(count..self.color_channels);
        self.color_channels = count;
    }

    pub(crate) fn fill_opaque_alpha(&mut self, ec_info: &[jxl_image::ExtraChannelInfo]) {
        for (ec_idx, ec_info) in ec_info.iter().enumerate() {
            if !matches!(ec_info.ty, jxl_image::ExtraChannelType::Alpha { .. }) {
                continue;
            }

            let channel = &mut self.buffer[self.color_channels + ec_idx];
            let opaque_int = match ec_info.bit_depth {
                BitDepth::IntegerSample { bits_per_sample } => (1u32 << bits_per_sample) - 1,
                BitDepth::FloatSample {
                    bits_per_sample,
                    exp_bits,
                } => {
                    let mantissa_bits = bits_per_sample - exp_bits - 1;
                    ((1u32 << (exp_bits - 1)) - 1) << mantissa_bits
                }
            };
            match channel {
                ImageBuffer::I16(g) => {
                    g.buf_mut().fill(opaque_int as i16);
                }
                ImageBuffer::I32(g) => {
                    g.buf_mut().fill(opaque_int as i32);
                }
                ImageBuffer::F32(g) => {
                    g.buf_mut().fill(1.0);
                }
            }
        }
    }

    #[inline]
    pub fn color_channels(&self) -> usize {
        self.color_channels
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
        assert_eq!(self.color_channels, 3);
        let [a, b, c, ..] = &*self.buffer else {
            panic!()
        };
        let a = a.as_float().unwrap();
        let b = b.as_float().unwrap();
        let c = c.as_float().unwrap();
        [a, b, c]
    }

    pub(crate) fn as_color_floats_mut(&mut self) -> [&mut AlignedGrid<f32>; 3] {
        assert_eq!(self.color_channels, 3);
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
        assert_eq!(self.color_channels, 3);
        let [fb_x, fb_y, fb_b, ..] = &mut *self.buffer else {
            panic!();
        };

        let group_dim = frame_header.group_dim();
        let base_group_x = self.regions[0].0.left as u32 / group_dim;
        let base_group_y = self.regions[0].0.top as u32 / group_dim;
        let width = self.regions[0].0.width;
        let frame_groups_per_row = frame_header.groups_per_row();
        let groups_per_row = width.div_ceil(group_dim);

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
    pub(crate) fn blend(
        &self,
        oriented_image_region: Option<Region>,
        pool: &JxlThreadPool,
    ) -> Result<Arc<ImageWithRegion>> {
        let image_header = self.image.frame.image_header();
        let image_region = self.image.image_region;
        let oriented_image_region = oriented_image_region
            .unwrap_or_else(|| util::apply_orientation_to_image_region(image_header, image_region));

        let mut grid_lock = self.image.wait_until_render()?;
        if let FrameRender::Blended(image) = &*grid_lock {
            return Ok(Arc::clone(image));
        }

        let render = std::mem::replace(&mut *grid_lock, FrameRender::ErrTaken);
        let FrameRender::Done(mut grid) = render else {
            panic!();
        };

        let skip_composition = composite_preprocess(&self.image.frame, &mut grid, pool)?;
        if skip_composition {
            let image = Arc::new(grid);
            *grid_lock = FrameRender::Blended(Arc::clone(&image));
            return Ok(image);
        }

        *grid_lock = FrameRender::Rendering;
        drop(grid_lock);

        composite(
            &self.image.frame,
            &mut grid,
            self.image.refs.clone(),
            oriented_image_region,
            pool,
        )?;

        let image = Arc::new(grid);
        drop(
            self.image
                .done_render(FrameRender::Blended(Arc::clone(&image))),
        );
        Ok(image)
    }

    pub(crate) fn try_take_blended(&self) -> Option<ImageWithRegion> {
        let mut grid_lock = self.image.render.lock().unwrap();
        match std::mem::take(&mut *grid_lock) {
            FrameRender::Blended(image) => {
                let cloned_image = Arc::clone(&image);
                let image_inner = Arc::into_inner(cloned_image);
                if image_inner.is_none() {
                    *grid_lock = FrameRender::Blended(image);
                }
                image_inner
            }
            render => {
                *grid_lock = render;
                None
            }
        }
    }
}

/// Prerocess image before composition.
///
/// Returns `Ok(true)` if no actual composition is needed.
pub(crate) fn composite_preprocess(
    frame: &IndexedFrame,
    grid: &mut ImageWithRegion,
    pool: &JxlThreadPool,
) -> Result<bool> {
    let image_header = frame.image_header();
    let frame_header = frame.header();

    if frame_header.can_reference() {
        let bit_depth_it =
            std::iter::repeat_n(image_header.metadata.bit_depth, grid.color_channels)
                .chain(image_header.metadata.ec_info.iter().map(|ec| ec.bit_depth));
        for (buffer, bit_depth) in grid.buffer.iter_mut().zip(bit_depth_it) {
            buffer.convert_to_float_modular(bit_depth)?;
        }
    }

    let skip_blending = !frame_header.frame_type.is_normal_frame() || frame_header.resets_canvas;

    if !(grid.ct_done() || frame_header.save_before_ct || skip_blending && frame_header.is_last) {
        util::convert_color_for_record(image_header, frame_header.do_ycbcr, grid, pool)?;
    }

    if skip_blending {
        grid.blend_done = true;
    }

    Ok(skip_blending)
}

pub(crate) fn composite<S: Sample>(
    frame: &IndexedFrame,
    grid: &mut ImageWithRegion,
    refs: [Option<crate::Reference<S>>; 4],
    oriented_image_region: Region,
    pool: &JxlThreadPool,
) -> Result<()> {
    let image_header = frame.image_header();
    let frame_header = frame.header();
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

    let image = crate::blend::blend(image_header, refs, frame, grid, frame_region, pool)?;
    *grid = image;
    Ok(())
}
