use jxl_image::BitDepth;
use jxl_render::{ImageBuffer, Region};
use private::Sealed;

/// Frame buffer representing a decoded image.
#[derive(Debug, Clone)]
pub struct FrameBuffer {
    width: usize,
    height: usize,
    channels: usize,
    buf: Vec<f32>,
}

impl FrameBuffer {
    /// Creates a new framebuffer with given dimension.
    ///
    /// Note that framebuffer allocations are not tracked.
    pub fn new(width: usize, height: usize, channels: usize) -> Self {
        Self {
            width,
            height,
            channels,
            buf: vec![0.0f32; width * height * channels],
        }
    }

    /// For internal use only.
    #[doc(hidden)]
    pub fn from_grids(
        grids: &[&ImageBuffer],
        bit_depth: &[BitDepth],
        grid_regions: &[Region],
        copy_region: Region,
        orientation: u32,
    ) -> Self {
        let channels = grids.len();
        if channels == 0 {
            panic!("framebuffer should have channels");
        }
        if !(1..=8).contains(&orientation) {
            panic!("Invalid orientation {orientation}");
        }

        let Region {
            left,
            top,
            width,
            height,
        } = copy_region;
        let width = width as usize;
        let height = height as usize;

        let (outw, outh) = match orientation {
            1..=4 => (width, height),
            5..=8 => (height, width),
            _ => unreachable!(),
        };
        let mut out = Self::new(outw, outh, channels);
        let buf = out.buf_mut();
        for y in 0..height {
            for x in 0..width {
                for (c, (g, region)) in grids.iter().zip(grid_regions).enumerate() {
                    let (outx, outy) = match orientation {
                        1 => (x, y),
                        2 => (width - x - 1, y),
                        3 => (width - x - 1, height - y - 1),
                        4 => (x, height - y - 1),
                        5 => (y, x),
                        6 => (height - y - 1, x),
                        7 => (height - y - 1, width - x - 1),
                        8 => (y, width - x - 1),
                        _ => unreachable!(),
                    };
                    let idx = c + (outx + outy * outw) * channels;

                    let base_x = (left - region.left) as isize;
                    let base_y = (top - region.top) as isize;
                    let Some(x) = x.checked_add_signed(base_x) else {
                        buf[idx] = 0.0;
                        continue;
                    };
                    let Some(y) = y.checked_add_signed(base_y) else {
                        buf[idx] = 0.0;
                        continue;
                    };
                    if x >= region.width as usize || y >= region.height as usize {
                        buf[idx] = 0.0;
                        continue;
                    }

                    buf[idx] = match g {
                        ImageBuffer::F32(g) => g.get(x, y).copied().unwrap_or(0.0),
                        ImageBuffer::I32(g) => {
                            bit_depth[c].parse_integer_sample(g.get(x, y).copied().unwrap_or(0))
                        }
                        ImageBuffer::I16(g) => bit_depth[c]
                            .parse_integer_sample(g.get(x, y).copied().unwrap_or(0) as i32),
                    };
                }
            }
        }

        out
    }

    /// Returns the width of the frame buffer.
    #[inline]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Returns the height of the frame buffer.
    #[inline]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Returns the number of channels of the frame buffer.
    #[inline]
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Returns the contents of frame buffer.
    ///
    /// The buffer has length of `width * height * channels`, where `n * channels + c`-th sample
    /// belongs to the `c`-th channel.
    #[inline]
    pub fn buf(&self) -> &[f32] {
        &self.buf
    }

    /// Returns the mutable reference to frame buffer.
    ///
    /// The buffer has length of `width * height * channels`, where `n * channels + c`-th sample
    /// belongs to the `c`-th channel.
    #[inline]
    pub fn buf_mut(&mut self) -> &mut [f32] {
        &mut self.buf
    }

    /// Returns the contents of frame buffer, grouped by pixels.
    ///
    /// # Panics
    /// Panics if `N != self.channels()`.
    #[inline]
    pub fn buf_grouped<const N: usize>(&self) -> &[[f32; N]] {
        let grouped_len = self.width * self.height;
        assert_eq!(self.buf.len(), grouped_len * N);
        // SAFETY: Arrays have size of size_of::<T> * N, alignment of T.
        // Buffer length is checked above.
        unsafe { std::slice::from_raw_parts(self.buf.as_ptr() as *const [f32; N], grouped_len) }
    }

    /// Returns the mutable reference to frame buffer, grouped by pixels.
    ///
    /// # Panics
    /// Panics if `N != self.channels()`.
    #[inline]
    pub fn buf_grouped_mut<const N: usize>(&mut self) -> &mut [[f32; N]] {
        let grouped_len = self.width * self.height;
        assert_eq!(self.buf.len(), grouped_len * N);
        // SAFETY: Arrays have size of size_of::<T> * N, alignment of T.
        // Buffer length is checked above.
        unsafe {
            std::slice::from_raw_parts_mut(self.buf.as_mut_ptr() as *mut [f32; N], grouped_len)
        }
    }
}

/// Image stream that writes to borrowed buffer.
pub struct ImageStream<'r> {
    orientation: u32,
    width: u32,
    height: u32,
    grids: Vec<&'r ImageBuffer>,
    start_offset_xy: Vec<(i32, i32)>,
    bit_depth: Vec<BitDepth>,
    spot_colors: Vec<ImageStreamSpotColor<'r>>,
    y: u32,
    x: u32,
    c: u32,
}

impl<'r> ImageStream<'r> {
    pub(crate) fn from_render(render: &'r crate::Render) -> Self {
        use jxl_image::ExtraChannelType;

        let orientation = render.orientation;
        assert!((1..=8).contains(&orientation));
        let Region {
            left,
            top,
            mut width,
            mut height,
        } = render.target_frame_region;
        if orientation >= 5 {
            std::mem::swap(&mut width, &mut height);
        }

        let fb = render.image.buffer();
        let color_channels = render.image.color_channels();
        let regions_and_shifts = render.image.regions_and_shifts();

        let mut grids: Vec<_> = render.color_channels().iter().collect();
        let mut bit_depth = vec![render.color_bit_depth; grids.len()];

        let mut start_offset_xy = Vec::new();
        for (region, _) in &regions_and_shifts[..color_channels] {
            start_offset_xy.push((left - region.left, top - region.top));
        }

        // Find black
        for (ec_idx, (ec, (region, _))) in render
            .extra_channels
            .iter()
            .zip(&regions_and_shifts[color_channels..])
            .enumerate()
        {
            if ec.is_black() {
                grids.push(&fb[color_channels + ec_idx]);
                bit_depth.push(ec.bit_depth);
                start_offset_xy.push((left - region.left, top - region.top));
                break;
            }
        }
        // Find alpha
        for (ec_idx, (ec, (region, _))) in render
            .extra_channels
            .iter()
            .zip(&regions_and_shifts[color_channels..])
            .enumerate()
        {
            if ec.is_alpha() {
                grids.push(&fb[color_channels + ec_idx]);
                bit_depth.push(ec.bit_depth);
                start_offset_xy.push((left - region.left, top - region.top));
                break;
            }
        }

        let mut spot_colors = Vec::new();
        if render.render_spot_color && color_channels == 3 {
            for (ec_idx, (ec, (region, _))) in render
                .extra_channels
                .iter()
                .zip(&regions_and_shifts[color_channels..])
                .enumerate()
            {
                if let ExtraChannelType::SpotColour {
                    red,
                    green,
                    blue,
                    solidity,
                } = ec.ty
                {
                    let grid = &fb[color_channels + ec_idx];
                    let xy = (left - region.left, top - region.top);
                    spot_colors.push(ImageStreamSpotColor {
                        grid,
                        start_offset_xy: xy,
                        bit_depth: ec.bit_depth,
                        rgb: (red, green, blue),
                        solidity,
                    });
                }
            }
        }

        ImageStream {
            orientation,
            width,
            height,
            grids,
            bit_depth,
            start_offset_xy,
            spot_colors,
            y: 0,
            x: 0,
            c: 0,
        }
    }
}

impl ImageStream<'_> {
    /// Returns width of the image.
    #[inline]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Returns height of the image.
    #[inline]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Returns the number of channels of the image.
    #[inline]
    pub fn channels(&self) -> u32 {
        self.grids.len() as u32
    }

    /// Writes next samples to the buffer, returning how many samples are written.
    pub fn write_to_buffer<Sample: FrameBufferSample>(&mut self, buf: &mut [Sample]) -> usize {
        let channels = self.grids.len() as u32;
        let mut buf_it = buf.iter_mut();
        let mut count = 0usize;
        'outer: while self.y < self.height {
            while self.x < self.width {
                while self.c < channels {
                    let Some(v) = buf_it.next() else {
                        break 'outer;
                    };
                    let (start_x, start_y) = self.start_offset_xy[self.c as usize];
                    let (orig_x, orig_y) = self.to_original_coord(self.x, self.y);
                    let (Some(x), Some(y)) = (
                        orig_x.checked_add_signed(start_x),
                        orig_y.checked_add_signed(start_y),
                    ) else {
                        *v = Sample::default();
                        count += 1;
                        self.c += 1;
                        continue;
                    };
                    let x = x as usize;
                    let y = y as usize;
                    let grid = &self.grids[self.c as usize];
                    let bit_depth = self.bit_depth[self.c as usize];

                    if self.c >= 3 || self.spot_colors.is_empty() {
                        v.copy_from_grid(grid, x, y, bit_depth);
                    } else {
                        let mut tmp_sample = 0f32;
                        tmp_sample.copy_from_grid(grid, x, y, bit_depth);

                        for spot in &self.spot_colors {
                            let ImageStreamSpotColor {
                                grid,
                                start_offset_xy: (start_x, start_y),
                                bit_depth,
                                rgb: (r, g, b),
                                solidity,
                            } = *spot;
                            let color = [r, g, b][self.c as usize];
                            let xy = (
                                orig_x.checked_add_signed(start_x),
                                orig_y.checked_add_signed(start_y),
                            );
                            let mix = if let (Some(x), Some(y)) = xy {
                                let x = x as usize;
                                let y = y as usize;
                                let mut val = 0f32;
                                val.copy_from_grid(grid, x, y, bit_depth);
                                val * solidity
                            } else {
                                0.0
                            };

                            tmp_sample = color * mix + tmp_sample * (1.0 - mix);
                        }

                        v.copy_from_f32(tmp_sample);
                    }

                    count += 1;
                    self.c += 1;
                }
                self.c = 0;
                self.x += 1;
            }
            self.x = 0;
            self.y += 1;
        }
        count
    }

    #[inline]
    fn to_original_coord(&self, x: u32, y: u32) -> (u32, u32) {
        let width = self.width;
        let height = self.height;
        match self.orientation {
            1 => (x, y),
            2 => (width - x - 1, y),
            3 => (width - x - 1, height - y - 1),
            4 => (x, height - y - 1),
            5 => (y, x),
            6 => (y, width - x - 1),
            7 => (height - y - 1, width - x - 1),
            8 => (height - y - 1, x),
            _ => unreachable!(),
        }
    }
}

struct ImageStreamSpotColor<'r> {
    grid: &'r ImageBuffer,
    start_offset_xy: (i32, i32),
    bit_depth: BitDepth,
    rgb: (f32, f32, f32),
    solidity: f32,
}

/// Marker trait for supported output sample types.
pub trait FrameBufferSample: private::Sealed {}

/// Output as 32-bit float samples, with nominal range of `[0, 1]`.
impl FrameBufferSample for f32 {}

/// Output as 16-bit unsigned integer samples.
impl FrameBufferSample for u16 {}

/// Output as 8-bit unsigned integer samples.
impl FrameBufferSample for u8 {}

mod private {
    use jxl_image::BitDepth;
    use jxl_render::ImageBuffer;

    pub trait Sealed: Sized + Default {
        fn copy_from_grid(&mut self, grid: &ImageBuffer, x: usize, y: usize, bit_depth: BitDepth);
        fn copy_from_f32(&mut self, val: f32);
    }

    impl Sealed for f32 {
        #[inline]
        fn copy_from_grid(&mut self, grid: &ImageBuffer, x: usize, y: usize, bit_depth: BitDepth) {
            *self = match grid {
                ImageBuffer::F32(g) => g.get(x, y).copied().unwrap_or(0.0),
                ImageBuffer::I32(g) => {
                    bit_depth.parse_integer_sample(g.get(x, y).copied().unwrap_or(0))
                }
                ImageBuffer::I16(g) => {
                    bit_depth.parse_integer_sample(g.get(x, y).copied().unwrap_or(0) as i32)
                }
            };
        }

        #[inline]
        fn copy_from_f32(&mut self, val: f32) {
            *self = val;
        }
    }

    impl Sealed for u16 {
        #[inline]
        fn copy_from_grid(&mut self, grid: &ImageBuffer, x: usize, y: usize, bit_depth: BitDepth) {
            if matches!(
                bit_depth,
                BitDepth::IntegerSample {
                    bits_per_sample: 16
                }
            ) {
                *self = match grid {
                    ImageBuffer::F32(g) => (g.get(x, y).copied().unwrap_or(0.0) * 65535.0 + 0.5)
                        .clamp(0.0, 65535.0) as u16,
                    ImageBuffer::I32(g) => g.get(x, y).copied().unwrap_or(0).clamp(0, 65535) as u16,
                    ImageBuffer::I16(g) => g.get(x, y).copied().unwrap_or(0).max(0) as u16,
                };
            } else {
                let flt = match grid {
                    ImageBuffer::F32(g) => g.get(x, y).copied().unwrap_or(0.0),
                    ImageBuffer::I32(g) => {
                        bit_depth.parse_integer_sample(g.get(x, y).copied().unwrap_or(0))
                    }
                    ImageBuffer::I16(g) => {
                        bit_depth.parse_integer_sample(g.get(x, y).copied().unwrap_or(0) as i32)
                    }
                };
                self.copy_from_f32(flt);
            }
        }

        #[inline]
        fn copy_from_f32(&mut self, val: f32) {
            *self = (val * 65535.0 + 0.5).clamp(0.0, 65535.0) as u16;
        }
    }

    impl Sealed for u8 {
        #[inline]
        fn copy_from_grid(&mut self, grid: &ImageBuffer, x: usize, y: usize, bit_depth: BitDepth) {
            if matches!(bit_depth, BitDepth::IntegerSample { bits_per_sample: 8 }) {
                *self = match grid {
                    ImageBuffer::F32(g) => {
                        (g.get(x, y).copied().unwrap_or(0.0) * 255.0 + 0.5).clamp(0.0, 255.0) as u8
                    }
                    ImageBuffer::I32(g) => g.get(x, y).copied().unwrap_or(0).clamp(0, 255) as u8,
                    ImageBuffer::I16(g) => g.get(x, y).copied().unwrap_or(0).clamp(0, 255) as u8,
                };
            } else {
                let flt = match grid {
                    ImageBuffer::F32(g) => g.get(x, y).copied().unwrap_or(0.0),
                    ImageBuffer::I32(g) => {
                        bit_depth.parse_integer_sample(g.get(x, y).copied().unwrap_or(0))
                    }
                    ImageBuffer::I16(g) => {
                        bit_depth.parse_integer_sample(g.get(x, y).copied().unwrap_or(0) as i32)
                    }
                };
                self.copy_from_f32(flt);
            }
        }

        #[inline]
        fn copy_from_f32(&mut self, val: f32) {
            *self = (val * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
        }
    }
}
