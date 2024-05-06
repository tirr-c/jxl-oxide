use jxl_grid::AlignedGrid;

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
    pub fn from_grids(grids: &[&AlignedGrid<f32>], orientation: u32) -> Self {
        let channels = grids.len();
        if channels == 0 {
            panic!("framebuffer should have channels");
        }
        if !(1..=8).contains(&orientation) {
            panic!("Invalid orientation {orientation}");
        }

        let mut width = grids[0].width();
        let mut height = grids[0].height();
        for g in grids {
            width = width.min(g.width());
            height = height.min(g.height());
        }

        let (outw, outh) = match orientation {
            1..=4 => (width, height),
            5..=8 => (height, width),
            _ => unreachable!(),
        };
        let mut out = Self::new(outw, outh, channels);
        let buf = out.buf_mut();
        for y in 0..height {
            for x in 0..width {
                for (c, g) in grids.iter().enumerate() {
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
                    buf[c + (outx + outy * outw) * channels] = g.get(x, y).copied().unwrap_or(0.0);
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
