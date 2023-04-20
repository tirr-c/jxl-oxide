use jxl_grid::Grid;

use crate::{Error, Result};

#[derive(Debug, Clone)]
pub struct FrameBuffer {
    width: usize,
    height: usize,
    channels: usize,
    buf: Vec<f32>,
}

impl FrameBuffer {
    pub fn new(width: usize, height: usize, channels: usize) -> Self {
        Self {
            width,
            height,
            channels,
            buf: vec![0.0f32; width * height * channels],
        }
    }

    pub fn from_grids(grids: &[Grid<f32>]) -> Result<Self> {
        let channels = grids.len();
        if channels == 0 {
            panic!("framebuffer should have channels");
        }

        let width = grids[0].width();
        let height = grids[0].height();
        if !grids.iter().all(|g| g.width() >= width && g.height() >= height) {
            return Err(Error::GridSizeMismatch);
        }

        let mut buf = vec![0.0f32; width * height * channels];
        for y in 0..height {
            for x in 0..width {
                for (c, g) in grids.iter().enumerate() {
                    buf[c + (x + y * width) * channels] = g.get(x, y).copied().unwrap_or(0.0);
                }
            }
        }

        Ok(Self {
            width,
            height,
            channels,
            buf,
        })
    }

    #[inline]
    pub fn width(&self) -> usize {
        self.width
    }

    #[inline]
    pub fn height(&self) -> usize {
        self.height
    }

    #[inline]
    pub fn channels(&self) -> usize {
        self.channels
    }

    #[inline]
    pub fn buf(&self) -> &[f32] {
        &self.buf
    }

    #[inline]
    pub fn buf_grouped<const N: usize>(&self) -> &[[f32; N]] {
        let grouped_len = self.width * self.height;
        assert_eq!(self.buf.len(), grouped_len * N);
        // SAFETY: Arrays have size of size_of::<T> * N, alignment of T.
        // Buffer length is checked above.
        unsafe {
            std::slice::from_raw_parts(self.buf.as_ptr() as *const [f32; N], grouped_len)
        }
    }

    #[inline]
    pub fn buf_mut(&mut self) -> &mut [f32] {
        &mut self.buf
    }

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
