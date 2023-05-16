use jxl_grid::SimpleGrid;

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

    pub fn from_grids(grids: &[SimpleGrid<f32>], orientation: u32) -> Result<Self> {
        let channels = grids.len();
        if channels == 0 {
            panic!("framebuffer should have channels");
        }
        if !(1..=8).contains(&orientation) {
            panic!("Invalid orientation {orientation}");
        }

        let width = grids[0].width();
        let height = grids[0].height();
        if !grids.iter().all(|g| g.width() >= width && g.height() >= height) {
            return Err(Error::GridSizeMismatch);
        }

        let (outw, outh) = match orientation {
            1..=4 => (width, height),
            5..=8 => (height, width),
            _ => unreachable!(),
        };
        let mut buf = vec![0.0f32; width * height * channels];
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

        Ok(Self {
            width: outw,
            height: outh,
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
