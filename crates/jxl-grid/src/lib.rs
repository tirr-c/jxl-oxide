//! This crate provides [`AlignedGrid`] and [`PaddedGrid`], used in various places involving
//! images.
mod alloc_tracker;
mod mutable_subgrid;
mod shared_subgrid;
mod simd;
pub use alloc_tracker::*;
pub use mutable_subgrid::*;
pub use shared_subgrid::*;
pub use simd::SimdVector;

#[derive(Debug)]
pub enum Error {
    OutOfMemory(usize),
}

impl std::error::Error for Error {}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OutOfMemory(bytes) => write!(f, "failed to allocate {bytes} byte(s)"),
        }
    }
}

/// A continuous buffer in the "raster order".
///
/// The buffer is aligned so that it can be used in SIMD instructions.
pub struct AlignedGrid<S> {
    width: usize,
    height: usize,
    offset: usize,
    buf: Vec<S>,
    handle: Option<AllocHandle>,
}

impl<S> std::fmt::Debug for AlignedGrid<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AlignedGrid")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("offset", &self.offset)
            .finish_non_exhaustive()
    }
}

impl<S> AlignedGrid<S> {
    #[inline]
    pub fn empty() -> Self {
        Self {
            width: 0,
            height: 0,
            offset: 0,
            buf: Vec::new(),
            handle: None,
        }
    }
}

impl<S: Default + Clone> AlignedGrid<S> {
    const ALIGN: usize = 32;

    /// Create a new buffer, recording the allocation if a tracker is given.
    #[inline]
    pub fn with_alloc_tracker(
        width: usize,
        height: usize,
        tracker: Option<&AllocTracker>,
    ) -> Result<Self, Error> {
        let len = width * height;
        let buf_len = len + (Self::ALIGN - 1) / std::mem::size_of::<S>();
        let handle = tracker
            .map(|tracker| tracker.alloc::<S>(buf_len))
            .transpose()?;
        let mut buf = vec![S::default(); buf_len];

        let extra = buf.as_ptr() as usize & (Self::ALIGN - 1);
        let offset = ((Self::ALIGN - extra) % Self::ALIGN) / std::mem::size_of::<S>();
        buf.resize_with(len + offset, S::default);

        Ok(Self {
            width,
            height,
            offset,
            buf,
            handle,
        })
    }

    #[inline]
    fn empty_aligned(
        width: usize,
        height: usize,
        tracker: Option<&AllocTracker>,
    ) -> Result<Self, Error> {
        let len = width * height;
        let buf_len = len + (Self::ALIGN - 1) / std::mem::size_of::<S>();
        let handle = tracker
            .map(|tracker| tracker.alloc::<S>(buf_len))
            .transpose()?;
        let mut buf = Vec::with_capacity(buf_len);

        let extra = buf.as_ptr() as usize & (Self::ALIGN - 1);
        let offset = ((Self::ALIGN - extra) % Self::ALIGN) / std::mem::size_of::<S>();
        buf.resize_with(offset, S::default);

        Ok(Self {
            width,
            height,
            offset,
            buf,
            handle,
        })
    }

    /// Clones the buffer without recording an allocation.
    pub fn clone_untracked(&self) -> Self {
        let mut out = Self::empty_aligned(self.width, self.height, None).unwrap();
        out.buf.extend_from_slice(self.buf());
        out
    }

    /// Tries to clone the buffer, and records the allocation in the same tracker as the original
    /// buffer.
    pub fn try_clone(&self) -> Result<Self, Error> {
        let mut out = Self::empty_aligned(self.width, self.height, self.tracker().as_ref())?;
        out.buf.extend_from_slice(self.buf());
        Ok(out)
    }
}

impl<S> AlignedGrid<S> {
    #[inline]
    pub fn width(&self) -> usize {
        self.width
    }

    #[inline]
    pub fn height(&self) -> usize {
        self.height
    }

    #[inline]
    pub fn tracker(&self) -> Option<AllocTracker> {
        self.handle.as_ref().map(|handle| handle.tracker())
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> Option<&S> {
        if x >= self.width || y >= self.height {
            return None;
        }

        Some(&self.buf[y * self.width + x + self.offset])
    }

    #[inline]
    pub fn get_mut(&mut self, x: usize, y: usize) -> Option<&mut S> {
        if x >= self.width || y >= self.height {
            return None;
        }

        Some(&mut self.buf[y * self.width + x + self.offset])
    }

    #[inline]
    pub fn get_row(&self, y: usize) -> Option<&[S]> {
        if y >= self.height {
            return None;
        }

        Some(&self.buf[y * self.width + self.offset..][..self.width])
    }

    #[inline]
    pub fn get_row_mut(&mut self, y: usize) -> Option<&mut [S]> {
        if y >= self.height {
            return None;
        }

        Some(&mut self.buf[y * self.width + self.offset..][..self.width])
    }

    /// Get the immutable slice to the underlying buffer.
    #[inline]
    pub fn buf(&self) -> &[S] {
        &self.buf[self.offset..]
    }

    /// Get the mutable slice to the underlying buffer.
    #[inline]
    pub fn buf_mut(&mut self) -> &mut [S] {
        &mut self.buf[self.offset..]
    }

    #[inline]
    pub fn as_subgrid(&self) -> SharedSubgrid<S> {
        SharedSubgrid::from(self)
    }

    #[inline]
    pub fn as_subgrid_mut(&mut self) -> MutableSubgrid<S> {
        MutableSubgrid::from(self)
    }
}

/// `[AlignedGrid]` with padding.
#[derive(Debug)]
pub struct PaddedGrid<S: Clone> {
    pub grid: AlignedGrid<S>,
    padding: usize,
}

impl<S: Default + Clone> PaddedGrid<S> {
    /// Create a new buffer.
    pub fn with_alloc_tracker(
        width: usize,
        height: usize,
        padding: usize,
        tracker: Option<&AllocTracker>,
    ) -> Result<Self, crate::Error> {
        Ok(Self {
            grid: AlignedGrid::with_alloc_tracker(
                width + padding * 2,
                height + padding * 2,
                tracker,
            )?,
            padding,
        })
    }
}

impl<S: Clone> PaddedGrid<S> {
    #[inline]
    pub fn width(&self) -> usize {
        self.grid.width - self.padding * 2
    }

    #[inline]
    pub fn height(&self) -> usize {
        self.grid.height - self.padding * 2
    }

    #[inline]
    pub fn padding(&self) -> usize {
        self.padding
    }

    #[inline]
    pub fn buf_padded(&self) -> &[S] {
        self.grid.buf()
    }

    #[inline]
    pub fn buf_padded_mut(&mut self) -> &mut [S] {
        self.grid.buf_mut()
    }

    /// Use mirror operator on padding
    pub fn mirror_edges_padding(&mut self) {
        let padding = self.padding;
        let stride = self.grid.width();
        let height = self.grid.height() - padding * 2;

        // Mirror horizontally.
        let buf = self.grid.buf_mut();
        for y in padding..height + padding {
            for x in 0..padding {
                buf[y * stride + x] = buf[y * stride + padding * 2 - x - 1].clone();
                buf[(y + 1) * stride - x - 1] = buf[(y + 1) * stride - padding * 2 + x].clone();
            }
        }

        // Mirror vertically.
        let (out_chunk, in_chunk) = buf.split_at_mut(stride * padding);
        let in_chunk = &in_chunk[..stride * padding];
        for (out_row, in_row) in out_chunk
            .chunks_exact_mut(stride)
            .zip(in_chunk.chunks_exact(stride).rev())
        {
            out_row.clone_from_slice(in_row);
        }

        let (in_chunk, out_chunk) = buf.split_at_mut(stride * (height + padding));
        for (out_row, in_row) in out_chunk
            .chunks_exact_mut(stride)
            .zip(in_chunk.chunks_exact(stride).rev())
        {
            out_row.clone_from_slice(in_row);
        }
    }
}
