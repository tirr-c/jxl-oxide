use std::{ptr::NonNull, ops::RangeBounds};

use crate::{SimdVector, SharedSubgrid};

const fn compute_align<S>() -> usize {
    let base_align = std::mem::align_of::<S>();
    let min_align = if cfg!(target_arch = "x86_64") {
        32usize
    } else {
        1usize
    };

    if base_align > min_align {
        base_align
    } else {
        min_align
    }
}

/// A continuous buffer in the "raster order".
///
/// The buffer is aligned so that it can be used in SIMD instructions.
#[derive(Debug, Clone)]
pub struct SimpleGrid<S> {
    width: usize,
    height: usize,
    offset: usize,
    buf: Vec<S>,
}

impl<S: Default + Clone> SimpleGrid<S> {
    const ALIGN: usize = compute_align::<S>();

    /// Create a new buffer.
    pub fn new(width: usize, height: usize) -> Self {
        let len = width * height;
        let mut buf = Vec::with_capacity(len + Self::ALIGN / std::mem::size_of::<S>());

        let extra = buf.as_ptr() as usize & (Self::ALIGN - 1);
        let offset = ((Self::ALIGN - extra) % Self::ALIGN) / std::mem::size_of::<S>();
        buf.resize(len + offset, S::default());
        Self {
            width,
            height,
            offset,
            buf,
        }
    }
}

impl<S> SimpleGrid<S> {
    #[inline]
    pub fn width(&self) -> usize {
        self.width
    }

    #[inline]
    pub fn height(&self) -> usize {
        self.height
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
    pub(crate) fn into_buf_iter(self) -> impl Iterator<Item = S> {
        self.buf.into_iter().skip(self.offset)
    }

    #[inline]
    pub fn subgrid(&self, range_x: impl RangeBounds<usize>, range_y: impl RangeBounds<usize>) -> crate::SharedSubgrid<'_, S> {
        SharedSubgrid::from(self).subgrid(range_x, range_y)
    }
}

/// A mutable subgrid of the underlying buffer.
#[derive(Debug)]
pub struct CutGrid<'g, V: Copy = f32> {
    ptr: NonNull<V>,
    width: usize,
    height: usize,
    stride: usize,
    _marker: std::marker::PhantomData<&'g mut [V]>,
}

impl<'g, V: Copy> CutGrid<'g, V> {
    /// Create a `CutGrid` from raw pointer to the buffer, width, height and stride.
    ///
    /// # Safety
    /// The area specified by `width`, `height` and `stride` must not overlap with other instances
    /// of `CutGrid`, and the memory region in the area must be valid.
    pub unsafe fn new(ptr: NonNull<V>, width: usize, height: usize, stride: usize) -> Self {
        Self {
            ptr,
            width,
            height,
            stride,
            _marker: Default::default(),
        }
    }

    /// Create a `CutGrid` from buffer slice, width, height and stride.
    ///
    /// # Panic
    /// Panics if:
    /// - either `width` or `height` is zero,
    /// - `width` is greater than `stride`,
    /// - or the area specified by `width`, `height` and `stride` is larger than `buf`.
    pub fn from_buf(buf: &'g mut [V], width: usize, height: usize, stride: usize) -> Self {
        assert!(width > 0);
        assert!(height > 0);
        assert!(width <= stride);
        assert!(buf.len() >= stride * (height - 1) + width);
        // SAFETY: We have unique access to `buf`, and the area is in bounds.
        unsafe {
            Self::new(
                NonNull::new(buf.as_mut_ptr()).unwrap(),
                width,
                height,
                stride,
            )
        }
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
    fn get_ptr(&self, x: usize, y: usize) -> *mut V {
        if x >= self.width || y >= self.height {
            panic!(
                "Coordinate out of range: ({}, {}) not in {}x{}",
                x, y, self.width, self.height
            );
        }

        // SAFETY: (x, y) is checked above and is in bounds.
        unsafe {
            let offset = y * self.stride + x;
            self.ptr.as_ptr().add(offset)
        }
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> V {
        let ptr = self.get_ptr(x, y);
        // SAFETY: get_ptr returns a valid pointer.
        unsafe { *ptr }
    }

    #[inline]
    pub fn get_row(&self, row: usize) -> &[V] {
        let ptr = self.get_ptr(0, row);
        unsafe { std::slice::from_raw_parts(ptr as *const _, self.width) }
    }

    #[inline]
    pub fn get_mut(&mut self, x: usize, y: usize) -> &mut V {
        let ptr = self.get_ptr(x, y);
        // SAFETY: get_ptr returns a valid pointer, and mutable borrow of `self` makes sure that
        // the access is exclusive.
        unsafe { ptr.as_mut().unwrap() }
    }

    #[inline]
    pub fn get_row_mut(&mut self, row: usize) -> &mut [V] {
        let ptr = self.get_ptr(0, row);
        unsafe { std::slice::from_raw_parts_mut(ptr, self.width) }
    }

    #[inline]
    pub fn swap(&mut self, (ax, ay): (usize, usize), (bx, by): (usize, usize)) {
        let a = self.get_ptr(ax, ay);
        let b = self.get_ptr(bx, by);
        if std::ptr::eq(a, b) {
            return;
        }

        // SAFETY: `a` and `b` are valid and aligned.
        unsafe {
            std::ptr::swap(a, b);
        }
    }
}

impl<V: Copy> CutGrid<'_, V> {
    /// Split the grid horizontally at an index.
    ///
    /// # Panics
    /// Panics if `x > self.width()`.
    pub fn split_horizontal(&mut self, x: usize) -> (CutGrid<'_, V>, CutGrid<'_, V>) {
        assert!(x <= self.width);

        let left_ptr = self.ptr;
        let right_ptr = NonNull::new(self.get_ptr(x, 0)).unwrap();
        // SAFETY: two grids are contained in `self` and disjoint.
        unsafe {
            let left_grid = CutGrid::new(left_ptr, x, self.height, self.stride);
            let right_grid = CutGrid::new(right_ptr, self.width - x, self.height, self.stride);
            (left_grid, right_grid)
        }
    }

    /// Split the grid vertically at an index.
    ///
    /// # Panics
    /// Panics if `y > self.height()`.
    pub fn split_vertical(&mut self, y: usize) -> (CutGrid<'_, V>, CutGrid<'_, V>) {
        assert!(y <= self.height);

        let top_ptr = self.ptr;
        let bottom_ptr = NonNull::new(self.get_ptr(0, y)).unwrap();
        // SAFETY: two grids are contained in `self` and disjoint.
        unsafe {
            let top_grid = CutGrid::new(top_ptr, self.width, y, self.stride);
            let bottom_grid = CutGrid::new(bottom_ptr, self.width, self.height - y, self.stride);
            (top_grid, bottom_grid)
        }
    }
}

impl<'g> CutGrid<'g, f32> {
    pub fn as_vectored<V: SimdVector>(&mut self) -> Option<CutGrid<'_, V>> {
        assert!(V::available(), "Vector type `{}` is not supported by current CPU", std::any::type_name::<V>());

        let mask = V::SIZE - 1;
        let align_mask = std::mem::align_of::<V>() - 1;

        (self.ptr.as_ptr() as usize & align_mask == 0
            && self.width & mask == 0
            && self.stride & mask == 0)
            .then(|| CutGrid {
                ptr: self.ptr.cast::<V>(),
                width: self.width / V::SIZE,
                height: self.height,
                stride: self.stride / V::SIZE,
                _marker: Default::default(),
            })
    }
}

/// `[SimpleGrid]` with padding.
#[derive(Debug, Clone)]
pub struct PaddedGrid<S: Clone> {
    pub grid: SimpleGrid<S>,
    padding: usize,
}

impl<S: Default + Clone> PaddedGrid<S> {
    /// Create a new buffer.
    pub fn new(width: usize, height: usize, padding: usize) -> Self {
        Self {
            grid: SimpleGrid::new(width + padding * 2, height + padding * 2),
            padding,
        }
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
        for (out_row, in_row) in out_chunk.chunks_exact_mut(stride).zip(in_chunk.chunks_exact(stride).rev()) {
            out_row.clone_from_slice(in_row);
        }

        let (in_chunk, out_chunk) = buf.split_at_mut(stride * (height + padding));
        for (out_row, in_row) in out_chunk.chunks_exact_mut(stride).zip(in_chunk.chunks_exact(stride).rev()) {
            out_row.clone_from_slice(in_row);
        }
    }
}
