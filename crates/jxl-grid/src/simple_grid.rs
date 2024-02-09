use std::{ops::RangeBounds, ptr::NonNull};

use crate::{AllocHandle, AllocTracker, Error, SharedSubgrid, SimdVector};

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
#[derive(Debug)]
pub struct SimpleGrid<S> {
    width: usize,
    height: usize,
    offset: usize,
    buf: Vec<S>,
    handle: Option<AllocHandle>,
}

impl<S: Default + Clone> SimpleGrid<S> {
    const ALIGN: usize = compute_align::<S>();

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

    /// Clones the buffer without recording an allocation.
    pub fn clone_untracked(&self) -> Self {
        let mut out = Self::with_alloc_tracker(self.width, self.height, None).unwrap();
        out.buf_mut().clone_from_slice(self.buf());
        out
    }

    /// Tries to clone the buffer, and records the allocation in the same tracker as the original
    /// buffer.
    pub fn try_clone(&self) -> Result<Self, Error> {
        let mut out = Self::with_alloc_tracker(self.width, self.height, self.tracker().as_ref())?;
        out.buf_mut().clone_from_slice(self.buf());
        Ok(out)
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
    pub fn subgrid(
        &self,
        range_x: impl RangeBounds<usize>,
        range_y: impl RangeBounds<usize>,
    ) -> crate::SharedSubgrid<'_, S> {
        SharedSubgrid::from(self).subgrid(range_x, range_y)
    }
}

/// A mutable subgrid of the underlying buffer.
#[derive(Debug)]
pub struct CutGrid<'g, V: Copy = f32> {
    ptr: NonNull<V>,
    split_base: Option<NonNull<()>>,
    width: usize,
    height: usize,
    stride: usize,
    _marker: std::marker::PhantomData<&'g mut [V]>,
}

unsafe impl<'g, V: Copy> Send for CutGrid<'g, V> where &'g mut [V]: Send {}
unsafe impl<'g, V: Copy> Sync for CutGrid<'g, V> where &'g mut [V]: Sync {}

impl<'g, V: Copy> CutGrid<'g, V> {
    /// Create a `CutGrid` from raw pointer to the buffer, width, height and stride.
    ///
    /// # Safety
    /// The area specified by `width`, `height` and `stride` must not overlap with other instances
    /// of `CutGrid`, and the memory region in the area must be valid.
    ///
    /// # Panics
    /// Panics if `width > stride`.
    pub unsafe fn new(ptr: NonNull<V>, width: usize, height: usize, stride: usize) -> Self {
        assert!(width == 0 || width <= stride);
        Self {
            ptr,
            split_base: None,
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
    /// - `width` is greater than `stride`,
    /// - or the area specified by `width`, `height` and `stride` is larger than `buf`.
    pub fn from_buf(buf: &'g mut [V], width: usize, height: usize, stride: usize) -> Self {
        assert!(width <= stride);
        if width == 0 || height == 0 {
            assert_eq!(buf.len(), 0);
        } else {
            assert!(buf.len() >= stride * (height - 1) + width);
        }
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

    pub fn from_simple_grid(grid: &'g mut SimpleGrid<V>) -> Self {
        let width = grid.width();
        let height = grid.height();
        Self::from_buf(grid.buf_mut(), width, height, width)
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
    pub fn stride(&self) -> usize {
        self.stride
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
        unsafe { self.get_ptr_unchecked(x, y) }
    }

    #[inline]
    unsafe fn get_ptr_unchecked(&self, x: usize, y: usize) -> *mut V {
        let offset = y * self.stride + x;
        self.ptr.as_ptr().add(offset)
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

impl<'g, V: Copy> CutGrid<'g, V> {
    pub fn subgrid_mut(
        &mut self,
        range_x: impl RangeBounds<usize>,
        range_y: impl RangeBounds<usize>,
    ) -> CutGrid<V> {
        use std::ops::Bound;

        let left = match range_x.start_bound() {
            Bound::Included(&v) => v,
            Bound::Excluded(&v) => v + 1,
            Bound::Unbounded => 0,
        };
        let right = match range_x.end_bound() {
            Bound::Included(&v) => v + 1,
            Bound::Excluded(&v) => v,
            Bound::Unbounded => self.width,
        };
        let top = match range_y.start_bound() {
            Bound::Included(&v) => v,
            Bound::Excluded(&v) => v + 1,
            Bound::Unbounded => 0,
        };
        let bottom = match range_y.end_bound() {
            Bound::Included(&v) => v + 1,
            Bound::Excluded(&v) => v,
            Bound::Unbounded => self.height,
        };

        // Bounds checks.
        assert!(left <= right);
        assert!(top <= bottom);
        assert!(right <= self.width);
        assert!(bottom <= self.height);

        let base_ptr = NonNull::new(self.get_ptr(left, top)).unwrap();
        // SAFETY: subgrid is contained in `self`.
        unsafe { CutGrid::new(base_ptr, right - left, bottom - top, self.stride) }
    }

    /// Split the grid horizontally at an index.
    ///
    /// # Panics
    /// Panics if `x > self.width()`.
    pub fn split_horizontal(&mut self, x: usize) -> (CutGrid<'_, V>, CutGrid<'_, V>) {
        assert!(x <= self.width);

        let left_ptr = self.ptr;
        let right_ptr = NonNull::new(unsafe { self.get_ptr_unchecked(x, 0) }).unwrap();
        // SAFETY: two grids are contained in `self` and disjoint.
        unsafe {
            let split_base = self.split_base.unwrap_or(self.ptr.cast());
            let mut left_grid = CutGrid::new(left_ptr, x, self.height, self.stride);
            let mut right_grid = CutGrid::new(right_ptr, self.width - x, self.height, self.stride);
            left_grid.split_base = Some(split_base);
            right_grid.split_base = Some(split_base);
            (left_grid, right_grid)
        }
    }

    /// Split the grid horizontally at an index in-place.
    ///
    /// # Panics
    /// Panics if `x > self.width()`.
    pub fn split_horizontal_in_place(&mut self, x: usize) -> CutGrid<'g, V> {
        assert!(x <= self.width);

        let right_width = self.width - x;
        let right_ptr = NonNull::new(unsafe { self.get_ptr_unchecked(x, 0) }).unwrap();
        // SAFETY: two grids are contained in `self` and disjoint.
        unsafe {
            let split_base = self.split_base.unwrap_or(self.ptr.cast());
            self.width = x;
            self.split_base = Some(split_base);
            let mut right_grid = CutGrid::new(right_ptr, right_width, self.height, self.stride);
            right_grid.split_base = Some(split_base);
            right_grid
        }
    }

    /// Split the grid vertically at an index.
    ///
    /// # Panics
    /// Panics if `y > self.height()`.
    pub fn split_vertical(&mut self, y: usize) -> (CutGrid<'_, V>, CutGrid<'_, V>) {
        assert!(y <= self.height);

        let top_ptr = self.ptr;
        let bottom_ptr = NonNull::new(unsafe { self.get_ptr_unchecked(0, y) }).unwrap();
        // SAFETY: two grids are contained in `self` and disjoint.
        unsafe {
            let split_base = self.split_base.unwrap_or(self.ptr.cast());
            let mut top_grid = CutGrid::new(top_ptr, self.width, y, self.stride);
            let mut bottom_grid =
                CutGrid::new(bottom_ptr, self.width, self.height - y, self.stride);
            top_grid.split_base = Some(split_base);
            bottom_grid.split_base = Some(split_base);
            (top_grid, bottom_grid)
        }
    }

    /// Split the grid vertically at an index in-place.
    ///
    /// # Panics
    /// Panics if `y > self.height()`.
    pub fn split_vertical_in_place(&mut self, y: usize) -> CutGrid<'g, V> {
        assert!(y <= self.height);

        let bottom_height = self.height - y;
        let bottom_ptr = NonNull::new(unsafe { self.get_ptr_unchecked(0, y) }).unwrap();
        // SAFETY: two grids are contained in `self` and disjoint.
        unsafe {
            let split_base = self.split_base.unwrap_or(self.ptr.cast());
            self.height = y;
            self.split_base = Some(split_base);
            let mut bottom_grid = CutGrid::new(bottom_ptr, self.width, bottom_height, self.stride);
            bottom_grid.split_base = Some(split_base);
            bottom_grid
        }
    }

    pub fn merge_horizontal_in_place(&mut self, right: Self) {
        assert!(self.split_base.is_some());
        assert_eq!(self.split_base, right.split_base);
        assert_eq!(self.stride, right.stride);
        assert_eq!(self.height, right.height);
        assert!(self.stride >= self.width + right.width);
        unsafe {
            assert!(std::ptr::eq(
                self.get_ptr_unchecked(self.width, 0) as *const _,
                right.ptr.as_ptr() as *const _,
            ));
        }

        let right_width = right.width;
        self.width += right_width;
    }

    pub fn merge_vertical_in_place(&mut self, bottom: Self) {
        assert!(self.split_base.is_some());
        assert_eq!(self.split_base, bottom.split_base);
        assert_eq!(self.stride, bottom.stride);
        assert_eq!(self.width, bottom.width);
        unsafe {
            assert!(std::ptr::eq(
                self.get_ptr_unchecked(0, self.height) as *const _,
                bottom.ptr.as_ptr() as *const _,
            ));
        }

        let bottom_height = bottom.height;
        self.height += bottom_height;
    }
}

impl<'g, V: Copy> CutGrid<'g, V> {
    pub fn into_groups(self, group_width: usize, group_height: usize) -> Vec<CutGrid<'g, V>> {
        let CutGrid {
            ptr,
            split_base,
            width,
            height,
            stride,
            ..
        } = self;
        let split_base = split_base.unwrap_or(ptr.cast());

        let groups_x = (width + group_width - 1) / group_width;
        let groups_y = (height + group_height - 1) / group_height;
        let mut groups = Vec::with_capacity(groups_x * groups_y);
        for gy in 0..groups_y {
            let y = gy * group_height;
            let gh = (height - y).min(group_height);
            let row_ptr = unsafe { ptr.as_ptr().add(y * stride) };
            for gx in 0..groups_x {
                let x = gx * group_width;
                let gw = (width - x).min(group_width);
                let ptr = unsafe { row_ptr.add(x) };

                let mut grid = unsafe { CutGrid::new(NonNull::new(ptr).unwrap(), gw, gh, stride) };
                grid.split_base = Some(split_base);
                groups.push(grid);
            }
        }

        groups
    }
}

impl<'g> CutGrid<'g, f32> {
    pub fn as_vectored<V: SimdVector>(&mut self) -> Option<CutGrid<'_, V>> {
        assert!(
            V::available(),
            "Vector type `{}` is not supported by current CPU",
            std::any::type_name::<V>()
        );

        let mask = V::SIZE - 1;
        let align_mask = std::mem::align_of::<V>() - 1;

        (self.ptr.as_ptr() as usize & align_mask == 0
            && self.width & mask == 0
            && self.stride & mask == 0)
            .then(|| CutGrid {
                ptr: self.ptr.cast::<V>(),
                split_base: self.split_base,
                width: self.width / V::SIZE,
                height: self.height,
                stride: self.stride / V::SIZE,
                _marker: Default::default(),
            })
    }
}

/// `[SimpleGrid]` with padding.
#[derive(Debug)]
pub struct PaddedGrid<S: Clone> {
    pub grid: SimpleGrid<S>,
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
            grid: SimpleGrid::with_alloc_tracker(
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
