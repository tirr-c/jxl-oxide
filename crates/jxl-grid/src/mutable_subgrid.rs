use std::{ops::RangeBounds, ptr::NonNull};

use crate::{SharedSubgrid, SimdVector};

/// A mutable subgrid of the underlying buffer.
#[derive(Debug)]
pub struct MutableSubgrid<'g, V = f32> {
    ptr: NonNull<V>,
    split_base: Option<NonNull<()>>,
    width: usize,
    height: usize,
    stride: usize,
    _marker: std::marker::PhantomData<&'g mut [V]>,
}

unsafe impl<'g, V> Send for MutableSubgrid<'g, V> where &'g mut [V]: Send {}
unsafe impl<'g, V> Sync for MutableSubgrid<'g, V> where &'g mut [V]: Sync {}

impl<'g, V> From<&'g mut crate::AlignedGrid<V>> for MutableSubgrid<'g, V> {
    fn from(grid: &'g mut crate::AlignedGrid<V>) -> Self {
        let width = grid.width();
        let height = grid.height();
        Self::from_buf(grid.buf_mut(), width, height, width)
    }
}

impl<'g, V> MutableSubgrid<'g, V> {
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

    pub fn empty() -> Self {
        Self {
            ptr: NonNull::dangling(),
            split_base: None,
            width: 0,
            height: 0,
            stride: 0,
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

    #[inline]
    pub fn into_ptr(self) -> NonNull<V> {
        self.ptr
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
        unsafe { self.ptr.as_ptr().add(offset) }
    }

    #[inline]
    pub fn get_row(&self, row: usize) -> &[V] {
        assert!(
            row < self.height,
            "Row index out of range: height is {} but index is {}",
            self.height,
            row,
        );

        // SAFETY: row is in bounds, `width` consecutive elements from the start of a row is valid.
        unsafe {
            let offset = row * self.stride;
            let ptr = self.ptr.as_ptr().add(offset);
            std::slice::from_raw_parts(ptr as *const _, self.width)
        }
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
        assert!(
            row < self.height,
            "Row index out of range: height is {} but index is {}",
            self.height,
            row,
        );

        // SAFETY: row is in bounds, `width` consecutive elements from the start of a row is valid.
        unsafe {
            let offset = row * self.stride;
            let ptr = self.ptr.as_ptr().add(offset);
            std::slice::from_raw_parts_mut(ptr, self.width)
        }
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

impl<V: Copy> MutableSubgrid<'_, V> {
    #[inline]
    pub fn get(&self, x: usize, y: usize) -> V {
        let ptr = self.get_ptr(x, y);
        // SAFETY: get_ptr returns a valid pointer.
        unsafe { *ptr }
    }
}

impl<'g, V> MutableSubgrid<'g, V> {
    pub fn borrow_mut(&mut self) -> MutableSubgrid<V> {
        // SAFETY: We have unique reference to the grid, and the new grid borrows it.
        unsafe { MutableSubgrid::new(self.ptr, self.width, self.height, self.stride) }
    }

    pub fn as_shared(&self) -> SharedSubgrid<V> {
        // SAFETY: We have unique reference to the grid.
        unsafe { SharedSubgrid::new(self.ptr, self.width, self.height, self.stride) }
    }

    pub fn subgrid(
        self,
        range_x: impl RangeBounds<usize>,
        range_y: impl RangeBounds<usize>,
    ) -> MutableSubgrid<'g, V> {
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

        // SAFETY: subgrid region is contained in `self`.
        unsafe {
            let base_ptr = NonNull::new(self.get_ptr_unchecked(left, top)).unwrap();
            MutableSubgrid::new(base_ptr, right - left, bottom - top, self.stride)
        }
    }

    /// Split the grid horizontally at an index.
    ///
    /// # Panics
    /// Panics if `x > self.width()`.
    pub fn split_horizontal(&mut self, x: usize) -> (MutableSubgrid<'_, V>, MutableSubgrid<'_, V>) {
        assert!(x <= self.width);

        let left_ptr = self.ptr;
        let right_ptr = NonNull::new(unsafe { self.get_ptr_unchecked(x, 0) }).unwrap();
        // SAFETY: two grids are contained in `self` and disjoint.
        unsafe {
            let split_base = self.split_base.unwrap_or(self.ptr.cast());
            let mut left_grid = MutableSubgrid::new(left_ptr, x, self.height, self.stride);
            let mut right_grid =
                MutableSubgrid::new(right_ptr, self.width - x, self.height, self.stride);
            left_grid.split_base = Some(split_base);
            right_grid.split_base = Some(split_base);
            (left_grid, right_grid)
        }
    }

    /// Split the grid horizontally at an index in-place.
    ///
    /// # Panics
    /// Panics if `x > self.width()`.
    pub fn split_horizontal_in_place(&mut self, x: usize) -> MutableSubgrid<'g, V> {
        assert!(x <= self.width);

        let right_width = self.width - x;
        let right_ptr = NonNull::new(unsafe { self.get_ptr_unchecked(x, 0) }).unwrap();
        // SAFETY: two grids are contained in `self` and disjoint.
        unsafe {
            let split_base = self.split_base.unwrap_or(self.ptr.cast());
            self.width = x;
            self.split_base = Some(split_base);
            let mut right_grid =
                MutableSubgrid::new(right_ptr, right_width, self.height, self.stride);
            right_grid.split_base = Some(split_base);
            right_grid
        }
    }

    /// Split the grid vertically at an index.
    ///
    /// # Panics
    /// Panics if `y > self.height()`.
    pub fn split_vertical(&mut self, y: usize) -> (MutableSubgrid<'_, V>, MutableSubgrid<'_, V>) {
        assert!(y <= self.height);

        let top_ptr = self.ptr;
        let bottom_ptr = NonNull::new(unsafe { self.get_ptr_unchecked(0, y) }).unwrap();
        // SAFETY: two grids are contained in `self` and disjoint.
        unsafe {
            let split_base = self.split_base.unwrap_or(self.ptr.cast());
            let mut top_grid = MutableSubgrid::new(top_ptr, self.width, y, self.stride);
            let mut bottom_grid =
                MutableSubgrid::new(bottom_ptr, self.width, self.height - y, self.stride);
            top_grid.split_base = Some(split_base);
            bottom_grid.split_base = Some(split_base);
            (top_grid, bottom_grid)
        }
    }

    /// Split the grid vertically at an index in-place.
    ///
    /// # Panics
    /// Panics if `y > self.height()`.
    pub fn split_vertical_in_place(&mut self, y: usize) -> MutableSubgrid<'g, V> {
        assert!(y <= self.height);

        let bottom_height = self.height - y;
        let bottom_ptr = NonNull::new(unsafe { self.get_ptr_unchecked(0, y) }).unwrap();
        // SAFETY: two grids are contained in `self` and disjoint.
        unsafe {
            let split_base = self.split_base.unwrap_or(self.ptr.cast());
            self.height = y;
            self.split_base = Some(split_base);
            let mut bottom_grid =
                MutableSubgrid::new(bottom_ptr, self.width, bottom_height, self.stride);
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

impl<'g, V: Copy> MutableSubgrid<'g, V> {
    pub fn into_groups(
        self,
        group_width: usize,
        group_height: usize,
    ) -> Vec<MutableSubgrid<'g, V>> {
        assert!(
            group_width > 0 && group_height > 0,
            "expected group width and height to be nonzero, got width = {group_width}, height = {group_height}"
        );

        let num_cols = self.width.div_ceil(group_width);
        let num_rows = self.height.div_ceil(group_height);
        self.into_groups_with_fixed_count(group_width, group_height, num_cols, num_rows)
    }

    pub fn into_groups_with_fixed_count(
        self,
        group_width: usize,
        group_height: usize,
        num_cols: usize,
        num_rows: usize,
    ) -> Vec<MutableSubgrid<'g, V>> {
        let MutableSubgrid {
            ptr,
            split_base,
            width,
            height,
            stride,
            ..
        } = self;
        let split_base = split_base.unwrap_or(ptr.cast());

        let mut groups = Vec::with_capacity(num_cols * num_rows);
        for gy in 0..num_rows {
            let y = (gy * group_height).min(height);
            let gh = (height - y).min(group_height);
            let row_ptr = unsafe { ptr.as_ptr().add(y * stride) };
            for gx in 0..num_cols {
                let x = (gx * group_width).min(width);
                let gw = (width - x).min(group_width);
                let ptr = unsafe { row_ptr.add(x) };

                let mut grid =
                    unsafe { MutableSubgrid::new(NonNull::new(ptr).unwrap(), gw, gh, stride) };
                grid.split_base = Some(split_base);
                groups.push(grid);
            }
        }

        groups
    }
}

impl MutableSubgrid<'_, f32> {
    pub fn as_vectored<V: SimdVector>(&mut self) -> Option<MutableSubgrid<'_, V>> {
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
            .then(|| MutableSubgrid {
                ptr: self.ptr.cast::<V>(),
                split_base: self.split_base,
                width: self.width / V::SIZE,
                height: self.height,
                stride: self.stride / V::SIZE,
                _marker: Default::default(),
            })
    }
}
