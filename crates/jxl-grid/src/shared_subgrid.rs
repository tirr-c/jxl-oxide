use std::{ops::RangeBounds, ptr::NonNull, sync::atomic::AtomicI32};

use crate::SimdVector;

#[derive(Debug)]
pub struct SharedSubgrid<'g, V = f32> {
    ptr: NonNull<V>,
    width: usize,
    height: usize,
    stride: usize,
    _marker: std::marker::PhantomData<&'g [V]>,
}

unsafe impl<'g, V> Send for SharedSubgrid<'g, V> where &'g [V]: Send {}
unsafe impl<'g, V> Sync for SharedSubgrid<'g, V> where &'g [V]: Sync {}

impl<'g, V> From<&'g crate::AlignedGrid<V>> for SharedSubgrid<'g, V> {
    fn from(value: &'g crate::AlignedGrid<V>) -> Self {
        SharedSubgrid::from_buf(value.buf(), value.width(), value.height(), value.width())
    }
}

impl<'g, V> SharedSubgrid<'g, V> {
    /// Create a `SharedSubgrid` from raw pointer to the buffer, width, height and stride.
    ///
    /// # Safety
    /// The memory region specified by `width`, `height` and `stride` must be valid.
    pub unsafe fn new(ptr: NonNull<V>, width: usize, height: usize, stride: usize) -> Self {
        Self {
            ptr,
            width,
            height,
            stride,
            _marker: Default::default(),
        }
    }

    /// Create a `SharedSubgrid` from buffer slice, width, height and stride.
    ///
    /// # Panic
    /// Panics if:
    /// - either `width` or `height` is zero,
    /// - `width` is greater than `stride`,
    /// - or the area specified by `width`, `height` and `stride` is larger than `buf`.
    pub fn from_buf(buf: &'g [V], width: usize, height: usize, stride: usize) -> Self {
        assert!(width > 0);
        assert!(height > 0);
        assert!(width <= stride);
        assert!(buf.len() >= stride * (height - 1) + width);
        // SAFETY: the area is in bounds.
        unsafe {
            Self::new(
                NonNull::new(buf.as_ptr() as *mut _).unwrap(),
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
    pub fn get(&self, x: usize, y: usize) -> &V {
        let ptr = self.get_ptr(x, y);
        // SAFETY: get_ptr returns a valid pointer.
        unsafe { ptr.as_ref().unwrap() }
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
}

impl<'g, V> SharedSubgrid<'g, V> {
    /// Split the grid horizontally at an index.
    ///
    /// # Panics
    /// Panics if `x > self.width()`.
    pub fn split_horizontal(&self, x: usize) -> (SharedSubgrid<'g, V>, SharedSubgrid<'g, V>) {
        assert!(x <= self.width);

        let left_ptr = self.ptr;
        let right_ptr = NonNull::new(self.get_ptr(x, 0)).unwrap();
        // SAFETY: two grids are contained in `self`.
        unsafe {
            let left_grid = SharedSubgrid::new(left_ptr, x, self.height, self.stride);
            let right_grid =
                SharedSubgrid::new(right_ptr, self.width - x, self.height, self.stride);
            (left_grid, right_grid)
        }
    }

    /// Split the grid vertically at an index.
    ///
    /// # Panics
    /// Panics if `y > self.height()`.
    pub fn split_vertical(&self, y: usize) -> (SharedSubgrid<'g, V>, SharedSubgrid<'g, V>) {
        assert!(y <= self.height);

        let top_ptr = self.ptr;
        let bottom_ptr = NonNull::new(self.get_ptr(0, y)).unwrap();
        // SAFETY: two grids are contained in `self`.
        unsafe {
            let top_grid = SharedSubgrid::new(top_ptr, self.width, y, self.stride);
            let bottom_grid =
                SharedSubgrid::new(bottom_ptr, self.width, self.height - y, self.stride);
            (top_grid, bottom_grid)
        }
    }

    pub fn subgrid(
        &self,
        range_x: impl RangeBounds<usize>,
        range_y: impl RangeBounds<usize>,
    ) -> SharedSubgrid<'g, V> {
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
        unsafe { SharedSubgrid::new(base_ptr, right - left, bottom - top, self.stride) }
    }
}

impl<'g> SharedSubgrid<'g, f32> {
    pub fn as_vectored<V: SimdVector>(&self) -> Option<SharedSubgrid<'g, V>> {
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
            .then(|| SharedSubgrid {
                ptr: self.ptr.cast::<V>(),
                width: self.width / V::SIZE,
                height: self.height,
                stride: self.stride / V::SIZE,
                _marker: Default::default(),
            })
    }

    /// # Safety
    /// Caller should make sure that atomic and non-atomic accesses are not mixed.
    pub unsafe fn as_atomic_i32(&self) -> SharedSubgrid<'g, AtomicI32> {
        assert_eq!(std::mem::size_of::<f32>(), std::mem::size_of::<AtomicI32>());
        assert_eq!(
            std::mem::align_of::<f32>(),
            std::mem::align_of::<AtomicI32>()
        );
        SharedSubgrid {
            ptr: self.ptr.cast(),
            width: self.width,
            height: self.height,
            stride: self.stride,
            _marker: Default::default(),
        }
    }
}
