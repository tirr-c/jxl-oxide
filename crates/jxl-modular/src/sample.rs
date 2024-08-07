use jxl_bitstream::unpack_signed;
use jxl_grid::{AlignedGrid, MutableSubgrid};

pub trait Sealed: Copy + Default + Send + Sync {
    fn try_as_mutable_subgrid_i32<'a, 'g>(
        grid: &'a mut MutableSubgrid<'g, Self>,
    ) -> Option<&'a mut MutableSubgrid<'g, i32>>;
    fn try_as_mutable_subgrid_i16<'a, 'g>(
        grid: &'a mut MutableSubgrid<'g, Self>,
    ) -> Option<&'a mut MutableSubgrid<'g, i16>>;
    fn try_into_grid_i32(grid: AlignedGrid<Self>) -> Result<AlignedGrid<i32>, AlignedGrid<Self>>;
    fn try_into_grid_i16(grid: AlignedGrid<Self>) -> Result<AlignedGrid<i16>, AlignedGrid<Self>>;

    fn unpack_signed_u32(value: u32) -> Self;

    /// Performs wrapping addition.
    fn add(self, rhs: Self) -> Self;

    /// Performs wrapping multiplication followed by wrapping addition.
    fn wrapping_muladd_i32(self, mul: i32, add: i32) -> Self;

    /// Computes clamped gradient, which is `(n + w - nw).clamp(w.min(n), w.max(n))`.
    fn grad_clamped(n: Self, w: Self, nw: Self) -> Self;
}

/// Type of Modular image samples.
///
/// Currently `i32` and `i16` implements `Sample`.
pub trait Sample: Copy + Default + Send + Sync + Sealed + 'static {
    fn from_i32(value: i32) -> Self;
    fn from_u32(value: u32) -> Self;

    fn to_i32(self) -> i32;
    fn to_i64(self) -> i64;
    fn to_f32(self) -> f32;
}

impl Sample for i32 {
    #[inline]
    fn from_i32(value: i32) -> Self {
        value
    }

    #[inline]
    fn from_u32(value: u32) -> Self {
        value as i32
    }

    #[inline]
    fn to_i32(self) -> i32 {
        self
    }

    #[inline]
    fn to_i64(self) -> i64 {
        self as i64
    }

    #[inline]
    fn to_f32(self) -> f32 {
        self as f32
    }
}

impl Sample for i16 {
    #[inline]
    fn from_i32(value: i32) -> Self {
        value as i16
    }

    #[inline]
    fn from_u32(value: u32) -> Self {
        value as i16
    }

    #[inline]
    fn to_i32(self) -> i32 {
        self as i32
    }

    #[inline]
    fn to_i64(self) -> i64 {
        self as i64
    }

    #[inline]
    fn to_f32(self) -> f32 {
        self as f32
    }
}

impl Sealed for i32 {
    fn try_as_mutable_subgrid_i32<'a, 'g>(
        grid: &'a mut MutableSubgrid<'g, i32>,
    ) -> Option<&'a mut MutableSubgrid<'g, i32>> {
        Some(grid)
    }

    fn try_as_mutable_subgrid_i16<'a, 'g>(
        _: &'a mut MutableSubgrid<'g, i32>,
    ) -> Option<&'a mut MutableSubgrid<'g, i16>> {
        None
    }

    fn try_into_grid_i32(grid: AlignedGrid<Self>) -> Result<AlignedGrid<i32>, AlignedGrid<Self>> {
        Ok(grid)
    }

    fn try_into_grid_i16(grid: AlignedGrid<Self>) -> Result<AlignedGrid<i16>, AlignedGrid<Self>> {
        Err(grid)
    }

    #[inline]
    fn unpack_signed_u32(value: u32) -> Self {
        unpack_signed(value)
    }

    #[inline]
    fn add(self, rhs: i32) -> i32 {
        self.wrapping_add(rhs)
    }

    #[inline]
    fn wrapping_muladd_i32(self, mul: i32, add: i32) -> i32 {
        self.wrapping_mul(mul).wrapping_add(add)
    }

    #[inline]
    fn grad_clamped(n: i32, w: i32, nw: i32) -> i32 {
        let (n, w) = if w > n {
            (w as i64, n as i64)
        } else {
            (n as i64, w as i64)
        };
        (w + n - nw as i64).clamp(w, n) as i32
    }
}

impl Sealed for i16 {
    fn try_as_mutable_subgrid_i32<'a, 'g>(
        _: &'a mut MutableSubgrid<'g, i16>,
    ) -> Option<&'a mut MutableSubgrid<'g, i32>> {
        None
    }

    fn try_as_mutable_subgrid_i16<'a, 'g>(
        grid: &'a mut MutableSubgrid<'g, i16>,
    ) -> Option<&'a mut MutableSubgrid<'g, i16>> {
        Some(grid)
    }

    fn try_into_grid_i32(grid: AlignedGrid<Self>) -> Result<AlignedGrid<i32>, AlignedGrid<Self>> {
        Err(grid)
    }

    fn try_into_grid_i16(grid: AlignedGrid<Self>) -> Result<AlignedGrid<i16>, AlignedGrid<Self>> {
        Ok(grid)
    }

    #[inline]
    fn unpack_signed_u32(value: u32) -> Self {
        let bit = (value & 1) as u16;
        let base = (value >> 1) as u16;
        let flip = 0u16.wrapping_sub(bit);
        (base ^ flip) as i16
    }

    #[inline]
    fn add(self, rhs: i16) -> i16 {
        self.wrapping_add(rhs)
    }

    #[inline]
    fn wrapping_muladd_i32(self, mul: i32, add: i32) -> i16 {
        self.wrapping_mul(mul as i16).wrapping_add(add as i16)
    }

    #[inline]
    fn grad_clamped(n: i16, w: i16, nw: i16) -> i16 {
        let (n, w) = if w > n {
            (w as i32, n as i32)
        } else {
            (n as i32, w as i32)
        };
        (w + n - nw as i32).clamp(w, n) as i16
    }
}
