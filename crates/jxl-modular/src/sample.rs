use jxl_bitstream::unpack_signed;
use jxl_grid::CutGrid;

pub trait Sealed: Copy + Default + Send + Sync {
    fn try_as_i32_cut_grid_mut<'a, 'g>(
        grid: &'a mut CutGrid<'g, Self>,
    ) -> Option<&'a mut CutGrid<'g, i32>>;
    fn try_as_i16_cut_grid_mut<'a, 'g>(
        grid: &'a mut CutGrid<'g, Self>,
    ) -> Option<&'a mut CutGrid<'g, i16>>;
}

pub trait Sample:
    Copy + Default + Send + Sync + Sealed + std::ops::Add<Self, Output = Self> + 'static
{
    fn from_i32(value: i32) -> Self;
    fn from_u32(value: u32) -> Self;
    fn unpack_signed_u32(value: u32) -> Self;
    fn to_i32(self) -> i32;
    fn to_i64(self) -> i64;
    fn to_f32(self) -> f32;
    fn wrapping_muladd_i32(self, mul: i32, add: i32) -> Self;
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
    fn unpack_signed_u32(value: u32) -> Self {
        unpack_signed(value)
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

    #[inline]
    fn wrapping_muladd_i32(self, mul: i32, add: i32) -> i32 {
        self.wrapping_mul(mul).wrapping_add(add)
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
    fn unpack_signed_u32(value: u32) -> Self {
        let bit = (value & 1) as u16;
        let base = (value >> 1) as u16;
        let flip = 0u16.wrapping_sub(bit);
        (base ^ flip) as i16
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

    #[inline]
    fn wrapping_muladd_i32(self, mul: i32, add: i32) -> i16 {
        self.wrapping_mul(mul as i16).wrapping_add(add as i16)
    }
}

impl Sealed for i32 {
    fn try_as_i32_cut_grid_mut<'a, 'g>(
        grid: &'a mut CutGrid<'g, i32>,
    ) -> Option<&'a mut CutGrid<'g, i32>> {
        Some(grid)
    }

    fn try_as_i16_cut_grid_mut<'a, 'g>(
        _: &'a mut CutGrid<'g, i32>,
    ) -> Option<&'a mut CutGrid<'g, i16>> {
        None
    }
}

impl Sealed for i16 {
    fn try_as_i32_cut_grid_mut<'a, 'g>(
        _: &'a mut CutGrid<'g, i16>,
    ) -> Option<&'a mut CutGrid<'g, i32>> {
        None
    }

    fn try_as_i16_cut_grid_mut<'a, 'g>(
        grid: &'a mut CutGrid<'g, i16>,
    ) -> Option<&'a mut CutGrid<'g, i16>> {
        Some(grid)
    }
}
