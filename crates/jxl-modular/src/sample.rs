use jxl_grid::CutGrid;

pub trait Sealed: Copy + Default + Send + Sync {
    fn try_as_i32_cut_grid_mut<'a, 'g>(grid: &'a mut CutGrid<'g, Self>) -> Option<&'a mut CutGrid<'g, i32>>;
    fn try_as_i16_cut_grid_mut<'a, 'g>(grid: &'a mut CutGrid<'g, Self>) -> Option<&'a mut CutGrid<'g, i16>>;
}

pub trait Sample: Copy + Default + Send + Sync + Sealed + 'static {
    fn from_i32(value: i32) -> Self;
    fn from_u32(value: u32) -> Self;
    fn to_i32(self) -> i32;
    fn to_i64(self) -> i64;
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
}

impl Sealed for i32 {
    fn try_as_i32_cut_grid_mut<'a, 'g>(grid: &'a mut CutGrid<'g, i32>) -> Option<&'a mut CutGrid<'g, i32>> {
        Some(grid)
    }

    fn try_as_i16_cut_grid_mut<'a, 'g>(_: &'a mut CutGrid<'g, i32>) -> Option<&'a mut CutGrid<'g, i16>> {
        None
    }
}

impl Sealed for i16 {
    fn try_as_i32_cut_grid_mut<'a, 'g>(_: &'a mut CutGrid<'g, i16>) -> Option<&'a mut CutGrid<'g, i32>> {
        None
    }

    fn try_as_i16_cut_grid_mut<'a, 'g>(grid: &'a mut CutGrid<'g, i16>) -> Option<&'a mut CutGrid<'g, i16>> {
        Some(grid)
    }
}
