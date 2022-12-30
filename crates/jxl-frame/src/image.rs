use std::collections::BTreeMap;

pub trait Sample: Default + Copy + Sized {
    type Signed: Sample;
    type Unsigned: Sample;

    fn bits() -> u32;
}

impl Sample for i16 {
    type Signed = i16;
    type Unsigned = u16;

    fn bits() -> u32 {
        Self::BITS
    }
}

impl Sample for u16 {
    type Signed = i16;
    type Unsigned = u16;

    fn bits() -> u32 {
        Self::BITS
    }
}

impl Sample for i32 {
    type Signed = i32;
    type Unsigned = u32;

    fn bits() -> u32 {
        Self::BITS
    }
}

impl Sample for u32 {
    type Signed = i32;
    type Unsigned = u32;

    fn bits() -> u32 {
        Self::BITS
    }
}

impl Sample for f32 {
    type Signed = f32;
    type Unsigned = f32;

    fn bits() -> u32 {
        32
    }
}

#[derive(Debug, Clone)]
pub struct Grid<S: Sample> {
    width: u32,
    height: u32,
    buffer: GridBuffer<S>,
}

#[derive(Debug, Clone)]
pub struct Subgrid<'g, S: Sample> {
    grid: &'g Grid<S>,
    left: i32,
    top: i32,
    width: u32,
    height: u32,
}

#[derive(Debug)]
pub struct SubgridMut<'g, S: Sample> {
    grid: &'g mut Grid<S>,
    left: i32,
    top: i32,
    width: u32,
    height: u32,
}

impl<S: Sample> Grid<S> {
    pub fn new(width: u32, height: u32, group_size: (u32, u32)) -> Self {
        Self {
            width,
            height,
            buffer: GridBuffer::new(width, height, group_size),
        }
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn mirror(&self, x: i32, y: i32) -> (u32, u32) {
        mirror_2d(self.width, self.height, x, y)
    }

    pub fn subgrid(&self, left: i32, top: i32, width: u32, height: u32) -> Subgrid<'_, S> {
        Subgrid {
            grid: self,
            left,
            top,
            width,
            height,
        }
    }

    pub fn subgrid_mut(&mut self, left: i32, top: i32, width: u32, height: u32) -> SubgridMut<'_, S> {
        SubgridMut {
            grid: self,
            left,
            top,
            width,
            height,
        }
    }

    pub fn anchor(&self, x: i32, y: i32) -> GridAnchor<'_, S> {
        GridAnchor { grid: self, x, y }
    }
}

impl<S: Sample> Grid<S> {
    pub fn insert_subgrid(&mut self, mut subgrid: Grid<S>, left: i32, top: i32) {
        if left == 0 && top == 0 && self.width == subgrid.width && self.height == subgrid.height {
            *self = subgrid;
            return;
        }

        if left + subgrid.width as i32 > self.width as i32 || top + subgrid.height as i32 > self.height as i32 {
            self.insert_subgrid_slow(subgrid, left, top);
            return;
        }

        let (gw, gh, group_stride, subgrid_group_stride, groups, subgrid_groups) = match (&mut self.buffer, &mut subgrid.buffer) {
            (GridBuffer::Single(_), _) | (_, GridBuffer::Single(_)) => {
                self.insert_subgrid_slow(subgrid, left, top);
                return;
            },
            (GridBuffer::Grouped { group_size, group_stride, groups }, GridBuffer::Grouped { group_size: subgrid_group_size, group_stride: subgrid_group_stride, groups: subgrid_groups }) => {
                if group_size != subgrid_group_size {
                    // group size mismatch
                    self.insert_subgrid_slow(subgrid, left, top);
                    return;
                }
                let (gw, gh) = *group_size;
                if left % gw as i32 != 0 || top % gh as i32 != 0 {
                    // not aligned
                    self.insert_subgrid_slow(subgrid, left, top);
                    return;
                }
                (gw, gh, *group_stride, *subgrid_group_stride, groups, subgrid_groups)
            },
        };

        let group_left = left / gw as i32;
        let group_top = top / gh as i32;
        let subgrid_group_left = group_left.min(0).unsigned_abs();
        let subgrid_group_top = group_top.min(0).unsigned_abs();
        let group_left = group_left.max(0) as u32;
        let group_top = group_top.max(0) as u32;
        if group_left >= group_stride || subgrid_group_left >= subgrid_group_stride {
            return;
        }

        let group_lines = (self.height + gh - 1) / gh;
        let subgrid_group_lines = (subgrid.height + gh - 1) / gh;
        if group_top >= group_lines || subgrid_group_top >= subgrid_group_lines {
            return;
        }

        let group_per_line = (group_stride - group_left).min(subgrid_group_stride - subgrid_group_left);
        let actual_group_lines = (group_lines - group_top).min(subgrid_group_lines - subgrid_group_top);

        for r in 0..actual_group_lines {
            for c in 0..group_per_line {
                let group_idx = (r + group_top) * group_stride + c + group_left;
                let subgrid_group_idx = (r + subgrid_group_top) * subgrid_group_stride + c + subgrid_group_left;
                let group = groups.get_mut(&group_idx).expect("group out of bounds");
                let subgrid_group = subgrid_groups.remove(&subgrid_group_idx).expect("subgrid group out of bounds");
                if group.stride == subgrid_group.stride && group.scanlines == subgrid_group.scanlines {
                    *group = subgrid_group;
                } else {
                    todo!()
                }
            }
        }
    }

    fn insert_subgrid_slow(&mut self, subgrid: Grid<S>, left: i32, top: i32) {
        let actual_left = left.max(0);
        let actual_top = top.max(0);
        let actual_subgrid_left = left.min(0).abs();
        let actual_subgrid_top = top.min(0).abs();

        let actual_width = (subgrid.width as i32 - actual_subgrid_left).min(self.width as i32 - actual_left);
        let actual_height = (subgrid.height as i32 - actual_subgrid_top).min(self.height as i32 - actual_top);
        if actual_width <= 0 || actual_height <= 0 {
            return;
        }
        let actual_width = actual_width as u32;
        let actual_height = actual_height as u32;

        let mut this = self.subgrid_mut(actual_left, actual_top, actual_width, actual_height);
        let subgrid = subgrid.subgrid(actual_subgrid_left, actual_subgrid_top, actual_width, actual_height);
        for y in 0..actual_height {
            for x in 0..actual_width {
                this[(x, y)] = subgrid[(x, y)];
            }
        }
    }
}

impl<S: Sample> std::ops::Index<(i32, i32)> for Grid<S> {
    type Output = S;

    fn index(&self, index: (i32, i32)) -> &Self::Output {
        let (x, y) = self.mirror(index.0, index.1);
        &self.buffer[(x, y)]
    }
}

impl<S: Sample> std::ops::IndexMut<(i32, i32)> for Grid<S> {
    fn index_mut(&mut self, index: (i32, i32)) -> &mut Self::Output {
        let (x, y) = self.mirror(index.0, index.1);
        &mut self.buffer[(x, y)]
    }
}

impl<S: Sample> Subgrid<'_, S> {
    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    #[inline]
    fn in_bounds(&self, x: u32, y: u32) -> bool {
        x < self.width && y < self.height
    }
}

impl<S: Sample> std::ops::Index<(u32, u32)> for Subgrid<'_, S> {
    type Output = S;

    fn index(&self, (x, y): (u32, u32)) -> &Self::Output {
        if !self.in_bounds(x, y) {
            panic!("index out of range")
        }

        let x = x as i32 + self.left;
        let y = y as i32 + self.top;
        &self.grid[(x, y)]
    }
}

impl<S: Sample> SubgridMut<'_, S> {
    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    #[inline]
    fn in_bounds(&self, x: u32, y: u32) -> bool {
        x < self.width && y < self.height
    }
}

impl<S: Sample> std::ops::Index<(u32, u32)> for SubgridMut<'_, S> {
    type Output = S;

    fn index(&self, (x, y): (u32, u32)) -> &Self::Output {
        if !self.in_bounds(x, y) {
            panic!("index out of range")
        }

        let x = x as i32 + self.left;
        let y = y as i32 + self.top;
        &self.grid[(x, y)]
    }
}

impl<S: Sample> std::ops::IndexMut<(u32, u32)> for SubgridMut<'_, S> {
    fn index_mut(&mut self, (x, y): (u32, u32)) -> &mut Self::Output {
        if !self.in_bounds(x, y) {
            panic!("index out of range")
        }

        let x = x as i32 + self.left;
        let y = y as i32 + self.top;
        &mut self.grid[(x, y)]
    }
}

#[derive(Clone)]
enum GridBuffer<S: Sample> {
    Single(GridGroup<S>),
    Grouped {
        group_size: (u32, u32),
        group_stride: u32,
        groups: BTreeMap<u32, GridGroup<S>>,
    },
}

impl<S: Sample> GridBuffer<S> {
    fn new(width: u32, height: u32, (gw, gh): (u32, u32)) -> Self {
        if width <= gw * 8 && height <= gh * 8 {
            Self::Single(GridGroup::new(width, height))
        } else {
            Self::Grouped {
                group_size: (gw, gh),
                group_stride: (width + gw - 1) / gw,
                groups: BTreeMap::new(),
            }
        }
    }
}

impl<S: Sample> std::ops::Index<(u32, u32)> for GridBuffer<S> {
    type Output = S;

    fn index(&self, (x, y): (u32, u32)) -> &Self::Output {
        match *self {
            Self::Single(ref group) => {
                let idx = x as usize + y as usize * group.stride as usize;
                &group.buf[idx]
            },
            Self::Grouped { group_size: (gw, gh), group_stride, ref groups } => {
                let group_row = y / gh;
                let group_col = x / gw;
                let y = y % gh;
                let x = x % gw;
                let group_idx = group_row * group_stride + group_col;
                if let Some(group) = groups.get(&group_idx) {
                    let idx = y * group.stride + x;
                    &group.buf[idx as usize]
                } else {
                    panic!()
                }
            },
        }
    }
}

impl<S: Sample> std::ops::IndexMut<(u32, u32)> for GridBuffer<S> {
    fn index_mut(&mut self, (x, y): (u32, u32)) -> &mut Self::Output {
        match *self {
            Self::Single(ref mut group) => {
                let idx = x as usize + y as usize * group.stride as usize;
                &mut group.buf[idx]
            },
            Self::Grouped { group_size: (gw, gh), group_stride, ref mut groups } => {
                let group_row = y / gh;
                let group_col = x / gw;
                let y = y % gh;
                let x = x % gw;
                let group_idx = group_row * group_stride + group_col;
                let group = groups
                    .entry(group_idx)
                    .or_insert_with(|| GridGroup::new(gw, gh));
                let idx = y * group.stride + x;
                &mut group.buf[idx as usize]
            },
        }
    }
}

impl<S: Sample> std::fmt::Debug for GridBuffer<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Single(_) => f.debug_tuple("Single").field(&format_args!("_")).finish(),
            Self::Grouped { group_size, group_stride, .. } => {
                f
                    .debug_struct("Grouped")
                    .field("group_size", group_size)
                    .field("group_stride", group_stride)
                    .finish_non_exhaustive()
            },
        }
    }
}

#[derive(Clone)]
struct GridGroup<S: Sample> {
    stride: u32,
    scanlines: u32,
    buf: Vec<S>,
}

impl<S: Sample> GridGroup<S> {
    fn new(stride: u32, scanlines: u32) -> Self {
        let size = stride as usize * scanlines as usize;
        Self {
            stride,
            scanlines,
            buf: vec![S::default(); size],
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct GridAnchor<'g, S: Sample> {
    grid: &'g Grid<S>,
    x: i32,
    y: i32,
}

impl<'g, S: Sample> GridAnchor<'g, S> {
    #[inline]
    pub fn w(self) -> S {
        let GridAnchor { grid, x, y } = self;
        if x > 0 {
            grid[(x - 1, y)]
        } else if y > 0 {
            grid[(x, y - 1)]
        } else {
            S::default()
        }
    }

    #[inline]
    pub fn n(self) -> S {
        let GridAnchor { grid, x, y } = self;
        if y > 0 {
            grid[(x, y - 1)]
        } else {
            self.w()
        }
    }

    #[inline]
    pub fn nw(self) -> S {
        let GridAnchor { grid, x, y } = self;
        if x > 0 && y > 0 {
            grid[(x - 1, y - 1)]
        } else {
            self.w()
        }
    }

    #[inline]
    pub fn ne(self) -> S {
        let GridAnchor { grid, x, y } = self;
        let w = grid.width as i32;
        if x + 1 < w && y > 0 {
            grid[(x + 1, y - 1)]
        } else {
            self.n()
        }
    }

    #[inline]
    pub fn nn(self) -> S {
        let GridAnchor { grid, x, y } = self;
        if y > 1 {
            grid[(x, y - 2)]
        } else {
            self.n()
        }
    }

    #[inline]
    pub fn nee(self) -> S {
        let GridAnchor { grid, x, y } = self;
        let w = grid.width as i32;
        if x + 2 < w && y > 0 {
            grid[(x + 2, y - 1)]
        } else {
            self.ne()
        }
    }

    #[inline]
    pub fn ww(self) -> S {
        let GridAnchor { grid, x, y } = self;
        if x > 1 {
            grid[(x - 2, y)]
        } else {
            self.w()
        }
    }
}

fn mirror_1d(len: u32, offset: i32) -> u32 {
    let offset = if offset < 0 {
        offset.abs_diff(-1)
    } else {
        offset as u32
    };
    if offset < len {
        return offset;
    }

    let offset = offset % (2 * len);
    if offset >= len {
        2 * len - offset - 1
    } else {
        offset
    }
}

fn mirror_2d(width: u32, height: u32, col: i32, row: i32) -> (u32, u32) {
    (mirror_1d(width, col), mirror_1d(height, row))
}
