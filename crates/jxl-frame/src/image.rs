pub trait Sample: Copy + Sized {
    fn bits() -> u32;
    fn zero() -> Self;
}

impl Sample for u16 {
    fn bits() -> u32 {
        Self::BITS
    }

    fn zero() -> u16 {
        0
    }
}

impl Sample for u32 {
    fn bits() -> u32 {
        Self::BITS
    }

    fn zero() -> u32 {
        0
    }
}

impl Sample for f32 {
    fn bits() -> u32 {
        32
    }

    fn zero() -> f32 {
        0.0
    }
}

#[derive(Clone)]
pub struct Grid<S: Sample> {
    width: u32,
    height: u32,
    buffer: Vec<S>,
}

impl<S: Sample> std::fmt::Debug for Grid<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Grid")
            .field("width", &self.width)
            .field("height", &self.height)
            .finish_non_exhaustive()
    }
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
    pub fn new(width: u32, height: u32) -> Self {
        let count = width as usize * height as usize;
        let buffer = vec![S::zero(); count];
        Self { width, height, buffer }
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
}

impl<S: Sample> std::ops::Index<(i32, i32)> for Grid<S> {
    type Output = S;

    fn index(&self, index: (i32, i32)) -> &Self::Output {
        let (x, y) = self.mirror(index.0, index.1);
        let idx = x as usize + y as usize * self.width as usize;
        &self.buffer[idx]
    }
}

impl<S: Sample> std::ops::IndexMut<(i32, i32)> for Grid<S> {
    fn index_mut(&mut self, index: (i32, i32)) -> &mut Self::Output {
        let (x, y) = self.mirror(index.0, index.1);
        let idx = x as usize + y as usize * self.width as usize;
        &mut self.buffer[idx]
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
