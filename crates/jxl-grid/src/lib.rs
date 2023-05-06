//! This crate provides [`Grid`], [`SimpleGrid`], [`CutGrid`] and [`PaddedGrid`], used in various
//! places involving images.
mod simd;
mod simple_grid;
pub use simd::SimdVector;
pub use simple_grid::*;

/// A sample grid, possibly divided into smaller groups.
#[derive(Debug, Clone)]
pub enum Grid<S> {
    Simple(Option<SimpleGrid<S>>),
    Grouped {
        width: usize,
        height: usize,
        group_width: usize,
        group_height: usize,
        groups: Vec<Option<SimpleGrid<S>>>,
    },
}

impl<S> From<SimpleGrid<S>> for Grid<S> {
    fn from(value: SimpleGrid<S>) -> Self {
        Self::Simple(Some(value))
    }
}

impl<S: Default + Clone> Grid<S> {
    /// Create a new grid with given dimension.
    ///
    /// This method accepts `u32` for the convenience.
    pub fn new(width: u32, height: u32, group_width: u32, group_height: u32) -> Self {
        let width = width as usize;
        let height = height as usize;
        let group_width = group_width as usize;
        let group_height = group_height as usize;
        Self::new_usize(width, height, group_width, group_height)
    }

    /// Create a new grid with given dimension.
    pub fn new_usize(width: usize, height: usize, group_width: usize, group_height: usize) -> Self {
        if group_width == 0 || group_height == 0 {
            return Self::Simple(Some(SimpleGrid::new(width, height)));
        }

        let num_groups = ((width + group_width - 1) / group_width) * ((height + group_height - 1) / group_height);
        if num_groups == 1 {
            Self::Simple(Some(SimpleGrid::new(width, height)))
        } else {
            let mut groups = Vec::with_capacity(num_groups);
            groups.resize_with(num_groups, || None);
            Self::Grouped {
                width,
                height,
                group_width,
                group_height,
                groups,
            }
        }
    }
}

impl<S> Grid<S> {
    #[inline]
    pub fn width(&self) -> usize {
        match *self {
            Self::Simple(Some(ref g)) => g.width(),
            Self::Grouped { width, .. } => width,
            _ => unreachable!(),
        }
    }

    #[inline]
    pub fn height(&self) -> usize {
        match *self {
            Self::Simple(Some(ref g)) => g.height(),
            Self::Grouped { height, .. } => height,
            _ => unreachable!(),
        }
    }

    #[inline]
    pub fn group_dim(&self) -> (usize, usize) {
        match *self {
            Self::Simple(Some(ref g)) => (g.width(), g.height()),
            Self::Grouped { group_width, group_height, .. } => (group_width, group_height),
            _ => unreachable!(),
        }
    }

    /// Get the number of groups in a single row.
    #[inline]
    pub fn groups_per_row(&self) -> usize {
        match *self {
            Self::Simple(_) => 1,
            Self::Grouped { width, group_width, .. } => (width + group_width - 1) / group_width,
        }
    }

    pub fn get(&self, x: usize, y: usize) -> Option<&S> {
        let groups_per_row = self.groups_per_row();
        match *self {
            Self::Simple(Some(ref g)) => g.get(x, y),
            Self::Grouped { width, height, group_width, group_height, ref groups } => {
                if x >= width || y >= height {
                    return None;
                }

                let group_col = x / group_width;
                let x = x % group_width;
                let group_row = y / group_height;
                let y = y % group_height;
                let group_idx = group_row * groups_per_row + group_col;
                groups[group_idx].as_ref().and_then(|g| g.get(x, y))
            },
            _ => unreachable!(),
        }
    }

    pub fn get_mut(&mut self, x: usize, y: usize) -> Option<&mut S> {
        let groups_per_row = self.groups_per_row();
        match *self {
            Self::Simple(Some(ref mut g)) => g.get_mut(x, y),
            Self::Grouped { width, height, group_width, group_height, ref mut groups } => {
                if x >= width || y >= height {
                    return None;
                }

                let group_col = x / group_width;
                let x = x % group_width;
                let group_row = y / group_height;
                let y = y % group_height;
                let group_idx = group_row * groups_per_row + group_col;
                groups[group_idx].as_mut().and_then(|g| g.get_mut(x, y))
            },
            _ => unreachable!(),
        }
    }

    /// Get the reference to the [`SimpleGrid`] if the grid consists of a single group.
    #[inline]
    pub fn as_simple(&self) -> Option<&SimpleGrid<S>> {
        if let Self::Simple(Some(g)) = self {
            Some(g)
        } else {
            None
        }
    }

    /// Get the mutable reference to the [`SimpleGrid`] if the grid consists of a single group.
    #[inline]
    pub fn as_simple_mut(&mut self) -> Option<&mut SimpleGrid<S>> {
        if let Self::Simple(Some(g)) = self {
            Some(g)
        } else {
            None
        }
    }

    /// Make this grid into a [`SimpleGrid`] if the grid consists of a single group.
    #[inline]
    pub fn into_simple(self) -> Result<SimpleGrid<S>, Self> {
        if let Self::Simple(Some(g)) = self {
            Ok(g)
        } else {
            Err(self)
        }
    }
}

impl<S: Default + Clone> Grid<S> {
    pub fn set(&mut self, x: usize, y: usize, sample: S) -> Option<S> {
        let groups_per_row = self.groups_per_row();
        match *self {
            Self::Simple(Some(ref mut g)) => {
                let v = g.get_mut(x, y)?;
                let out = std::mem::replace(v, sample);
                Some(out)
            },
            Self::Grouped { width, height, group_width, group_height, ref mut groups } => {
                if x >= width || y >= height {
                    return None;
                }

                let group_col = x / group_width;
                let x = x % group_width;
                let group_row = y / group_height;
                let y = y % group_height;
                let group_idx = group_row * groups_per_row + group_col;

                let g = groups[group_idx]
                    .get_or_insert_with(|| SimpleGrid::new(group_width, group_height));
                let v = g.get_mut(x, y)?;
                let out = std::mem::replace(v, sample);
                Some(out)
            },
            _ => unreachable!(),
        }
    }
}

impl<S> Grid<S> {
    /// Iterate over the initialized groups of the grid.
    #[inline]
    pub fn groups(&self) -> impl Iterator<Item = (usize, &SimpleGrid<S>)> + '_ {
        let groups = self.all_groups();
        groups.iter().enumerate().filter_map(|(idx, g)| g.as_ref().map(|g| (idx, g)))
    }

    /// Iterate over the initialized groups of the grid mutably.
    #[inline]
    pub fn groups_mut(&mut self) -> impl Iterator<Item = (usize, &mut SimpleGrid<S>)> + '_ {
        let groups = self.all_groups_mut();
        groups.iter_mut().enumerate().filter_map(|(idx, g)| g.as_mut().map(|g| (idx, g)))
    }

    /// Get all groups of the grid in raster order.
    #[inline]
    pub fn all_groups(&self) -> &[Option<SimpleGrid<S>>] {
        match self {
            Self::Simple(g) => std::slice::from_ref(g),
            Self::Grouped { groups, .. } => groups,
        }
    }

    /// Get all groups of the grid in raster order mutably.
    #[inline]
    pub fn all_groups_mut(&mut self) -> &mut [Option<SimpleGrid<S>>] {
        match self {
            Self::Simple(g) => std::slice::from_mut(g),
            Self::Grouped { groups, .. } => groups,
        }
    }

    #[inline]
    pub fn subgrid(&self, left: usize, top: usize, width: usize, height: usize) -> Subgrid<'_, S> {
        Subgrid {
            grid: self,
            left,
            top,
            width,
            height,
        }
    }

    #[inline]
    pub fn as_subgrid(&self) -> Subgrid<'_, S> {
        Subgrid {
            grid: self,
            left: 0,
            top: 0,
            width: self.width(),
            height: self.height(),
        }
    }
}

impl<S> Grid<S> {
    /// Iterate over initialized samples using the callback function.
    pub fn iter_init_mut(&mut self, mut f: impl FnMut(usize, usize, &mut S)) {
        let groups_per_row = self.groups_per_row();
        match *self {
            Self::Simple(Some(ref mut g)) => {
                let width = g.width();
                let height = g.height();
                for y in 0..height {
                    for x in 0..width {
                        f(x, y, &mut g.buf_mut()[y * width + x]);
                    }
                }
            },
            Self::Grouped { group_width: gw, group_height: gh, ref mut groups, .. } => {
                let it = groups.iter_mut().enumerate().filter_map(|(idx, g)| g.as_mut().map(|g| (idx, g)));
                for (group_idx, g) in it {
                    let group_row = group_idx / groups_per_row;
                    let group_col = group_idx % groups_per_row;
                    let base_x = group_col * gw;
                    let base_y = group_row * gh;
                    let width = g.width();
                    let height = g.height();
                    for y in 0..height {
                        for x in 0..width {
                            f(base_x + x, base_y + y, &mut g.buf_mut()[y * width + x]);
                        }
                    }
                }
            },
            _ => unreachable!(),
        }
    }

    /// Zip three grids, and iterate over initialized samples using the callback function.
    pub fn zip3_mut(&mut self, b: &mut Grid<S>, c: &mut Grid<S>, mut f: impl FnMut(&mut S, &mut S, &mut S)) {
        assert!(self.width() == b.width() && b.width() == c.width());
        assert!(self.height() == b.height() && b.height() == c.height());
        assert!(self.group_dim() == b.group_dim() && b.group_dim() == c.group_dim());

        let a = self.all_groups_mut();
        let b = b.all_groups_mut();
        let c = c.all_groups_mut();
        let it = a.iter_mut().zip(b).zip(c).filter_map(|((a, b), c)| -> Option<_> {
            Some((a.as_mut()?, b.as_mut()?, c.as_mut()?))
        });

        for (a, b, c) in it {
            let width = a.width();
            let height = a.height();
            for y in 0..height {
                for x in 0..width {
                    let a = a.get_mut(x, y).unwrap();
                    let b = b.get_mut(x, y).unwrap();
                    let c = c.get_mut(x, y).unwrap();
                    f(a, b, c);
                }
            }
        }
    }
}

impl<S: Default + Clone> Grid<S> {
    /// Insert the grid into the given position.
    pub fn insert_subgrid(&mut self, subgrid: &mut Grid<S>, left: isize, top: isize) {
        let width = self.width();
        let height = self.height();
        let subgrid_width = subgrid.width();
        let subgrid_height = subgrid.height();

        // Return if subgrid doesn't overlap with canvas
        if left >= width as isize || top >= height as isize {
            return;
        }
        if subgrid_width.saturating_add_signed(left) == 0 || subgrid_height.saturating_add_signed(top) == 0 {
            return;
        }

        // Slow path if groups are not aligned
        if self.group_dim() != subgrid.group_dim() {
            return self.insert_subgrid_slow(subgrid.as_subgrid(), left, top);
        }
        let (gw, gh) = self.group_dim();
        if left % gw as isize != 0 || top % gh as isize != 0 {
            return self.insert_subgrid_slow(subgrid.as_subgrid(), left, top);
        }

        // Fast path
        let group_stride = self.groups_per_row();
        let group_max_height = (height + gh - 1) / gh;
        let subgrid_group_stride = subgrid.groups_per_row();
        let subgrid_group_max_height = (subgrid_height + gh - 1) / gh;

        let group_left;
        let group_top;
        let group_width;
        let group_height;
        let subgrid_group_left;
        let subgrid_group_top;
        if left < 0 {
            group_left = 0usize;
            subgrid_group_left = left.unsigned_abs() / gw;
            group_width = (subgrid_group_stride - subgrid_group_left).min(group_stride);
        } else {
            group_left = left as usize / gw;
            subgrid_group_left = 0usize;
            group_width = (group_stride - group_left).min(subgrid_group_stride);
        }
        if top < 0 {
            group_top = 0usize;
            subgrid_group_top = top.unsigned_abs() / gh;
            group_height = (subgrid_group_max_height - subgrid_group_top).min(group_max_height);
        } else {
            group_top = top as usize / gh;
            subgrid_group_top = 0usize;
            group_height = (group_max_height - group_top).min(subgrid_group_max_height);
        }

        let target_groups = self.all_groups_mut();
        let subgrid_groups = subgrid.all_groups_mut();
        for y in 0..group_height {
            for x in 0..group_width {
                let group_idx = (group_top + y) * group_stride + (group_left + x);
                let subgrid_group_idx = (subgrid_group_top + y) * subgrid_group_stride + (subgrid_group_left + x);
                let target_group = &mut target_groups[group_idx];
                let subgrid_group = &mut subgrid_groups[subgrid_group_idx];

                if let Some(subgrid_group) = subgrid_group.take() {
                    let subgrid_group_width = subgrid_group.width();
                    let subgrid_group_height = subgrid_group.height();
                    if gw == subgrid_group_width && gh == subgrid_group_height {
                        *target_group = Some(subgrid_group);
                    } else {
                        let target_group = target_group
                            .get_or_insert_with(|| SimpleGrid::new(gw, gh));
                        for (idx, sample) in subgrid_group.into_buf_iter().enumerate() {
                            let y = idx / subgrid_group_width;
                            let x = idx % subgrid_group_width;
                            *target_group.get_mut(x, y).unwrap() = sample;
                        }
                    }
                }
            }
        }
    }

    fn insert_subgrid_slow(&mut self, subgrid: Subgrid<'_, S>, left: isize, top: isize) {
        let target_left = left.max(0) as usize;
        let target_top = top.max(0) as usize;
        let subgrid_left = left.min(0).unsigned_abs();
        let subgrid_top = top.min(0).unsigned_abs();
        let width = (self.width() - target_left).min(subgrid.width - subgrid_left);
        let height = (self.height() - target_top).min(subgrid.height - subgrid_top);

        for y in 0..height {
            for x in 0..width {
                if let Some(s) = subgrid.get(subgrid_left + x, subgrid_top + y) {
                    self.set(target_left + x, target_top + y, s.clone());
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Subgrid<'g, S> {
    grid: &'g Grid<S>,
    left: usize,
    top: usize,
    width: usize,
    height: usize,
}

impl<S> Subgrid<'_, S> {
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
        self.grid.get(self.left + x, self.top + y)
    }
}
