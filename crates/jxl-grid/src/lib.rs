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
pub struct Grid<S> {
    width: u32,
    height: u32,
    group_size: (u32, u32),
    buffer: GridBuffer<S>,
}

#[derive(Debug, Clone)]
pub struct Subgrid<'g, S> {
    grid: &'g Grid<S>,
    left: i32,
    top: i32,
    width: u32,
    height: u32,
}

#[derive(Debug)]
pub struct SubgridMut<'g, S> {
    grid: &'g mut Grid<S>,
    left: i32,
    top: i32,
    width: u32,
    height: u32,
}

impl<S: Default + Clone> Grid<S> {
    pub fn new(width: u32, height: u32, group_size: (u32, u32)) -> Self {
        Self {
            width,
            height,
            group_size,
            buffer: GridBuffer::new(width, height, group_size),
        }
    }
}

impl<S> Grid<S> {
    #[inline]
    pub fn width(&self) -> u32 {
        self.width
    }

    #[inline]
    pub fn height(&self) -> u32 {
        self.height
    }

    #[inline]
    pub fn group_size(&self) -> (u32, u32) {
        self.group_size
    }

    #[inline]
    pub fn mirror(&self, x: i32, y: i32) -> (u32, u32) {
        mirror_2d(self.width, self.height, x, y)
    }

    pub fn get_initialized(&self, coord: (u32, u32)) -> Option<&S> {
        self.buffer.get_initialized(coord)
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

    pub fn iter_init_mut(&mut self, mut f: impl FnMut(u32, u32, &mut S)) {
        let (gw, gh) = self.group_size;
        let groups = match &mut self.buffer {
            GridBuffer::Single(buf) => vec![(0u32, 0u32, buf)],
            GridBuffer::Grouped { group_stride, groups, .. } => {
                groups.iter_mut().map(|(&idx, v)| {
                    let group_left = idx % *group_stride;
                    let group_top = idx / *group_stride;
                    let x = group_left * gw;
                    let y = group_top * gh;
                    (x, y, v)
                }).collect()
            },
        };

        for (base_x, base_y, group) in groups {
            let stride = group.stride;
            for (idx, sample) in group.buf.iter_mut().enumerate() {
                let y = idx as u32 / stride;
                let x = idx as u32 % stride;
                f(base_x + x, base_y + y, sample);
            }
        }
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

        if self.group_size != subgrid.group_size {
            // group size mismatch
            self.insert_subgrid_slow(subgrid, left, top);
            return;
        }

        let (gw, gh) = self.group_size;
        if left % gw as i32 != 0 || top % gh as i32 != 0 {
            // not aligned
            self.insert_subgrid_slow(subgrid, left, top);
            return;
        }

        let (group_stride, subgrid_group_stride, groups, subgrid_groups) = match (&mut self.buffer, &mut subgrid.buffer) {
            (GridBuffer::Single(_), _) => {
                self.insert_subgrid_slow(subgrid, left, top);
                return;
            },
            (GridBuffer::Grouped { group_stride, groups, .. }, GridBuffer::Single(subgrid_group)) => {
                if left < 0 || top < 0 {
                    return;
                }
                let group_left = left as u32 / gw;
                let group_top = top as u32 / gh;
                let group_idx = group_top * *group_stride + group_left;
                if let Some(group) = groups.get_mut(&group_idx) {
                    if group.stride == subgrid_group.stride && group.scanlines == subgrid_group.scanlines {
                        std::mem::swap(group, subgrid_group);
                    } else {
                        let group_stride = group.stride as usize;
                        let subgrid_group_stride = subgrid_group.stride as usize;
                        let width = group_stride.min(subgrid_group_stride);
                        let height = group.scanlines.min(subgrid_group.scanlines) as usize;
                        for y in 0..height {
                            for x in 0..width {
                                group.buf[group_stride * y + x] = subgrid_group.buf[subgrid_group_stride * y + x];
                            }
                        }
                    }
                } else {
                    let stride = (self.width - left as u32).min(gw);
                    let scanlines = (self.height - top as u32).min(gh);
                    let subgrid_group = if stride == subgrid_group.stride && scanlines == subgrid_group.scanlines {
                        std::mem::replace(subgrid_group, GridGroup::new(0, 0))
                    } else {
                        let group_stride = stride as usize;
                        let subgrid_group_stride = subgrid_group.stride as usize;
                        let width = group_stride.min(subgrid_group_stride);
                        let height = scanlines.min(subgrid_group.scanlines) as usize;

                        let mut group = GridGroup::new(stride, scanlines);
                        for y in 0..height {
                            for x in 0..width {
                                group.buf[group_stride * y + x] = subgrid_group.buf[subgrid_group_stride * y + x];
                            }
                        }
                        group
                    };
                    groups.insert(group_idx, subgrid_group);
                }
                return;
            },
            (GridBuffer::Grouped { group_stride, groups, .. }, GridBuffer::Grouped { group_stride: subgrid_group_stride, groups: subgrid_groups, .. }) => {
                (*group_stride, *subgrid_group_stride, groups, subgrid_groups)
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
                    let group_stride = group.stride as usize;
                    let subgrid_group_stride = subgrid_group.stride as usize;
                    let width = group_stride.min(subgrid_group_stride);
                    let height = group.scanlines.min(subgrid_group.scanlines) as usize;
                    for y in 0..height {
                        for x in 0..width {
                            group.buf[group_stride * y + x] = subgrid_group.buf[subgrid_group_stride * y + x];
                        }
                    }
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

impl<S> std::ops::Index<(u32, u32)> for Grid<S> {
    type Output = S;

    fn index(&self, coord: (u32, u32)) -> &Self::Output {
        &self.buffer[coord]
    }
}

impl<S: Default + Clone> std::ops::IndexMut<(u32, u32)> for Grid<S> {
    fn index_mut(&mut self, coord: (u32, u32)) -> &mut Self::Output {
        &mut self.buffer[coord]
    }
}

impl<S> Subgrid<'_, S> {
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

impl<S> std::ops::Index<(u32, u32)> for Subgrid<'_, S> {
    type Output = S;

    fn index(&self, (x, y): (u32, u32)) -> &Self::Output {
        if !self.in_bounds(x, y) {
            panic!("index out of range")
        }

        let x = x.saturating_add_signed(self.left);
        let y = y.saturating_add_signed(self.top);
        &self.grid[(x, y)]
    }
}

impl<S> SubgridMut<'_, S> {
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

impl<S> std::ops::Index<(u32, u32)> for SubgridMut<'_, S> {
    type Output = S;

    fn index(&self, (x, y): (u32, u32)) -> &Self::Output {
        if !self.in_bounds(x, y) {
            panic!("index out of range")
        }

        let x = x.saturating_add_signed(self.left);
        let y = y.saturating_add_signed(self.top);
        &self.grid[(x, y)]
    }
}

impl<S: Default + Clone> std::ops::IndexMut<(u32, u32)> for SubgridMut<'_, S> {
    fn index_mut(&mut self, (x, y): (u32, u32)) -> &mut Self::Output {
        if !self.in_bounds(x, y) {
            panic!("index out of range")
        }

        let x = x.saturating_add_signed(self.left);
        let y = y.saturating_add_signed(self.top);
        &mut self.grid[(x, y)]
    }
}

#[derive(Clone)]
enum GridBuffer<S> {
    Single(GridGroup<S>),
    Grouped {
        group_size: (u32, u32),
        group_stride: u32,
        groups: BTreeMap<u32, GridGroup<S>>,
        def: S,
    },
}

impl<S: Default + Clone> GridBuffer<S> {
    fn new(width: u32, height: u32, (gw, gh): (u32, u32)) -> Self {
        if width <= gw && height <= gh {
            Self::Single(GridGroup::new(width, height))
        } else {
            Self::Grouped {
                group_size: (gw, gh),
                group_stride: (width + gw - 1) / gw,
                groups: BTreeMap::new(),
                def: S::default(),
            }
        }
    }
}

impl<S> GridBuffer<S> {
    fn get_group(&self, idx: u32) -> Option<&GridGroup<S>> {
        match self {
            Self::Single(buf) => (idx == 0).then_some(buf),
            Self::Grouped { groups, .. } => groups.get(&idx),
        }
    }

    fn get_initialized(&self, (x, y): (u32, u32)) -> Option<&S> {
        match *self {
            Self::Single(ref group) => {
                assert!(x < group.stride);
                let idx = x as usize + y as usize * group.stride as usize;
                group.buf.get(idx)
            },
            Self::Grouped { group_size: (gw, gh), group_stride, ref groups, .. } => {
                let group_row = y / gh;
                let group_col = x / gw;
                let y = y % gh;
                let x = x % gw;
                let group_idx = group_row * group_stride + group_col;
                if let Some(group) = groups.get(&group_idx) {
                    let idx = y * group.stride + x;
                    group.buf.get(idx as usize)
                } else {
                    None
                }
            },
        }
    }
}

impl<S> std::ops::Index<(u32, u32)> for GridBuffer<S> {
    type Output = S;

    fn index(&self, (x, y): (u32, u32)) -> &Self::Output {
        self.get_initialized((x, y)).unwrap()
    }
}

impl<S: Default + Clone> std::ops::IndexMut<(u32, u32)> for GridBuffer<S> {
    fn index_mut(&mut self, (x, y): (u32, u32)) -> &mut Self::Output {
        match *self {
            Self::Single(ref mut group) => {
                assert!(x < group.stride);
                let idx = x as usize + y as usize * group.stride as usize;
                &mut group.buf[idx]
            },
            Self::Grouped { group_size: (gw, gh), group_stride, ref mut groups, .. } => {
                let group_row = y / gh;
                let group_col = x / gw;
                let y = y % gh;
                let x = x % gw;
                let group_idx = group_row * group_stride + group_col;
                let group = groups.entry(group_idx)
                    .or_insert_with(|| GridGroup::new(gw, gh));
                let idx = y * group.stride + x;
                &mut group.buf[idx as usize]
            },
        }
    }
}

impl<S> std::fmt::Debug for GridBuffer<S> {
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
struct GridGroup<S> {
    stride: u32,
    scanlines: u32,
    buf: Vec<S>,
}

impl<S: Default + Clone> GridGroup<S> {
    fn new(stride: u32, scanlines: u32) -> Self {
        let size = stride as usize * scanlines as usize;
        Self {
            stride,
            scanlines,
            buf: vec![S::default(); size],
        }
    }
}

impl<S> GridGroup<S> {
    fn idx(&self, x: u32, y: u32) -> usize {
        (y * self.stride + x) as usize
    }

    fn get(&self, x: u32, y: u32) -> Option<&S> {
        self.buf.get(self.idx(x, y))
    }

    fn get_mut(&mut self, x: u32, y: u32) -> Option<&mut S> {
        let idx = self.idx(x, y);
        self.buf.get_mut(idx)
    }
}

impl<S> std::ops::Index<(u32, u32)> for GridGroup<S> {
    type Output = S;

    fn index(&self, (x, y): (u32, u32)) -> &Self::Output {
        &self.buf[self.idx(x, y)]
    }
}

impl<S> std::ops::IndexMut<(u32, u32)> for GridGroup<S> {
    fn index_mut(&mut self, (x, y): (u32, u32)) -> &mut Self::Output {
        let idx = self.idx(x, y);
        &mut self.buf[idx]
    }
}

impl<S> std::fmt::Debug for GridGroup<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GridGroup")
            .field("stride", &self.stride)
            .field("scanlines", &self.scanlines)
            .finish_non_exhaustive()
    }
}

#[inline(always)]
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

#[inline(always)]
fn mirror_2d(width: u32, height: u32, col: i32, row: i32) -> (u32, u32) {
    (mirror_1d(width, col), mirror_1d(height, row))
}

#[cfg(not(feature = "mt"))]
pub fn zip_iterate<S>(grids: &mut [&mut Grid<S>], f: impl Fn(&mut [&mut S])) {
    if grids.is_empty() {
        return;
    }

    let mut groups = grids
        .iter_mut()
        .map(|g| match &mut g.buffer {
            GridBuffer::Single(buf) => vec![(0, buf)],
            GridBuffer::Grouped { groups, .. } => {
                groups.iter_mut().map(|(k, v)| (*k, v)).collect()
            },
        })
        .map(|v| v.into_iter().peekable())
        .collect::<Vec<_>>();

    let mut target_group_list = Vec::new();
    'main_loop: loop {
        let mut target_group_idx = 0;
        for it in groups.iter_mut() {
            let Some(&(idx, _)) = it.peek() else { break 'main_loop; };
            target_group_idx = target_group_idx.max(idx);
        }

        let mut target_groups = Vec::with_capacity(groups.len());
        for it in groups.iter_mut() {
            loop {
                let Some(&(idx, _)) = it.peek() else { break 'main_loop; };
                match idx.cmp(&target_group_idx) {
                    std::cmp::Ordering::Greater => continue 'main_loop,
                    std::cmp::Ordering::Equal => break,
                    std::cmp::Ordering::Less => { it.next(); },
                }
            }
        }
        for it in groups.iter_mut() {
            target_groups.push(&mut it.next().unwrap().1.buf);
        }

        target_group_list.push(target_groups);
    }

    for mut target_groups in target_group_list {
        let len = target_groups.iter().map(|buf| buf.len()).min().unwrap();
        for i in 0..len {
            let mut samples = target_groups.iter_mut().map(|buf| &mut buf[i]).collect::<Vec<_>>();
            f(&mut samples);
        }
    }
}

#[cfg(feature = "mt")]
pub fn zip_iterate<S: Send>(grids: &mut [&mut Grid<S>], f: impl Fn(&mut [&mut S]) + Send + Sync) {
    use rayon::prelude::*;

    if grids.is_empty() {
        return;
    }

    let mut groups = grids
        .iter_mut()
        .map(|g| match &mut g.buffer {
            GridBuffer::Single(buf) => vec![(0, buf)],
            GridBuffer::Grouped { groups, .. } => {
                groups.iter_mut().map(|(k, v)| (*k, v)).collect()
            },
        })
        .map(|v| v.into_iter().peekable())
        .collect::<Vec<_>>();

    let mut target_group_list = Vec::new();
    'main_loop: loop {
        let mut target_group_idx = 0;
        for it in groups.iter_mut() {
            let Some(&(idx, _)) = it.peek() else { break 'main_loop; };
            target_group_idx = target_group_idx.max(idx);
        }

        let mut target_groups = Vec::with_capacity(groups.len());
        for it in groups.iter_mut() {
            loop {
                let Some(&(idx, _)) = it.peek() else { break 'main_loop; };
                match idx.cmp(&target_group_idx) {
                    std::cmp::Ordering::Greater => continue 'main_loop,
                    std::cmp::Ordering::Equal => break,
                    std::cmp::Ordering::Less => { it.next(); },
                }
            }
        }
        for it in groups.iter_mut() {
            target_groups.push(&mut it.next().unwrap().1.buf);
        }

        target_group_list.push(target_groups);
    }

    target_group_list
        .into_par_iter()
        .for_each(|mut target_groups| {
            let len = target_groups.iter().map(|buf| buf.len()).min().unwrap();
            for i in 0..len {
                let mut samples = target_groups.iter_mut().map(|buf| &mut buf[i]).collect::<Vec<_>>();
                f(&mut samples);
            }
        });
}

pub fn rgba_be_interleaved<F, E>(
    rgb: [&Grid<i32>; 3],
    a: Option<&Grid<i32>>,
    bit_depth: u32,
    mut f: F,
) -> Result<(), E>
where
    F: FnMut(&[u8]) -> Result<(), E>,
{
    if bit_depth != 8 && bit_depth != 16 {
        todo!("rgba_be_interleaved currently supports only 8-bit and 16-bit images");
    }

    let bytes_per_sample = (3 + (a.is_some() as usize)) * (bit_depth / 8) as usize;

    let width = rgb[0].width;
    let height = rgb[0].height;
    let (gw, gh) = rgb[0].group_size;
    let mut buf = vec![0u8; width as usize * gh as usize * bytes_per_sample];

    let empty_grid = GridGroup::new(gw, gh);

    let group_stride = (width + gw - 1) / gw;
    let group_height = (height + gh - 1) / gh;
    for gy in 0..group_height {
        let scanlines = (height - gy * gh).min(gh);
        for gx in 0..group_stride {
            let idx = gy * group_stride + gx;
            let stride = (width - gx * gw).min(gw);

            let r = rgb[0].buffer.get_group(idx).unwrap_or(&empty_grid);
            let g = rgb[1].buffer.get_group(idx).unwrap_or(&empty_grid);
            let b = rgb[2].buffer.get_group(idx).unwrap_or(&empty_grid);
            let a = a.as_ref().map(|g| g.buffer.get_group(idx).unwrap_or(&empty_grid));

            for y in 0..scanlines {
                for x in 0..stride {
                    let buf_idx = (y as usize * width as usize + (gx * gw + x) as usize) * bytes_per_sample;
                    let r = r.buf[(y * r.stride + x) as usize];
                    let g = g.buf[(y * g.stride + x) as usize];
                    let b = b.buf[(y * b.stride + x) as usize];
                    let a = a.map(|g| g.buf[(y * g.stride + x) as usize]);
                    if bit_depth == 8 {
                        let buf = &mut buf[buf_idx..];
                        buf[0] = r.clamp(0, 255) as u8;
                        buf[1] = g.clamp(0, 255) as u8;
                        buf[2] = b.clamp(0, 255) as u8;
                        if let Some(a) = a {
                            buf[3] = a.clamp(0, 255) as u8;
                        }
                    } else {
                        let buf = &mut buf[buf_idx..];
                        buf[0..2].copy_from_slice(&(r.clamp(0, 65535) as u16).to_be_bytes());
                        buf[2..4].copy_from_slice(&(g.clamp(0, 65535) as u16).to_be_bytes());
                        buf[4..6].copy_from_slice(&(b.clamp(0, 65535) as u16).to_be_bytes());
                        if let Some(a) = a {
                            buf[6..8].copy_from_slice(&(a as u16).to_be_bytes());
                        }
                    }
                }
            }
        }
        f(&buf[..((scanlines * width) as usize * bytes_per_sample)])?;
    }
    Ok(())
}
