use std::{ptr::NonNull, collections::HashMap};

use jxl_frame::data::HfCoeff;
use jxl_grid::SimpleGrid;
use jxl_modular::ChannelShift;

use crate::lane::SimdLane;

#[derive(Debug)]
pub struct CutGrid<'g, Lane: Copy = f32> {
    ptr: NonNull<Lane>,
    width: usize,
    height: usize,
    stride: usize,
    _marker: std::marker::PhantomData<&'g mut [Lane]>
}

impl<'g, Lane: Copy> CutGrid<'g, Lane> {
    pub fn from_buf(buf: &'g mut [Lane], width: usize, height: usize, stride: usize) -> Self {
        assert!(width <= stride);
        assert!(buf.len() >= stride * height);
        Self {
            ptr: NonNull::new(buf.as_mut_ptr()).unwrap(),
            width,
            height,
            stride,
            _marker: Default::default(),
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
    fn get_ptr(&self, x: usize, y: usize) -> *mut Lane {
        if x >= self.width || y >= self.height {
            panic!("Coordinate out of range: ({}, {}) not in {}x{}", x, y, self.width, self.height);
        }

        // SAFETY: (x, y) is checked above and is in bounds.
        unsafe {
            let offset = y * self.stride + x;
            self.ptr.as_ptr().add(offset)
        }
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> Lane {
        let ptr = self.get_ptr(x, y);
        // SAFETY: get_ptr returns a valid pointer.
        unsafe { *ptr }
    }

    #[inline]
    pub fn get_row(&self, row: usize) -> &[Lane] {
        let ptr = self.get_ptr(0, row);
        unsafe { std::slice::from_raw_parts(ptr as *const _, self.width) }
    }

    #[inline]
    pub fn get_mut(&mut self, x: usize, y: usize) -> &mut Lane {
        let ptr = self.get_ptr(x, y);
        // SAFETY: get_ptr returns a valid pointer, and mutable borrow of `self` makes sure that
        // the access is exclusive.
        unsafe { ptr.as_mut().unwrap() }
    }

    #[inline]
    pub fn get_row_mut(&mut self, row: usize) -> &mut [Lane] {
        let ptr = self.get_ptr(0, row);
        unsafe { std::slice::from_raw_parts_mut(ptr, self.width) }
    }
}

pub fn cut_with_block_info<'g>(
    grid: &'g mut SimpleGrid<f32>,
    group_coeffs: &HashMap<usize, HfCoeff>,
    group_dim: usize,
    jpeg_upsampling: ChannelShift,
) -> HashMap<usize, HashMap<(usize, usize), CutGrid<'g>>> {
    let grid_width = grid.width();
    let grid_height = grid.height();
    let buf = grid.buf_mut();
    let ptr = NonNull::new(buf.as_mut_ptr()).unwrap();

    let hshift = jpeg_upsampling.hshift();
    let vshift = jpeg_upsampling.vshift();
    let groups_per_row = (grid_width + group_dim - 1) / group_dim;

    group_coeffs
        .iter()
        .map(|(&idx, group)| {
            let group_y = idx / groups_per_row;
            let group_x = idx % groups_per_row;
            let base_y = (group_y * group_dim) >> vshift;
            let base_x = (group_x * group_dim) >> hshift;
            let mut check_flags = vec![false; group_dim * group_dim];

            let mut subgrids = HashMap::new();
            for (&(x, y), coeff) in &group.data {
                let sx = x >> hshift;
                let sy = y >> hshift;
                if (sx << hshift) != x || (sy << vshift) != y {
                    continue;
                }

                let dct_select = coeff.dct_select;
                let x8 = sx * 8;
                let y8 = sy * 8;
                let (bw, bh) = dct_select.dct_select_size();
                for dy in 0..bh as usize {
                    for dx in 0..bw as usize {
                        let idx = (sy + dy) * group_dim + (sx + dx);
                        if check_flags[idx] {
                            panic!("Invalid block_info");
                        }
                        check_flags[idx] = true;
                    }
                }

                let block_width = bw as usize * 8;
                let block_height = bh as usize * 8;
                let grid_x = base_x + x8;
                let grid_y = base_y + y8;
                if grid_x + block_width > grid_width || grid_y + block_height > grid_height {
                    panic!("grid_x={grid_x}, grid_y={grid_y}, block_width={block_width}, block_height={block_height}, grid_width={grid_width}, grid_height={grid_height}");
                }

                let offset = grid_y * grid_width + grid_x;
                let stride = grid_width;

                // SAFETY: check_flags makes sure that the subgrids are disjoint.
                let subgrid = unsafe {
                    CutGrid {
                        ptr: NonNull::new_unchecked(ptr.as_ptr().add(offset)),
                        width: block_width,
                        height: block_height,
                        stride,
                        _marker: Default::default(),
                    }
                };
                subgrids.insert((x, y), subgrid);
            }

            (idx, subgrids)
        })
        .collect()
}

impl<'g, Lane: SimdLane> CutGrid<'g, Lane> {
    pub fn convert_grid(grid: &'g mut CutGrid<'_, f32>) -> Option<Self> {
        let mask = Lane::SIZE - 1;
        let align_mask = std::mem::align_of::<Lane>() - 1;

        (
            grid.ptr.as_ptr() as usize & align_mask == 0 &&
            grid.width & mask == 0 &&
            grid.stride & mask == 0
        ).then(|| Self {
            ptr: grid.ptr.cast::<Lane>(),
            width: grid.width / Lane::SIZE,
            height: grid.height,
            stride: grid.stride / Lane::SIZE,
            _marker: Default::default(),
        })
    }
}
