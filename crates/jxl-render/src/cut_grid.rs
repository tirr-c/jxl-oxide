use std::{ptr::NonNull, collections::HashMap};

use jxl_grid::{CutGrid, Grid, SimpleGrid};
use jxl_modular::ChannelShift;
use jxl_vardct::HfCoeff;

pub fn make_quant_cut_grid<'g>(
    buf: &'g mut SimpleGrid<f32>,
    left: usize,
    top: usize,
    shift: ChannelShift,
    channel_data: &Grid<i32>,
) -> CutGrid<'g, f32> {
    let left = left >> shift.hshift();
    let top = top >> shift.vshift();
    let width = channel_data.width();
    let height = channel_data.height();
    let stride = buf.width();
    let buf = &mut buf.buf_mut()[top * stride + left..];
    let mut grid = CutGrid::from_buf(buf, width, height, stride);
    if let Some(channel_data) = channel_data.as_simple() {
        let buf = channel_data.buf();
        for y in 0..height {
            let row = grid.get_row_mut(y);
            let quant = &buf[y * width..][..width];
            for (out, &q) in row.iter_mut().zip(quant) {
                *out = q as f32;
            }
        }
    } else {
        for y in 0..height {
            let row = grid.get_row_mut(y);
            for (x, out) in row.iter_mut().enumerate() {
                *out = *channel_data.get(x, y).unwrap() as f32;
            }
        }
    }
    grid
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
            for coeff in &group.data {
                let x = coeff.bx;
                let y = coeff.by;
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
                    panic!(
                        "Invalid group_coeffs? \
                        grid_x={grid_x}, grid_y={grid_y}, \
                        block_width={block_width}, block_height={block_height}, \
                        grid_width={grid_width}, grid_height={grid_height}"
                    );
                }

                let offset = grid_y * grid_width + grid_x;
                let stride = grid_width;

                // SAFETY: check_flags makes sure that the subgrids are disjoint.
                let subgrid = unsafe {
                    CutGrid::new(
                        NonNull::new_unchecked(ptr.as_ptr().add(offset)),
                        block_width,
                        block_height,
                        stride,
                    )
                };
                subgrids.insert((x, y), subgrid);
            }

            (idx, subgrids)
        })
        .collect()
}
