use std::{ptr::NonNull, collections::HashMap};

use jxl_frame::data::HfCoeff;
use jxl_grid::{CutGrid, SimpleGrid};
use jxl_modular::ChannelShift;

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
