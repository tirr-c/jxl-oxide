use jxl_grid::{AlignedGrid, MutableSubgrid};
use jxl_threadpool::JxlThreadPool;

use crate::{ImageWithRegion, Region};

use super::impls::generic::gabor::gabor_row_edge;

pub fn apply_gabor_like(
    fb: &mut ImageWithRegion,
    color_padded_region: Region,
    fb_scratch: &mut [AlignedGrid<f32>; 3],
    weights: [[f32; 2]; 3],
    pool: &jxl_threadpool::JxlThreadPool,
) {
    tracing::debug!("Running gaborish");
    let region = fb.regions_and_shifts()[0].0;
    assert!(region.contains(color_padded_region));
    let left = region.left.abs_diff(color_padded_region.left) as usize;
    let top = region.top.abs_diff(color_padded_region.top) as usize;
    let right = left + color_padded_region.width as usize;
    let bottom = top + color_padded_region.height as usize;

    let buffers = fb.as_color_floats_mut();
    let buffers = buffers.map(|g| g.as_subgrid_mut().subgrid(left..right, top..bottom));

    super::impls::apply_gabor_like(buffers, fb_scratch, weights, pool);

    let left = color_padded_region.left;
    let top = color_padded_region.top;
    for (idx, grid) in fb_scratch.iter_mut().enumerate() {
        let width = grid.width() as u32;
        let height = grid.height() as u32;
        let region = Region {
            width,
            height,
            left,
            top,
        };
        fb.swap_channel_f32(idx, grid, region);
    }
}

pub(super) struct GaborRow<'buf> {
    pub input_rows: [&'buf [f32]; 3],
    pub output_row: &'buf mut [f32],
    pub weights: [f32; 2],
}

pub(super) fn run_gabor_rows<'buf>(
    input: MutableSubgrid<'buf, f32>,
    output: &'buf mut AlignedGrid<f32>,
    weights: [f32; 2],
    pool: &JxlThreadPool,
    handle_row: for<'a> fn(GaborRow<'a>),
) {
    unsafe { run_gabor_rows_unsafe(input, output, weights, pool, handle_row) }
}

pub(super) unsafe fn run_gabor_rows_unsafe<'buf>(
    input: MutableSubgrid<'buf, f32>,
    output: &'buf mut AlignedGrid<f32>,
    weights: [f32; 2],
    pool: &JxlThreadPool,
    handle_row: for<'a> unsafe fn(GaborRow<'a>),
) {
    let width = input.width();
    let height = input.height();
    let output_stride = output.width();
    let output_buf = output.buf_mut();
    assert!(output_stride >= width);

    if height == 1 {
        let input_buf = input.get_row(0);
        gabor_row_edge(input_buf, None, &mut output_buf[..width], weights);
        return;
    }

    {
        let input_buf_c = input.get_row(0);
        let input_buf_a = input.get_row(1);
        let output_buf = &mut output_buf[..width];
        gabor_row_edge(input_buf_c, Some(input_buf_a), output_buf, weights);
    }

    let (inner_rows, bottom_row) =
        output_buf[output_stride..].split_at_mut((height - 2) * output_stride);
    let output_rows = inner_rows
        .chunks_mut(output_stride * 8)
        .enumerate()
        .collect::<Vec<_>>();

    pool.for_each_vec(output_rows, |(y8, output_rows)| {
        let it = output_rows.chunks_exact_mut(output_stride);
        for (dy, output_row) in it.enumerate() {
            let y_up = y8 * 8 + dy;
            let input_rows = [
                input.get_row(y_up),
                input.get_row(y_up + 1),
                input.get_row(y_up + 2),
            ];
            let output_row = &mut output_row[..width];
            let row = GaborRow {
                input_rows,
                output_row,
                weights,
            };
            handle_row(row);
        }
    });

    {
        let input_buf_c = input.get_row(height - 1);
        let input_buf_a = input.get_row(height - 2);
        let output_buf = &mut bottom_row[..width];
        gabor_row_edge(input_buf_c, Some(input_buf_a), output_buf, weights);
    }
}

#[allow(unused)]
pub(crate) fn run_gabor_row_generic(row: GaborRow) {
    super::impls::generic::gabor::run_gabor_row_generic(row)
}
