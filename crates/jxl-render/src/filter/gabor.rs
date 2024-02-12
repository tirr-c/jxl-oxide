use jxl_frame::FrameHeader;
use jxl_grid::SimpleGrid;
use jxl_threadpool::JxlThreadPool;

use crate::{ImageWithRegion, Region};

use super::impls::generic::gabor::gabor_row_edge;

pub fn apply_gabor_like(
    fb: &mut ImageWithRegion,
    fb_scratch: &mut [SimpleGrid<f32>; 3],
    frame_header: &FrameHeader,
    weights: [[f32; 2]; 3],
    pool: &jxl_threadpool::JxlThreadPool,
) {
    tracing::debug!("Running gaborish");
    let region = fb.region();
    let fb = <&mut [_; 3]>::try_from(fb.buffer_mut()).unwrap();
    super::impls::apply_gabor_like(fb, fb_scratch, frame_header, region, weights, pool);
    fb.swap_with_slice(fb_scratch);
}

pub(super) struct GaborRow<'buf> {
    pub input_rows: &'buf [f32],
    pub input_start: usize,
    pub input_stride: usize,
    pub output_row: &'buf mut [f32],
    pub weights: [f32; 2],
}

pub(super) fn run_gabor_rows<'buf>(
    input: &'buf SimpleGrid<f32>,
    output: &'buf mut SimpleGrid<f32>,
    frame_header: &FrameHeader,
    region: Region,
    weights: [f32; 2],
    pool: &JxlThreadPool,
    handle_row: for<'a> fn(GaborRow<'a>),
) {
    unsafe {
        run_gabor_rows_unsafe(
            input,
            output,
            frame_header,
            region,
            weights,
            pool,
            handle_row,
        )
    }
}

pub(super) unsafe fn run_gabor_rows_unsafe<'buf>(
    input: &'buf SimpleGrid<f32>,
    output: &'buf mut SimpleGrid<f32>,
    frame_header: &FrameHeader,
    region: Region,
    weights: [f32; 2],
    pool: &JxlThreadPool,
    handle_row: for<'a> unsafe fn(GaborRow<'a>),
) {
    let stride = input.width();
    let height = input.height();
    let input_buf = input.buf();
    let output_buf = output.buf_mut();
    assert_eq!(input_buf.len(), stride * height);
    assert_eq!(output_buf.len(), stride * height);

    let full_frame_region = Region::with_size(
        frame_header.color_sample_width(),
        frame_header.color_sample_height(),
    );
    let actual_region = full_frame_region.intersection(region);
    let start_x = region.left.abs_diff(actual_region.left) as usize;
    let start_y = region.top.abs_diff(actual_region.top) as usize;
    let actual_width = actual_region.width as usize;
    let actual_height = actual_region.height as usize;

    if actual_height == 0 || actual_width == 0 {
        return;
    }

    let input_buf = &input_buf[start_y * stride..];
    let output_buf = &mut output_buf[start_y * stride..];

    if actual_height == 1 {
        if actual_width == 1 {
            return;
        }

        let input_buf = &input_buf[start_x..][..actual_width];
        let output_buf = &mut output_buf[start_x..][..actual_width];
        gabor_row_edge(input_buf, None, output_buf, weights);
        return;
    }

    {
        let input_buf_c = &input_buf[start_x..][..actual_width];
        let input_buf_a = &input_buf[stride + start_x..][..actual_width];
        let output_buf = &mut output_buf[start_x..][..actual_width];
        gabor_row_edge(input_buf_c, Some(input_buf_a), output_buf, weights);
    }

    let (inner_rows, bottom_row) = output_buf[stride..].split_at_mut((actual_height - 2) * stride);
    let output_rows = inner_rows
        .chunks_mut(stride * 8)
        .enumerate()
        .collect::<Vec<_>>();

    pool.for_each_vec(output_rows, |(y8, output_rows)| {
        let it = output_rows.chunks_exact_mut(stride);
        for (dy, output_row) in it.enumerate() {
            let y_up = y8 * 8 + dy;
            let input_rows = &input_buf[y_up * stride..][..stride * 3];
            let output_row = &mut output_row[start_x..][..actual_width];
            let row = GaborRow {
                input_rows,
                input_start: start_x,
                input_stride: stride,
                output_row,
                weights,
            };
            handle_row(row);
        }
    });

    {
        let input_buf_c = &input_buf[(actual_height - 1) * stride + start_x..][..actual_width];
        let input_buf_a = &input_buf[(actual_height - 2) * stride + start_x..][..actual_width];
        let output_buf = &mut bottom_row[start_x..][..actual_width];
        gabor_row_edge(input_buf_c, Some(input_buf_a), output_buf, weights);
    }
}

pub(crate) fn run_gabor_row_generic(row: GaborRow) {
    super::impls::generic::gabor::run_gabor_row_generic(row)
}
