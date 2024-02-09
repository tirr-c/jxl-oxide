#![allow(dead_code)]

use jxl_frame::{filter::EpfParams, FrameHeader};
use jxl_grid::SimpleGrid;
use jxl_threadpool::JxlThreadPool;

use crate::{util, Region, Result};

pub(crate) mod epf;

pub(crate) struct EpfRow<'buf, 'epf> {
    pub(crate) input_rows: [[&'buf [f32]; 7]; 3],
    pub(crate) merged_input_rows: Option<[&'buf [f32]; 3]>,
    pub(crate) output_rows: [&'buf mut [f32]; 3],
    pub(crate) width: usize,
    pub(crate) x: usize,
    pub(crate) y: usize,
    pub(crate) sigma_row: &'buf [f32],
    pub(crate) epf_params: &'epf EpfParams,
    pub(crate) skip_inner: bool,
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn epf_common<'buf>(
    input: &'buf [SimpleGrid<f32>; 3],
    output: &'buf mut [SimpleGrid<f32>; 3],
    frame_header: &FrameHeader,
    sigma_grid_map: &[Option<&SimpleGrid<f32>>],
    region: Region,
    epf_params: &EpfParams,
    pool: &JxlThreadPool,
    handle_row_simd: Option<for<'a, 'b> unsafe fn(EpfRow<'a, 'b>)>,
    handle_row_generic: for<'a, 'b> fn(EpfRow<'a, 'b>),
) {
    struct EpfJob<'buf> {
        base_y: usize,
        output0: &'buf mut [f32],
        output1: &'buf mut [f32],
        output2: &'buf mut [f32],
    }

    let width = region.width as usize;
    let height = region.height as usize;
    assert!(region.left >= 0);
    assert!(region.top >= 0);
    let left = region.left as usize;
    let top = region.top as usize;

    let input_buf = [input[0].buf(), input[1].buf(), input[2].buf()];
    let (mut output0, mut output1, mut output2) = {
        let [a, b, c] = output;
        (a.buf_mut(), b.buf_mut(), c.buf_mut())
    };

    assert_eq!(input_buf[0].len(), width * height);
    assert_eq!(input_buf[1].len(), width * height);
    assert_eq!(input_buf[2].len(), width * height);
    assert_eq!(output0.len(), width * height);
    assert_eq!(output1.len(), width * height);
    assert_eq!(output2.len(), width * height);

    let mut jobs = Vec::new();
    let mut image_y = top;
    while image_y < top + height {
        let next = ((image_y + 8) & !7).min(top + height);
        let job_height = next - image_y;

        let (job_output0, next_output0) = output0.split_at_mut(job_height * width);
        let (job_output1, next_output1) = output1.split_at_mut(job_height * width);
        let (job_output2, next_output2) = output2.split_at_mut(job_height * width);
        jobs.push(EpfJob {
            base_y: image_y - top,
            output0: job_output0,
            output1: job_output1,
            output2: job_output2,
        });

        output0 = next_output0;
        output1 = next_output1;
        output2 = next_output2;
        image_y = next;
    }

    let sigma_group_dim_shift = frame_header.group_dim().trailing_zeros();
    let sigma_group_dim_mask = (frame_header.group_dim() - 1) as usize;
    let groups_per_row = frame_header.lf_groups_per_row() as usize;
    let sigma_len = (left + width + 7) / 8 - left / 8;
    pool.for_each_vec_with(
        jobs,
        vec![epf_params.sigma_for_modular; sigma_len],
        |sigma_row, EpfJob { base_y, output0, output1, output2 }| {
            let sigma_y = (top + base_y) / 8;
            let sigma_group_y = sigma_y >> sigma_group_dim_shift;
            let sigma_inner_y = sigma_y & sigma_group_dim_mask;

            for (dx, sigma) in sigma_row.iter_mut().enumerate() {
                let sigma_x = left / 8 + dx;
                let sigma_group_x = sigma_x >> sigma_group_dim_shift;
                let sigma_inner_x = sigma_x & sigma_group_dim_mask;
                let sigma_grid_idx = sigma_group_y * groups_per_row + sigma_group_x;
                if let Some(grid) = sigma_grid_map[sigma_grid_idx] {
                    let width = grid.width();
                    *sigma = grid.buf()[sigma_inner_y * width + sigma_inner_x];
                }
            }

            let it = output0
                .chunks_exact_mut(width)
                .zip(output1.chunks_exact_mut(width))
                .zip(output2.chunks_exact_mut(width));
            for (dy, ((output0, output1), output2)) in it.enumerate() {
                let y = base_y + dy;
                let input_rows: [[_; 7]; 3] = std::array::from_fn(|c| {
                    std::array::from_fn(|idx| {
                        let y = util::mirror((y + idx) as isize - 3, height);
                        &input_buf[c][y * width..][..width]
                    })
                });
                let merged_input_rows: Option<[_; 3]> = if y >= 3 && y < height - 3 {
                    Some(std::array::from_fn(|c| {
                        &input_buf[c][(y - 3) * width..][..7 * width]
                    }))
                } else {
                    None
                };

                let image_y = top + base_y + dy;

                let mut skip_inner = false;
                if merged_input_rows.is_some() {
                    if let Some(handle_row_simd) = handle_row_simd {
                        skip_inner = true;
                        let output_rows = [&mut *output0, &mut *output1, &mut *output2];
                        let row = EpfRow {
                            input_rows,
                            merged_input_rows,
                            output_rows,
                            width,
                            x: left,
                            y: image_y,
                            sigma_row,
                            epf_params,
                            skip_inner,
                        };
                        handle_row_simd(row);
                    }
                }

                let output_rows = [output0, output1, output2];
                let row = EpfRow {
                    input_rows,
                    output_rows,
                    merged_input_rows,
                    width,
                    x: left,
                    y: image_y,
                    sigma_row,
                    epf_params,
                    skip_inner,
                };
                handle_row_generic(row);
            }
        },
    );
}

pub fn epf_step0(
    input: &[SimpleGrid<f32>; 3],
    output: &mut [SimpleGrid<f32>; 3],
    frame_header: &FrameHeader,
    sigma_grid_map: &[Option<&SimpleGrid<f32>>],
    region: Region,
    epf_params: &EpfParams,
    pool: &JxlThreadPool,
) {
    unsafe {
        epf_common(
            input,
            output,
            frame_header,
            sigma_grid_map,
            region,
            epf_params,
            pool,
            None,
            epf::epf_row::<0>,
        )
    }
}

pub fn epf_step1(
    input: &[SimpleGrid<f32>; 3],
    output: &mut [SimpleGrid<f32>; 3],
    frame_header: &FrameHeader,
    sigma_grid_map: &[Option<&SimpleGrid<f32>>],
    region: Region,
    epf_params: &EpfParams,
    pool: &JxlThreadPool,
) {
    unsafe {
        epf_common(
            input,
            output,
            frame_header,
            sigma_grid_map,
            region,
            epf_params,
            pool,
            None,
            epf::epf_row::<1>,
        )
    }
}

pub fn epf_step2(
    input: &[SimpleGrid<f32>; 3],
    output: &mut [SimpleGrid<f32>; 3],
    frame_header: &FrameHeader,
    sigma_grid_map: &[Option<&SimpleGrid<f32>>],
    region: Region,
    epf_params: &EpfParams,
    pool: &JxlThreadPool,
) {
    unsafe {
        epf_common(
            input,
            output,
            frame_header,
            sigma_grid_map,
            region,
            epf_params,
            pool,
            None,
            epf::epf_row::<2>,
        )
    }
}

pub fn apply_gabor_like(fb: [&mut SimpleGrid<f32>; 3], weights_xyb: [[f32; 2]; 3]) -> Result<()> {
    for (fb, [weight1, weight2]) in fb.into_iter().zip(weights_xyb) {
        run_gabor_inner(fb, weight1, weight2)?;
    }
    Ok(())
}

#[inline(always)]
pub fn run_gabor_inner(
    fb: &mut jxl_grid::SimpleGrid<f32>,
    weight1: f32,
    weight2: f32,
) -> Result<()> {
    let global_weight = (1.0 + weight1 * 4.0 + weight2 * 4.0).recip();

    let tracker = fb.tracker();
    let width = fb.width();
    let height = fb.height();
    if width * height <= 1 {
        return Ok(());
    }

    let mut input = SimpleGrid::with_alloc_tracker(width, height + 2, tracker.as_ref())?;
    let input = input.buf_mut();
    input[width..][..width * height].copy_from_slice(fb.buf());
    input[..width].copy_from_slice(&fb.buf()[..width]);
    input[width * (height + 1)..][..width]
        .copy_from_slice(&fb.buf()[width * (height - 1)..][..width]);

    let input = &*input;
    let output = fb.buf_mut();

    if width == 1 {
        for idx in 0..height {
            output[idx] = (input[idx + 1]
                + (input[idx] + input[idx + 1] + input[idx + 1] + input[idx + 2]) * weight1
                + (input[idx] + input[idx + 2]) * weight2 * 2.0)
                * global_weight;
        }
        return Ok(());
    }

    let len = width * height - 2;
    let center = &input[width + 1..][..len];
    let sides = [
        &input[1..][..len],
        &input[width..][..len],
        &input[width + 2..][..len],
        &input[width * 2 + 1..][..len],
    ];
    let diags = [
        &input[..len],
        &input[2..][..len],
        &input[width * 2..][..len],
        &input[width * 2 + 2..][..len],
    ];

    for (idx, out) in output[1..][..len].iter_mut().enumerate() {
        *out = (center[idx]
            + (sides[0][idx] + sides[1][idx] + sides[2][idx] + sides[3][idx]) * weight1
            + (diags[0][idx] + diags[1][idx] + diags[2][idx] + diags[3][idx]) * weight2)
            * global_weight;
    }

    // left side
    let center = &input[width..];
    let sides = [
        input,
        &input[width..],
        &input[width + 1..],
        &input[width * 2..],
    ];
    let diags = [
        input,
        &input[1..],
        &input[width * 2..],
        &input[width * 2 + 1..],
    ];
    for idx in 0..height {
        let offset = idx * width;
        output[offset] = (center[offset]
            + (sides[0][offset] + sides[1][offset] + sides[2][offset] + sides[3][offset])
                * weight1
            + (diags[0][offset] + diags[1][offset] + diags[2][offset] + diags[3][offset])
                * weight2)
            * global_weight;
    }

    // right side
    let center = &input[width * 2 - 1..];
    let sides = [
        &input[width - 1..],
        &input[width * 2 - 2..],
        &input[width * 2 - 1..],
        &input[width * 3 - 1..],
    ];
    let diags = [
        &input[width - 2..],
        &input[width - 1..],
        &input[width * 3 - 2..],
        &input[width * 3 - 1..],
    ];
    for idx in 0..height {
        let offset = idx * width;
        output[width - 1 + offset] = (center[offset]
            + (sides[0][offset] + sides[1][offset] + sides[2][offset] + sides[3][offset])
                * weight1
            + (diags[0][offset] + diags[1][offset] + diags[2][offset] + diags[3][offset])
                * weight2)
            * global_weight;
    }
    Ok(())
}
