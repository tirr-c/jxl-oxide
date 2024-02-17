use std::collections::HashMap;

use jxl_frame::{data::LfGroup, filter::EpfParams, FrameHeader};
use jxl_grid::SimpleGrid;
use jxl_modular::Sample;
use jxl_threadpool::JxlThreadPool;

use crate::{util, ImageWithRegion, Region};

pub fn apply_epf<S: Sample>(
    fb: &mut ImageWithRegion,
    fb_scratch: &mut [SimpleGrid<f32>; 3],
    lf_groups: &HashMap<u32, LfGroup<S>>,
    frame_header: &FrameHeader,
    epf_params: &EpfParams,
    pool: &jxl_threadpool::JxlThreadPool,
) {
    let iters = epf_params.iters;

    let span = tracing::span!(tracing::Level::TRACE, "Edge-preserving filter");
    let _guard = span.enter();

    let region = fb.region();
    let fb = <&mut [_; 3]>::try_from(fb.buffer_mut()).unwrap();

    let num_lf_groups = frame_header.num_lf_groups() as usize;
    let mut sigma_grid_map = vec![None::<&SimpleGrid<f32>>; num_lf_groups];

    for (&lf_group_idx, lf_group) in lf_groups {
        if let Some(hf_meta) = &lf_group.hf_meta {
            sigma_grid_map[lf_group_idx as usize] = Some(&hf_meta.epf_sigma);
        }
    }

    // Step 0
    if iters == 3 {
        tracing::debug!("Running step 0");
        super::impls::epf::<0>(
            fb,
            fb_scratch,
            frame_header,
            &sigma_grid_map,
            region,
            epf_params,
            pool,
        );
        std::mem::swap(&mut fb[0], &mut fb_scratch[0]);
        std::mem::swap(&mut fb[1], &mut fb_scratch[1]);
        std::mem::swap(&mut fb[2], &mut fb_scratch[2]);
    }

    // Step 1
    {
        tracing::debug!("Running step 1");
        super::impls::epf::<1>(
            fb,
            fb_scratch,
            frame_header,
            &sigma_grid_map,
            region,
            epf_params,
            pool,
        );
        std::mem::swap(&mut fb[0], &mut fb_scratch[0]);
        std::mem::swap(&mut fb[1], &mut fb_scratch[1]);
        std::mem::swap(&mut fb[2], &mut fb_scratch[2]);
    }

    // Step 2
    if iters >= 2 {
        tracing::debug!("Running step 2");
        super::impls::epf::<2>(
            fb,
            fb_scratch,
            frame_header,
            &sigma_grid_map,
            region,
            epf_params,
            pool,
        );
        std::mem::swap(&mut fb[0], &mut fb_scratch[0]);
        std::mem::swap(&mut fb[1], &mut fb_scratch[1]);
        std::mem::swap(&mut fb[2], &mut fb_scratch[2]);
    }
}

pub(super) struct EpfRow<'buf, 'epf> {
    pub input_rows: [[&'buf [f32]; 7]; 3],
    pub merged_input_rows: Option<[&'buf [f32]; 3]>,
    pub output_rows: [&'buf mut [f32]; 3],
    pub width: usize,
    pub x: usize,
    pub y: usize,
    pub sigma_row: &'buf [f32],
    pub epf_params: &'epf EpfParams,
    pub skip_inner: bool,
}

#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn run_epf_rows<'buf>(
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
        |sigma_row,
         EpfJob {
             base_y,
             output0,
             output1,
             output2,
         }| {
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

pub(super) const fn epf_kernel_offsets<const STEP: usize>() -> &'static [(isize, isize)] {
    const EPF_KERNEL_1: [(isize, isize); 4] = [(0, -1), (0, 1), (-1, 0), (1, 0)];
    #[rustfmt::skip]
    const EPF_KERNEL_2: [(isize, isize); 12] = [
        (0, -2), (-1, -1), (0, -1), (1, -1),
        (-2, 0), (-1, 0), (1, 0), (2, 0),
        (-1, 1), (0, 1), (1, 1), (0, 2),
    ];

    if STEP == 0 {
        &EPF_KERNEL_2
    } else if STEP == 1 || STEP == 2 {
        &EPF_KERNEL_1
    } else {
        panic!()
    }
}

pub(super) const fn epf_dist_offsets<const STEP: usize>() -> &'static [(isize, isize)] {
    if STEP == 0 {
        &[(0, -1), (1, 0), (0, 0), (-1, 0), (0, 1)]
    } else if STEP == 1 {
        &[(0, -1), (0, 0), (0, 1), (-1, 0), (1, 0)]
    } else if STEP == 2 {
        &[(0, 0)]
    } else {
        panic!()
    }
}
