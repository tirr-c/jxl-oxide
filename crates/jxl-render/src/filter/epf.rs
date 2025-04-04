use std::collections::HashMap;

use jxl_frame::{data::LfGroup, filter::EpfParams, FrameHeader};
use jxl_grid::{AlignedGrid, MutableSubgrid, SharedSubgrid};
use jxl_modular::Sample;
use jxl_threadpool::JxlThreadPool;

use crate::{util, ImageWithRegion, Region};

pub fn apply_epf<S: Sample>(
    fb_image: &mut ImageWithRegion,
    mut fb_scratch_arr: [AlignedGrid<f32>; 3],
    color_padded_region: Region,
    lf_groups: &HashMap<u32, LfGroup<S>>,
    frame_header: &FrameHeader,
    epf_params: &EpfParams,
    pool: &jxl_threadpool::JxlThreadPool,
) {
    let iters = epf_params.iters;

    let span = tracing::span!(tracing::Level::TRACE, "Edge-preserving filter");
    let _guard = span.enter();

    let region = fb_image.regions_and_shifts()[0].0;
    assert!(region.contains(color_padded_region));
    let left = region.left.abs_diff(color_padded_region.left) as usize;
    let top = region.top.abs_diff(color_padded_region.top) as usize;
    let right = left + color_padded_region.width as usize;
    let bottom = top + color_padded_region.height as usize;

    let fb = fb_image.as_color_floats_mut();
    let mut fb = fb.map(|g| g.as_subgrid_mut().subgrid(left..right, top..bottom));
    let mut fb_scratch = fb_scratch_arr.each_mut().map(|g| g.as_subgrid_mut());

    let num_lf_groups = frame_header.num_lf_groups() as usize;
    let mut sigma_grid_map = vec![None::<&AlignedGrid<f32>>; num_lf_groups];

    for (&lf_group_idx, lf_group) in lf_groups {
        if let Some(hf_meta) = &lf_group.hf_meta {
            sigma_grid_map[lf_group_idx as usize] = Some(&hf_meta.epf_sigma);
        }
    }

    // Step 0
    if iters == 3 {
        tracing::debug!("Running step 0");
        super::impls::epf::<0>(
            &mut fb,
            &mut fb_scratch,
            color_padded_region,
            frame_header,
            &sigma_grid_map,
            epf_params,
            pool,
        );
        fb.swap_with_slice(&mut fb_scratch);
    }

    // Step 1
    {
        tracing::debug!("Running step 1");
        super::impls::epf::<1>(
            &mut fb,
            &mut fb_scratch,
            color_padded_region,
            frame_header,
            &sigma_grid_map,
            epf_params,
            pool,
        );
        fb.swap_with_slice(&mut fb_scratch);
    }

    // Step 2
    if iters >= 2 {
        tracing::debug!("Running step 2");
        super::impls::epf::<2>(
            &mut fb,
            &mut fb_scratch,
            color_padded_region,
            frame_header,
            &sigma_grid_map,
            epf_params,
            pool,
        );
        fb.swap_with_slice(&mut fb_scratch);
    }

    if iters == 1 || iters == 3 {
        let left = color_padded_region.left;
        let top = color_padded_region.top;
        for (idx, grid) in fb_scratch_arr.into_iter().enumerate() {
            let width = grid.width() as u32;
            let height = grid.height() as u32;
            let region = Region {
                width,
                height,
                left,
                top,
            };
            fb_image.replace_channel(idx, crate::ImageBuffer::F32(grid), region);
        }
    }
}

pub(super) struct EpfRow<'buf, 'epf> {
    pub input_rows: [[&'buf [f32]; 7]; 3],
    #[allow(unused)]
    pub merged_input_rows: Option<[SharedSubgrid<'buf, f32>; 3]>,
    pub output_rows: [&'buf mut [f32]; 3],
    pub width: usize,
    pub y: usize,
    pub sigma_row: &'buf [f32],
    pub epf_params: &'epf EpfParams,
    pub skip_inner: bool,
}

#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn run_epf_rows(
    input: &mut [MutableSubgrid<f32>; 3],
    output: &mut [MutableSubgrid<f32>; 3],
    color_padded_region: Region,
    frame_header: &FrameHeader,
    sigma_grid_map: &[Option<&AlignedGrid<f32>>],
    epf_params: &EpfParams,
    pool: &JxlThreadPool,
    handle_row_simd: Option<for<'a, 'b> unsafe fn(EpfRow<'a, 'b>)>,
    handle_row_generic: for<'a, 'b> fn(EpfRow<'a, 'b>),
) {
    struct EpfJob<'buf> {
        base_y: usize,
        output0: MutableSubgrid<'buf, f32>,
        output1: MutableSubgrid<'buf, f32>,
        output2: MutableSubgrid<'buf, f32>,
    }

    let input = input.each_ref().map(|g| g.as_shared());
    let [mut output0, mut output1, mut output2] = output.each_mut().map(|g| g.borrow_mut());
    let Region {
        left,
        top,
        width,
        height,
    } = color_padded_region;
    assert!(left >= 0 && top >= 0);
    assert!(left % 8 == 0);
    assert!(top % 8 == 0);

    let width = width as usize;
    let height = height as usize;
    let left = left as usize;
    let top = top as usize;

    let mut jobs = Vec::new();
    for dy in (0..height).step_by(8) {
        let image_y = top + dy;
        let job_height = (height - dy).min(8);

        let next_output0 = output0.split_vertical_in_place(job_height);
        let next_output1 = output1.split_vertical_in_place(job_height);
        let next_output2 = output2.split_vertical_in_place(job_height);
        jobs.push(EpfJob {
            base_y: image_y - top,
            output0,
            output1,
            output2,
        });

        output0 = next_output0;
        output1 = next_output1;
        output2 = next_output2;
    }

    let sigma_group_dim_shift = frame_header.group_dim().trailing_zeros();
    let sigma_group_dim_mask = (frame_header.group_dim() - 1) as usize;
    let groups_per_row = frame_header.lf_groups_per_row() as usize;
    let sigma_len = width.div_ceil(8);
    pool.for_each_vec_with(
        jobs,
        vec![epf_params.sigma_for_modular; sigma_len],
        |sigma_row, job| {
            let EpfJob {
                base_y,
                mut output0,
                mut output1,
                mut output2,
            } = job;
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

            let job_height = output0.height();
            for dy in 0..job_height {
                let y = base_y + dy;
                let input_rows: [[_; 7]; 3] = std::array::from_fn(|c| {
                    std::array::from_fn(|idx| {
                        let y = util::mirror((y + idx) as isize - 3, height);
                        input[c].get_row(y)
                    })
                });
                let merged_input_rows = if y >= 3 && y + 4 <= height {
                    Some(std::array::from_fn(|c| {
                        input[c].subgrid(.., (y - 3)..(y + 4))
                    }))
                } else {
                    None
                };

                let output0 = output0.get_row_mut(dy);
                let output1 = output1.get_row_mut(dy);
                let output2 = output2.get_row_mut(dy);
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
