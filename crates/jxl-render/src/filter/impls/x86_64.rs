use jxl_frame::{filter::EpfParams, FrameHeader};
use jxl_grid::{AlignedGrid, MutableSubgrid};
use jxl_threadpool::JxlThreadPool;

use crate::{
    filter::{
        epf::run_epf_rows,
        gabor::{run_gabor_row_generic, run_gabor_rows, run_gabor_rows_unsafe},
    },
    Region,
};

mod epf_sse41;
mod gabor_avx2;

pub fn epf<const STEP: usize>(
    input: &mut [MutableSubgrid<f32>; 3],
    output: &mut [MutableSubgrid<f32>; 3],
    color_padded_region: Region,
    frame_header: &FrameHeader,
    sigma_grid_map: &[Option<&AlignedGrid<f32>>],
    epf_params: &EpfParams,
    pool: &JxlThreadPool,
) {
    if is_x86_feature_detected!("sse4.1") {
        // SAFETY: Features are checked above.
        unsafe {
            return run_epf_rows(
                input,
                output,
                color_padded_region,
                frame_header,
                sigma_grid_map,
                epf_params,
                pool,
                Some(epf_sse41::epf_row_x86_64_sse41::<STEP>),
                super::generic::epf::epf_row::<STEP>,
            );
        }
    }

    unsafe {
        run_epf_rows(
            input,
            output,
            color_padded_region,
            frame_header,
            sigma_grid_map,
            epf_params,
            pool,
            None,
            super::generic::epf::epf_row::<STEP>,
        )
    }
}

pub fn apply_gabor_like(
    fb: [MutableSubgrid<f32>; 3],
    fb_scratch: &mut [AlignedGrid<f32>; 3],
    weights: [[f32; 2]; 3],
    pool: &jxl_threadpool::JxlThreadPool,
) {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        // SAFETY: Features are checked above.
        unsafe {
            for ((input, output), weights) in fb.into_iter().zip(fb_scratch).zip(weights) {
                run_gabor_rows_unsafe(
                    input,
                    output,
                    weights,
                    pool,
                    gabor_avx2::run_gabor_row_x86_64_avx2,
                );
            }
        }
        return;
    }

    for ((input, output), weights) in fb.into_iter().zip(fb_scratch).zip(weights) {
        run_gabor_rows(input, output, weights, pool, run_gabor_row_generic);
    }
}
