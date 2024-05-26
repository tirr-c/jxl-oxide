use std::arch::is_aarch64_feature_detected;

use jxl_frame::{filter::EpfParams, FrameHeader};
use jxl_grid::{AlignedGrid, MutableSubgrid};
use jxl_threadpool::JxlThreadPool;

use crate::filter::epf::run_epf_rows;
use crate::filter::gabor::{run_gabor_row_generic, run_gabor_rows, run_gabor_rows_unsafe};
use crate::Region;

mod epf;
mod gabor;

use gabor::run_gabor_row_aarch64_neon;

pub fn epf<const STEP: usize>(
    input: &mut [MutableSubgrid<f32>; 3],
    output: &mut [MutableSubgrid<f32>; 3],
    color_padded_region: Region,
    frame_header: &FrameHeader,
    sigma_grid_map: &[Option<&AlignedGrid<f32>>],
    epf_params: &EpfParams,
    pool: &JxlThreadPool,
) {
    if is_aarch64_feature_detected!("neon") {
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
                Some(epf::epf_row_aarch64_neon::<STEP>),
                super::generic::epf::epf_row::<STEP>,
            );
        }
    }

    // SAFETY: row handler is safe.
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
    if is_aarch64_feature_detected!("neon") {
        // SAFETY: Features are checked above.
        unsafe {
            for ((input, output), weights) in fb.into_iter().zip(fb_scratch).zip(weights) {
                run_gabor_rows_unsafe(input, output, weights, pool, run_gabor_row_aarch64_neon);
            }
        }
        return;
    }

    for ((input, output), weights) in fb.into_iter().zip(fb_scratch).zip(weights) {
        run_gabor_rows(input, output, weights, pool, run_gabor_row_generic);
    }
}
