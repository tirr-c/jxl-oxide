use std::arch::is_aarch64_feature_detected;

use jxl_frame::{filter::EpfParams, FrameHeader};
use jxl_grid::SimpleGrid;
use jxl_threadpool::JxlThreadPool;

use crate::filter::epf::run_epf_rows;
use crate::filter::gabor::{run_gabor_row_generic, run_gabor_rows, run_gabor_rows_unsafe};
use crate::Region;

mod epf;
mod gabor;

use gabor::run_gabor_row_aarch64_neon;

pub fn epf<const STEP: usize>(
    input: &[SimpleGrid<f32>; 3],
    output: &mut [SimpleGrid<f32>; 3],
    frame_header: &FrameHeader,
    sigma_grid_map: &[Option<&SimpleGrid<f32>>],
    region: Region,
    epf_params: &EpfParams,
    pool: &JxlThreadPool,
) {
    if is_aarch64_feature_detected!("neon") {
        // SAFETY: Features are checked above.
        unsafe {
            return run_epf_rows(
                input,
                output,
                frame_header,
                sigma_grid_map,
                region,
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
            frame_header,
            sigma_grid_map,
            region,
            epf_params,
            pool,
            None,
            super::generic::epf::epf_row::<STEP>,
        )
    }
}

pub fn apply_gabor_like(
    fb: &[SimpleGrid<f32>; 3],
    fb_scratch: &mut [SimpleGrid<f32>; 3],
    frame_header: &FrameHeader,
    region: Region,
    weights: [[f32; 2]; 3],
    pool: &jxl_threadpool::JxlThreadPool,
) {
    if is_aarch64_feature_detected!("neon") {
        // SAFETY: Features are checked above.
        unsafe {
            for ((input, output), weights) in fb.iter().zip(fb_scratch).zip(weights) {
                run_gabor_rows_unsafe(
                    input,
                    output,
                    frame_header,
                    region,
                    weights,
                    pool,
                    run_gabor_row_aarch64_neon,
                );
            }
        }
        return;
    }

    for ((input, output), weights) in fb.iter().zip(fb_scratch).zip(weights) {
        run_gabor_rows(
            input,
            output,
            frame_header,
            region,
            weights,
            pool,
            run_gabor_row_generic,
        );
    }
}
