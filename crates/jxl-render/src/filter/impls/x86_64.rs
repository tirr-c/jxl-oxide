use jxl_frame::{filter::EpfParams, FrameHeader};
use jxl_grid::AlignedGrid;
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
    input: [&AlignedGrid<f32>; 3],
    output: &mut [AlignedGrid<f32>; 3],
    frame_header: &FrameHeader,
    sigma_grid_map: &[Option<&AlignedGrid<f32>>],
    region: Region,
    epf_params: &EpfParams,
    pool: &JxlThreadPool,
) {
    if is_x86_feature_detected!("sse4.1") {
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
                Some(epf_sse41::epf_row_x86_64_sse41::<STEP>),
                super::generic::epf::epf_row::<STEP>,
            );
        }
    }

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
    fb: [&AlignedGrid<f32>; 3],
    fb_scratch: &mut [AlignedGrid<f32>; 3],
    frame_header: &FrameHeader,
    region: Region,
    weights: [[f32; 2]; 3],
    pool: &jxl_threadpool::JxlThreadPool,
) {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
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
                    gabor_avx2::run_gabor_row_x86_64_avx2,
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
