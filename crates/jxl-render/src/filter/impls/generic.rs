#![allow(dead_code)]

use jxl_frame::{filter::EpfParams, FrameHeader};
use jxl_grid::{AlignedGrid, MutableSubgrid};
use jxl_threadpool::JxlThreadPool;

use crate::{
    filter::{epf::run_epf_rows, gabor::run_gabor_rows},
    Region,
};

pub(crate) mod epf;
pub(crate) mod gabor;

pub fn epf<const STEP: usize>(
    input: &mut [MutableSubgrid<f32>; 3],
    output: &mut [MutableSubgrid<f32>; 3],
    color_padded_region: Region,
    frame_header: &FrameHeader,
    sigma_grid_map: &[Option<&AlignedGrid<f32>>],
    epf_params: &EpfParams,
    pool: &JxlThreadPool,
) {
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
            epf::epf_row::<STEP>,
        )
    }
}

pub fn apply_gabor_like(
    fb: [MutableSubgrid<f32>; 3],
    fb_scratch: &mut [AlignedGrid<f32>; 3],
    weights: [[f32; 2]; 3],
    pool: &jxl_threadpool::JxlThreadPool,
) {
    for ((input, output), weights) in fb.into_iter().zip(fb_scratch).zip(weights) {
        run_gabor_rows(input, output, weights, pool, gabor::run_gabor_row_generic);
    }
}
