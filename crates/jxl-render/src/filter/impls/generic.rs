#![allow(dead_code)]

use jxl_frame::{filter::EpfParams, FrameHeader};
use jxl_grid::SimpleGrid;
use jxl_threadpool::JxlThreadPool;

use crate::{
    filter::{epf::run_epf_rows, gabor::run_gabor_rows},
    Region,
};

pub(crate) mod epf;
pub(crate) mod gabor;

pub fn epf<const STEP: usize>(
    input: &[SimpleGrid<f32>; 3],
    output: &mut [SimpleGrid<f32>; 3],
    frame_header: &FrameHeader,
    sigma_grid_map: &[Option<&SimpleGrid<f32>>],
    region: Region,
    epf_params: &EpfParams,
    pool: &JxlThreadPool,
) {
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
            epf::epf_row::<STEP>,
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
    for ((input, output), weights) in fb.iter().zip(fb_scratch).zip(weights) {
        run_gabor_rows(
            input,
            output,
            frame_header,
            region,
            weights,
            pool,
            gabor::run_gabor_row_generic,
        );
    }
}
