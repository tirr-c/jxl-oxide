use jxl_frame::{filter::EpfParams, FrameHeader};
use jxl_grid::SimpleGrid;
use jxl_threadpool::JxlThreadPool;

use crate::Region;

use super::generic::epf_common;

mod epf_sse41;

pub fn epf_step0(
    input: &[SimpleGrid<f32>; 3],
    output: &mut [SimpleGrid<f32>; 3],
    frame_header: &FrameHeader,
    sigma_grid_map: &[Option<&SimpleGrid<f32>>],
    region: Region,
    epf_params: &EpfParams,
    pool: &JxlThreadPool,
) {
    if is_x86_feature_detected!("sse4.1") {
        // SAFETY: Features are checked above.
        unsafe {
            return epf_common(
                input,
                output,
                frame_header,
                sigma_grid_map,
                region,
                epf_params,
                pool,
                Some(epf_sse41::epf_row_x86_64_sse41::<0>),
                super::generic::epf_row::<0>,
            );
        }
    }

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
            super::generic::epf_row::<0>,
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
    if is_x86_feature_detected!("sse4.1") {
        // SAFETY: Features are checked above.
        unsafe {
            return epf_common(
                input,
                output,
                frame_header,
                sigma_grid_map,
                region,
                epf_params,
                pool,
                Some(epf_sse41::epf_row_x86_64_sse41::<1>),
                super::generic::epf_row::<1>,
            );
        }
    }

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
            super::generic::epf_row::<1>,
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
    if is_x86_feature_detected!("sse4.1") {
        // SAFETY: Features are checked above.
        unsafe {
            return epf_common(
                input,
                output,
                frame_header,
                sigma_grid_map,
                region,
                epf_params,
                pool,
                Some(epf_sse41::epf_row_x86_64_sse41::<2>),
                super::generic::epf_row::<2>,
            );
        }
    }

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
            super::generic::epf_row::<2>,
        )
    }
}

pub fn apply_gabor_like(
    fb: [&mut SimpleGrid<f32>; 3],
    weights_xyb: [[f32; 2]; 3],
) -> crate::Result<()> {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        // SAFETY: Features are checked above.
        unsafe {
            for (fb, [weight1, weight2]) in fb.into_iter().zip(weights_xyb) {
                run_gabor_inner_avx2(fb, weight1, weight2)?
            }
        }
        return Ok(());
    }

    for (fb, [weight1, weight2]) in fb.into_iter().zip(weights_xyb) {
        super::generic::run_gabor_inner(fb, weight1, weight2)?;
    }
    Ok(())
}

#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
unsafe fn run_gabor_inner_avx2(
    fb: &mut SimpleGrid<f32>,
    weight1: f32,
    weight2: f32,
) -> crate::Result<()> {
    super::generic::run_gabor_inner(fb, weight1, weight2)
}
