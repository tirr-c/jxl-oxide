use jxl_grid::SimpleGrid;
use jxl_threadpool::JxlThreadPool;

use super::generic::epf_common;

mod epf_avx2;
mod epf_sse2;

pub fn epf_step0(
    input: &[SimpleGrid<f32>; 3],
    output: &mut [SimpleGrid<f32>; 3],
    sigma_grid: &SimpleGrid<f32>,
    channel_scale: [f32; 3],
    border_sad_mul: f32,
    step_multiplier: f32,
    pool: &JxlThreadPool,
) {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        // SAFETY: Features are checked above.
        unsafe {
            return epf_common(
                input,
                output,
                sigma_grid,
                channel_scale,
                border_sad_mul,
                step_multiplier,
                pool,
                Some(epf_avx2::epf_row_step0_avx2),
                super::generic::epf_row_step0,
            );
        }
    }

    // SAFETY: x86_64 always supports SSE2.
    unsafe {
        epf_common(
            input,
            output,
            sigma_grid,
            channel_scale,
            border_sad_mul,
            step_multiplier,
            pool,
            Some(epf_sse2::epf_row_step0_sse2),
            super::generic::epf_row_step0,
        )
    }
}

pub fn epf_step1(
    input: &[SimpleGrid<f32>; 3],
    output: &mut [SimpleGrid<f32>; 3],
    sigma_grid: &SimpleGrid<f32>,
    channel_scale: [f32; 3],
    border_sad_mul: f32,
    step_multiplier: f32,
    pool: &JxlThreadPool,
) {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        // SAFETY: Features are checked above.
        unsafe {
            return epf_common(
                input,
                output,
                sigma_grid,
                channel_scale,
                border_sad_mul,
                step_multiplier,
                pool,
                Some(epf_avx2::epf_row_step1_avx2),
                super::generic::epf_row_step1,
            );
        }
    }

    // SAFETY: x86_64 always supports SSE2.
    unsafe {
        epf_common(
            input,
            output,
            sigma_grid,
            channel_scale,
            border_sad_mul,
            step_multiplier,
            pool,
            Some(epf_sse2::epf_row_step1_sse2),
            super::generic::epf_row_step1,
        )
    }
}

pub fn epf_step2(
    input: &[SimpleGrid<f32>; 3],
    output: &mut [SimpleGrid<f32>; 3],
    sigma_grid: &SimpleGrid<f32>,
    channel_scale: [f32; 3],
    border_sad_mul: f32,
    step_multiplier: f32,
    pool: &JxlThreadPool,
) {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        // SAFETY: Features are checked above.
        unsafe {
            return epf_common(
                input,
                output,
                sigma_grid,
                channel_scale,
                border_sad_mul,
                step_multiplier,
                pool,
                Some(epf_avx2::epf_row_step2_avx2),
                super::generic::epf_row_step2,
            );
        }
    }

    // SAFETY: x86_64 always supports SSE2.
    unsafe {
        epf_common(
            input,
            output,
            sigma_grid,
            channel_scale,
            border_sad_mul,
            step_multiplier,
            pool,
            Some(epf_sse2::epf_row_step2_sse2),
            super::generic::epf_row_step2,
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
