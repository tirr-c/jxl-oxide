use std::arch::is_aarch64_feature_detected;

use jxl_grid::SimpleGrid;

mod epf;

pub fn epf_step0(
    input: &[SimpleGrid<f32>; 3],
    output: &mut [SimpleGrid<f32>; 3],
    sigma_grid: &SimpleGrid<f32>,
    channel_scale: [f32; 3],
    border_sad_mul: f32,
    step_multiplier: f32,
) {
    if is_aarch64_feature_detected!("neon") {
        // SAFETY: Features are checked above.
        unsafe {
            return epf::epf_step0_neon(
                input,
                output,
                sigma_grid,
                channel_scale,
                border_sad_mul,
                step_multiplier,
            );
        }
    }

    super::generic::epf_step0(
        input,
        output,
        sigma_grid,
        channel_scale,
        border_sad_mul,
        step_multiplier,
    )
}

pub fn epf_step1(
    input: &[SimpleGrid<f32>; 3],
    output: &mut [SimpleGrid<f32>; 3],
    sigma_grid: &SimpleGrid<f32>,
    channel_scale: [f32; 3],
    border_sad_mul: f32,
    step_multiplier: f32,
) {
    if is_aarch64_feature_detected!("neon") {
        // SAFETY: Features are checked above.
        unsafe {
            return epf::epf_step1_neon(
                input,
                output,
                sigma_grid,
                channel_scale,
                border_sad_mul,
                step_multiplier,
            );
        }
    }

    super::generic::epf_step1(
        input,
        output,
        sigma_grid,
        channel_scale,
        border_sad_mul,
        step_multiplier,
    )
}

pub fn epf_step2(
    input: &[SimpleGrid<f32>; 3],
    output: &mut [SimpleGrid<f32>; 3],
    sigma_grid: &SimpleGrid<f32>,
    channel_scale: [f32; 3],
    border_sad_mul: f32,
    step_multiplier: f32,
) {
    if is_aarch64_feature_detected!("neon") {
        // SAFETY: Features are checked above.
        unsafe {
            return epf::epf_step2_neon(
                input,
                output,
                sigma_grid,
                channel_scale,
                border_sad_mul,
                step_multiplier,
            );
        }
    }

    super::generic::epf_step2(
        input,
        output,
        sigma_grid,
        channel_scale,
        border_sad_mul,
        step_multiplier,
    )
}

pub fn apply_gabor_like(fb: [&mut SimpleGrid<f32>; 3], weights_xyb: [[f32; 2]; 3]) {
    if is_aarch64_feature_detected!("neon") {
        // SAFETY: Features are checked above.
        unsafe {
            for (fb, [weight1, weight2]) in fb.into_iter().zip(weights_xyb) {
                run_gabor_inner_neon(fb, weight1, weight2)
            }
        }
        return;
    }

    for (fb, [weight1, weight2]) in fb.into_iter().zip(weights_xyb) {
        super::generic::run_gabor_inner(fb, weight1, weight2);
    }
}

#[target_feature(enable = "neon")]
unsafe fn run_gabor_inner_neon(fb: &mut SimpleGrid<f32>, weight1: f32, weight2: f32) {
    super::generic::run_gabor_inner(fb, weight1, weight2)
}
