use jxl_grid::SimpleGrid;

mod epf_sse2;
mod epf_avx2;

pub fn epf_step0(
    input: &[SimpleGrid<f32>; 3],
    output: &mut [SimpleGrid<f32>; 3],
    sigma_grid: &SimpleGrid<f32>,
    channel_scale: [f32; 3],
    border_sad_mul: f32,
    step_multiplier: f32,
) {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        // SAFETY: Features are checked above.
        unsafe {
            return epf_avx2::epf_step0_avx2(
                input,
                output,
                sigma_grid,
                channel_scale,
                border_sad_mul,
                step_multiplier,
            );
        }
    }

    // SAFETY: x86_64 always supports SSE2.
    unsafe {
        epf_sse2::epf_step0_sse2(
            input,
            output,
            sigma_grid,
            channel_scale,
            border_sad_mul,
            step_multiplier,
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
) {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        // SAFETY: Features are checked above.
        unsafe {
            return epf_avx2::epf_step1_avx2(
                input,
                output,
                sigma_grid,
                channel_scale,
                border_sad_mul,
                step_multiplier,
            );
        }
    }

    // SAFETY: x86_64 always supports SSE2.
    unsafe {
        epf_sse2::epf_step1_sse2(
            input,
            output,
            sigma_grid,
            channel_scale,
            border_sad_mul,
            step_multiplier,
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
) {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        // SAFETY: Features are checked above.
        unsafe {
            return epf_avx2::epf_step2_avx2(
                input,
                output,
                sigma_grid,
                channel_scale,
                border_sad_mul,
                step_multiplier,
            );
        }
    }

    // SAFETY: x86_64 always supports SSE2.
    unsafe {
        epf_sse2::epf_step2_sse2(
            input,
            output,
            sigma_grid,
            channel_scale,
            border_sad_mul,
            step_multiplier,
        )
    }
}

pub fn apply_gabor_like(fb: [&mut SimpleGrid<f32>; 3], weights_xyb: [[f32; 2]; 3]) {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        // SAFETY: Features are checked above.
        unsafe {
            for (fb, [weight1, weight2]) in fb.into_iter().zip(weights_xyb) {
                run_gabor_inner_avx2(fb, weight1, weight2)
            }
        }
        return;
    }

    for (fb, [weight1, weight2]) in fb.into_iter().zip(weights_xyb) {
        super::run_gabor_inner(fb, weight1, weight2);
    }
}

#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
unsafe fn run_gabor_inner_avx2(fb: &mut SimpleGrid<f32>, weight1: f32, weight2: f32) {
    super::run_gabor_inner(fb, weight1, weight2)
}
