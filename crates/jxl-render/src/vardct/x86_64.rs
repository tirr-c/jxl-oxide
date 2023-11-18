use super::generic;

pub fn adaptive_lf_smoothing_impl(
    width: usize,
    height: usize,
    lf_image: [&mut [f32]; 3],
    lf_scale: [f32; 3],
) {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        // SAFETY: Feature set is checked above.
        return unsafe { adaptive_lf_smoothing_core_avx2(width, height, lf_image, lf_scale) };
    }

    generic::adaptive_lf_smoothing_impl(width, height, lf_image, lf_scale)
}

#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
unsafe fn adaptive_lf_smoothing_core_avx2(
    width: usize,
    height: usize,
    lf_image: [&mut [f32]; 3],
    lf_scale: [f32; 3],
) {
    generic::adaptive_lf_smoothing_impl(width, height, lf_image, lf_scale)
}
