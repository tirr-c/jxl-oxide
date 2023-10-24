use std::arch::is_aarch64_feature_detected;

use super::generic;

pub fn adaptive_lf_smoothing_impl(
    width: usize,
    height: usize,
    lf_image: [&mut [f32]; 3],
    lf_scale: [f32; 3],
) {
    if is_aarch64_feature_detected!("neon") {
        // SAFETY: Feature set is checked above.
        return unsafe {
            adaptive_lf_smoothing_core_neon(width, height, lf_image, lf_scale)
        };
    }

    generic::adaptive_lf_smoothing_impl(width, height, lf_image, lf_scale)
}

#[target_feature(enable = "neon")]
unsafe fn adaptive_lf_smoothing_core_neon(
    width: usize,
    height: usize,
    lf_image: [&mut [f32]; 3],
    lf_scale: [f32; 3],
) {
    generic::adaptive_lf_smoothing_impl(width, height, lf_image, lf_scale)
}
