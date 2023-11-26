use std::arch::is_aarch64_feature_detected;

use jxl_grid::AllocTracker;

use super::generic;

pub fn adaptive_lf_smoothing_impl(
    width: usize,
    height: usize,
    lf_image: [&mut [f32]; 3],
    lf_scale: [f32; 3],
    tracker: Option<&AllocTracker>,
) -> crate::Result<()> {
    if is_aarch64_feature_detected!("neon") {
        // SAFETY: Feature set is checked above.
        return unsafe {
            adaptive_lf_smoothing_core_neon(width, height, lf_image, lf_scale, tracker)
        };
    }

    generic::adaptive_lf_smoothing_impl(width, height, lf_image, lf_scale, tracker)
}

#[target_feature(enable = "neon")]
unsafe fn adaptive_lf_smoothing_core_neon(
    width: usize,
    height: usize,
    lf_image: [&mut [f32]; 3],
    lf_scale: [f32; 3],
    tracker: Option<&AllocTracker>,
) -> crate::Result<()> {
    generic::adaptive_lf_smoothing_impl(width, height, lf_image, lf_scale, tracker)
}
