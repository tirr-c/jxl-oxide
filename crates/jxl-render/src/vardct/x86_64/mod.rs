use jxl_grid::AllocTracker;

use super::generic;

mod dct;
mod transform;
pub use transform::transform_varblocks;

pub fn adaptive_lf_smoothing_impl(
    width: usize,
    height: usize,
    lf_image: [&mut [f32]; 3],
    lf_scale: [f32; 3],
    tracker: Option<&AllocTracker>,
) -> crate::Result<()> {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        // SAFETY: Feature set is checked above.
        return unsafe {
            adaptive_lf_smoothing_core_avx2(width, height, lf_image, lf_scale, tracker)
        };
    }

    generic::adaptive_lf_smoothing_impl(width, height, lf_image, lf_scale, tracker)
}

#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
unsafe fn adaptive_lf_smoothing_core_avx2(
    width: usize,
    height: usize,
    lf_image: [&mut [f32]; 3],
    lf_scale: [f32; 3],
    tracker: Option<&AllocTracker>,
) -> crate::Result<()> {
    generic::adaptive_lf_smoothing_impl(width, height, lf_image, lf_scale, tracker)
}
