use std::arch::is_aarch64_feature_detected;

use jxl_grid::AllocTracker;
use jxl_modular::ChannelShift;

use super::generic;

mod dct;
mod transform;
pub use transform::transform_varblocks;

pub fn adaptive_lf_smoothing_impl(
    width: usize,
    height: usize,
    shifts: [ChannelShift; 3],
    lf_image: [&mut [f32]; 3],
    lf_scale: [f32; 3],
    tracker: Option<&AllocTracker>,
) -> crate::Result<()> {
    if is_aarch64_feature_detected!("neon") {
        // SAFETY: Feature set is checked above.
        return unsafe {
            adaptive_lf_smoothing_core_neon(width, height, shifts, lf_image, lf_scale, tracker)
        };
    }

    generic::adaptive_lf_smoothing_impl(width, height, shifts, lf_image, lf_scale, tracker)
}

#[target_feature(enable = "neon")]
unsafe fn adaptive_lf_smoothing_core_neon(
    width: usize,
    height: usize,
    shifts: [ChannelShift; 3],
    lf_image: [&mut [f32]; 3],
    lf_scale: [f32; 3],
    tracker: Option<&AllocTracker>,
) -> crate::Result<()> {
    generic::adaptive_lf_smoothing_impl(width, height, shifts, lf_image, lf_scale, tracker)
}
