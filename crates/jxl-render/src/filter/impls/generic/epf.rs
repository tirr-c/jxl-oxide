#[inline]
pub fn weight(scaled_distance: f32, sigma: f32, step_multiplier: f32) -> f32 {
    let inv_sigma = step_multiplier * 6.6 * (1.0 - std::f32::consts::FRAC_1_SQRT_2) / sigma;
    (1.0 - scaled_distance * inv_sigma).max(0.0)
}
