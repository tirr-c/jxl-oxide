//! Map out-of-gamut RGB samples, ported from libjxl.

#[inline]
pub(crate) fn map_gamut_generic(
    rgb: [f32; 3],
    luminance: [f32; 3],
    saturation_factor: f32,
) -> [f32; 3] {
    let [r, g, b] = rgb;
    let [lr, lg, lb] = luminance;
    let y = r * lr + g * lg + b * lb;

    let (gray_saturation, gray_luminance) =
        rgb.into_iter()
            .fold((0f32, 0f32), |(gray_saturation, gray_luminance), v| {
                let v_sub_y = v - y;
                let inv_v_sub_y = (if v_sub_y == 0.0 { 1.0 } else { v_sub_y }).recip();
                let v_over_v_sub_y = v * inv_v_sub_y;

                let gray_saturation = if v_sub_y >= 0.0 {
                    gray_saturation
                } else {
                    gray_saturation.max(v_over_v_sub_y)
                };
                let gray_luminance = if v_sub_y <= 0.0 {
                    gray_saturation
                } else {
                    v_over_v_sub_y - inv_v_sub_y
                }
                .max(gray_luminance);

                (gray_saturation, gray_luminance)
            });

    let gray_mix =
        (saturation_factor * (gray_saturation - gray_luminance) + gray_luminance).clamp(0.0, 1.0);

    let mixed_rgb = rgb.map(|v| gray_mix * (y - v) + v);
    let max_color_val = rgb.into_iter().fold(1f32, |max, v| v.max(max));
    mixed_rgb.map(|v| v / max_color_val)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub(crate) unsafe fn map_gamut_aarch64_neon(
    rgb: [std::arch::aarch64::float32x4_t; 3],
    luminance: [f32; 3],
    saturation_factor: f32,
) -> [std::arch::aarch64::float32x4_t; 3] {
    use jxl_grid::SimdVector;
    use std::arch::aarch64::*;

    let [vr, vg, vb] = rgb;
    let [lr, lg, lb] = luminance;
    let y = vfmaq_n_f32(vfmaq_n_f32(vmulq_n_f32(vr, lr), vg, lg), vb, lb);

    let (gray_saturation, gray_luminance) = rgb.into_iter().fold(
        (vdupq_n_f32(0.0), vdupq_n_f32(0.0)),
        |(gray_saturation, gray_luminance), v| {
            let v_sub_y = v.sub(y);
            let inv_v_sub_y = vdivq_f32(
                vdupq_n_f32(1.0),
                vbslq_f32(vceqzq_f32(v_sub_y), vdupq_n_f32(1.0), v_sub_y),
            );
            let v_over_v_sub_y = v.mul(inv_v_sub_y);

            let gray_saturation = vbslq_f32(
                vcgezq_f32(v_sub_y),
                gray_saturation,
                vmaxq_f32(gray_saturation, v_over_v_sub_y),
            );
            let gray_luminance = vmaxq_f32(
                vbslq_f32(
                    vclezq_f32(v_sub_y),
                    gray_saturation,
                    v_over_v_sub_y.sub(inv_v_sub_y),
                ),
                gray_luminance,
            );

            (gray_saturation, gray_luminance)
        },
    );

    let gray_mix = vfmaq_n_f32(
        gray_luminance,
        gray_saturation.sub(gray_luminance),
        saturation_factor,
    );
    let gray_mix = vmaxq_f32(vdupq_n_f32(0.0), vminq_f32(vdupq_n_f32(1.0), gray_mix));

    let mixed_rgb = rgb.map(|v| gray_mix.muladd(y.sub(v), v));
    let max_color_val = rgb
        .into_iter()
        .fold(vdupq_n_f32(1.0), |acc, v| vmaxq_f32(acc, v));
    mixed_rgb.map(|v| v.div(max_color_val))
}
