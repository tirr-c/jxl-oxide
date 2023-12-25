/// Map out-of-gamut RGB samples, ported from libjxl.
pub(crate) fn map_gamut_generic(
    rgb: [f32; 3],
    luminance: [f32; 3],
    saturation_factor: f32,
) -> [f32; 3] {
    let [r, g, b] = rgb;
    let [lr, lg, lb] = luminance;
    let y = r * lr + g * lg + b * lb;

    let (gray_saturation, gray_luminance) = rgb
        .into_iter()
        .fold(
            (0f32, 0f32),
            |(gray_saturation, gray_luminance), v| {
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
                }.max(gray_luminance);

                (gray_saturation, gray_luminance)
            },
        );

    let gray_mix = (saturation_factor * (gray_saturation - gray_luminance) + gray_luminance)
        .clamp(0.0, 1.0);

    let mixed_rgb = rgb.map(|v| gray_mix * (y - v) + v);
    let max_color_val = rgb.into_iter().fold(1f32, |max, v| v.max(max));
    mixed_rgb.map(|v| v / max_color_val)
}
