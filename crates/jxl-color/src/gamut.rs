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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
#[target_feature(enable = "sse4.1")]
#[inline]
pub(crate) unsafe fn map_gamut_x86_64_avx2(
    rgb: [std::arch::x86_64::__m256; 3],
    luminance: [f32; 3],
    saturation_factor: f32,
) -> [std::arch::x86_64::__m256; 3] {
    use jxl_grid::SimdVector;
    use std::arch::x86_64::*;

    let [vr, vg, vb] = rgb;
    let [vlr, vlg, vlb] = luminance.map(|v| _mm256_set1_ps(v));
    let y = _mm256_fmadd_ps(vb, vlb, _mm256_fmadd_ps(vg, vlg, vr.mul(vlr)));

    let (gray_saturation, gray_luminance) = rgb.into_iter().fold(
        (_mm256_setzero_ps(), _mm256_setzero_ps()),
        |(gray_saturation, gray_luminance), v| {
            let v_sub_y = v.sub(y);
            let inv_v_sub_y = _mm256_set1_ps(1.0).div(_mm256_blendv_ps(
                v_sub_y,
                _mm256_set1_ps(1.0),
                _mm256_cmp_ps::<_CMP_EQ_OQ>(v_sub_y, _mm256_setzero_ps()),
            ));
            let v_over_v_sub_y = v.mul(inv_v_sub_y);

            let gray_saturation = _mm256_blendv_ps(
                _mm256_max_ps(gray_saturation, v_over_v_sub_y),
                gray_saturation,
                _mm256_cmp_ps::<_CMP_GE_OQ>(v_sub_y, _mm256_setzero_ps()),
            );
            let gray_luminance = _mm256_max_ps(
                _mm256_blendv_ps(
                    v_over_v_sub_y.sub(inv_v_sub_y),
                    gray_saturation,
                    _mm256_cmp_ps::<_CMP_LE_OQ>(v_sub_y, _mm256_setzero_ps()),
                ),
                gray_luminance,
            );

            (gray_saturation, gray_luminance)
        },
    );

    let gray_mix = gray_saturation
        .sub(gray_luminance)
        .muladd(_mm256_set1_ps(saturation_factor), gray_luminance);
    let gray_mix = _mm256_max_ps(
        _mm256_setzero_ps(),
        _mm256_min_ps(_mm256_set1_ps(1.0), gray_mix),
    );

    let mixed_rgb = rgb.map(|v| gray_mix.muladd(y.sub(v), v));
    let max_color_val = rgb
        .into_iter()
        .fold(_mm256_set1_ps(1.0), |acc, v| _mm256_max_ps(acc, v));
    mixed_rgb.map(|v| v.div(max_color_val))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
#[target_feature(enable = "sse4.1")]
#[inline]
pub(crate) unsafe fn map_gamut_x86_64_fma(
    rgb: [std::arch::x86_64::__m128; 3],
    luminance: [f32; 3],
    saturation_factor: f32,
) -> [std::arch::x86_64::__m128; 3] {
    use jxl_grid::SimdVector;
    use std::arch::x86_64::*;

    let [vr, vg, vb] = rgb;
    let [vlr, vlg, vlb] = luminance.map(|v| _mm_set1_ps(v));
    let y = _mm_fmadd_ps(vb, vlb, _mm_fmadd_ps(vg, vlg, vr.mul(vlr)));

    let (gray_saturation, gray_luminance) = rgb.into_iter().fold(
        (_mm_setzero_ps(), _mm_setzero_ps()),
        |(gray_saturation, gray_luminance), v| {
            let v_sub_y = v.sub(y);
            let inv_v_sub_y = _mm_set1_ps(1.0).div(_mm_blendv_ps(
                v_sub_y,
                _mm_set1_ps(1.0),
                _mm_cmpeq_ps(v_sub_y, _mm_setzero_ps()),
            ));
            let v_over_v_sub_y = v.mul(inv_v_sub_y);

            let gray_saturation = _mm_blendv_ps(
                _mm_max_ps(gray_saturation, v_over_v_sub_y),
                gray_saturation,
                _mm_cmpge_ps(v_sub_y, _mm_setzero_ps()),
            );
            let gray_luminance = _mm_max_ps(
                _mm_blendv_ps(
                    v_over_v_sub_y.sub(inv_v_sub_y),
                    gray_saturation,
                    _mm_cmple_ps(v_sub_y, _mm_setzero_ps()),
                ),
                gray_luminance,
            );

            (gray_saturation, gray_luminance)
        },
    );

    let gray_mix = gray_saturation
        .sub(gray_luminance)
        .muladd(_mm_set1_ps(saturation_factor), gray_luminance);
    let gray_mix = _mm_max_ps(_mm_setzero_ps(), _mm_min_ps(_mm_set1_ps(1.0), gray_mix));

    let mixed_rgb = rgb.map(|v| gray_mix.muladd(y.sub(v), v));
    let max_color_val = rgb
        .into_iter()
        .fold(_mm_set1_ps(1.0), |acc, v| _mm_max_ps(acc, v));
    mixed_rgb.map(|v| v.div(max_color_val))
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) unsafe fn map_gamut_x86_64_sse2(
    rgb: [std::arch::x86_64::__m128; 3],
    luminance: [f32; 3],
    saturation_factor: f32,
) -> [std::arch::x86_64::__m128; 3] {
    use jxl_grid::SimdVector;
    use std::arch::x86_64::*;

    let [vr, vg, vb] = rgb;
    let [vlr, vlg, vlb] = luminance.map(|v| _mm_set1_ps(v));
    let y = vb.muladd(vlb, vg.muladd(vlg, vr.mul(vlr)));

    let (gray_saturation, gray_luminance) = rgb.into_iter().fold(
        (_mm_setzero_ps(), _mm_setzero_ps()),
        |(gray_saturation, gray_luminance), v| {
            let v_sub_y = v.sub(y);
            let mask = _mm_cmpeq_ps(v_sub_y, _mm_setzero_ps());
            let inv_v_sub_y = _mm_set1_ps(1.0).div(_mm_or_ps(
                _mm_andnot_ps(mask, v_sub_y),
                _mm_and_ps(mask, _mm_set1_ps(1.0)),
            ));
            let v_over_v_sub_y = v.mul(inv_v_sub_y);

            let mask = _mm_cmpge_ps(v_sub_y, _mm_setzero_ps());
            let gray_saturation = _mm_or_ps(
                _mm_andnot_ps(mask, _mm_max_ps(gray_saturation, v_over_v_sub_y)),
                _mm_and_ps(mask, gray_saturation),
            );

            let mask = _mm_cmple_ps(v_sub_y, _mm_setzero_ps());
            let gray_luminance = _mm_max_ps(
                _mm_or_ps(
                    _mm_andnot_ps(mask, v_over_v_sub_y.sub(inv_v_sub_y)),
                    _mm_and_ps(mask, gray_saturation),
                ),
                gray_luminance,
            );

            (gray_saturation, gray_luminance)
        },
    );

    let gray_mix = gray_saturation
        .sub(gray_luminance)
        .muladd(_mm_set1_ps(saturation_factor), gray_luminance);
    let gray_mix = _mm_max_ps(_mm_setzero_ps(), _mm_min_ps(_mm_set1_ps(1.0), gray_mix));

    let mixed_rgb = rgb.map(|v| gray_mix.muladd(y.sub(v), v));
    let max_color_val = rgb
        .into_iter()
        .fold(_mm_set1_ps(1.0), |acc, v| _mm_max_ps(acc, v));
    mixed_rgb.map(|v| v.div(max_color_val))
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
