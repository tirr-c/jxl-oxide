#[cfg(target_arch = "aarch64")]
use std::arch::is_aarch64_feature_detected;
#[cfg(target_arch = "x86_64")]
use std::arch::is_x86_feature_detected;

use super::HdrParams;

#[allow(unreachable_code)]
pub(super) fn tone_map(
    r: &mut [f32],
    g: &mut [f32],
    b: &mut [f32],
    hdr_params: &HdrParams,
    target_display_luminance: f32,
    detect_peak: bool,
) {
    assert_eq!(r.len(), g.len());
    assert_eq!(g.len(), b.len());

    let luminances = hdr_params.luminances;
    let intensity_target = hdr_params.intensity_target;
    let min_nits = hdr_params.min_nits;
    let detected_peak_luminance = if detect_peak {
        detect_peak_luminance(r, g, b, luminances) * intensity_target
    } else {
        intensity_target
    };
    let peak_luminance = intensity_target.min(detected_peak_luminance);

    let from_luminance_range = (min_nits, peak_luminance);
    let to_luminance_range = (0.0, target_display_luminance);

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("fma") && is_x86_feature_detected!("sse4.1") {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                return tone_map_x86_64_avx2(
                    r,
                    g,
                    b,
                    luminances,
                    intensity_target,
                    from_luminance_range,
                    to_luminance_range,
                );
            }
        } else {
            unsafe {
                return tone_map_x86_64_fma(
                    r,
                    g,
                    b,
                    luminances,
                    intensity_target,
                    from_luminance_range,
                    to_luminance_range,
                );
            }
        }
    } else {
        unsafe {
            return tone_map_x86_64_sse2(
                r,
                g,
                b,
                luminances,
                intensity_target,
                from_luminance_range,
                to_luminance_range,
            );
        }
    }

    #[cfg(target_arch = "aarch64")]
    if is_aarch64_feature_detected!("neon") {
        // SAFETY: features are checked above.
        unsafe {
            return tone_map_aarch64_neon(
                r,
                g,
                b,
                luminances,
                intensity_target,
                from_luminance_range,
                to_luminance_range,
            );
        }
    }

    tone_map_generic(
        r,
        g,
        b,
        luminances,
        intensity_target,
        from_luminance_range,
        to_luminance_range,
    );
}

#[allow(unreachable_code)]
pub(super) fn tone_map_luma(
    luma: &mut [f32],
    hdr_params: &HdrParams,
    target_display_luminance: f32,
    detect_peak: bool,
) {
    let intensity_target = hdr_params.intensity_target;
    let min_nits = hdr_params.min_nits;
    let detected_peak_luminance = if detect_peak {
        let max_luma = luma.iter().copied().fold(0f32, |max, v| max.max(v));
        if max_luma == 0.0 {
            intensity_target
        } else {
            max_luma * intensity_target
        }
    } else {
        intensity_target
    };
    let peak_luminance = intensity_target.min(detected_peak_luminance);

    let from_luminance_range = (min_nits, peak_luminance);
    let to_luminance_range = (0.0, target_display_luminance);

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("fma") && is_x86_feature_detected!("sse4.1") {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                return tone_map_luma_x86_64_avx2(
                    luma,
                    intensity_target,
                    from_luminance_range,
                    to_luminance_range,
                );
            }
        } else {
            unsafe {
                return tone_map_luma_x86_64_fma(
                    luma,
                    intensity_target,
                    from_luminance_range,
                    to_luminance_range,
                );
            }
        }
    } else {
        unsafe {
            return tone_map_luma_x86_64_sse2(
                luma,
                intensity_target,
                from_luminance_range,
                to_luminance_range,
            );
        }
    }

    #[cfg(target_arch = "aarch64")]
    if is_aarch64_feature_detected!("neon") {
        // SAFETY: features are checked above.
        unsafe {
            return tone_map_luma_aarch64_neon(
                luma,
                intensity_target,
                from_luminance_range,
                to_luminance_range,
            );
        }
    }

    tone_map_luma_generic(
        luma,
        intensity_target,
        from_luminance_range,
        to_luminance_range,
    );
}

fn tone_map_generic(
    r: &mut [f32],
    g: &mut [f32],
    b: &mut [f32],
    luminances: [f32; 3],
    intensity_target: f32,
    from_luminance_range: (f32, f32),
    to_luminance_range: (f32, f32),
) {
    assert_eq!(r.len(), g.len());
    assert_eq!(g.len(), b.len());
    let scale = intensity_target / to_luminance_range.1;

    let [lr, lg, lb] = luminances;
    for ((r, g), b) in r.iter_mut().zip(g).zip(b) {
        let y = *r * lr + *g * lg + *b * lb;
        let y_pq = crate::tf::pq::linear_to_pq_generic(y, intensity_target);
        let y_mapped = crate::tf::rec2408::rec2408_eetf_generic(
            y_pq,
            intensity_target,
            from_luminance_range,
            to_luminance_range,
        );
        let y_mapped = crate::tf::pq::pq_to_linear_generic(y_mapped, intensity_target);
        let ratio = if y <= 1e-7 {
            y_mapped * scale
        } else {
            y_mapped / y * scale
        };
        *r *= ratio;
        *g *= ratio;
        *b *= ratio;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "sse4.1")]
#[target_feature(enable = "fma")]
unsafe fn tone_map_x86_64_avx2(
    r: &mut [f32],
    g: &mut [f32],
    b: &mut [f32],
    luminances: [f32; 3],
    intensity_target: f32,
    from_luminance_range: (f32, f32),
    to_luminance_range: (f32, f32),
) {
    use std::arch::x86_64::*;

    assert_eq!(r.len(), g.len());
    assert_eq!(g.len(), b.len());
    let scale = intensity_target / to_luminance_range.1;
    let vscale = _mm256_set1_ps(scale);

    let [lr, lg, lb] = luminances;
    let vlr = _mm256_set1_ps(lr);
    let vlg = _mm256_set1_ps(lg);
    let vlb = _mm256_set1_ps(lb);
    let mut r = r.chunks_exact_mut(8);
    let mut g = g.chunks_exact_mut(8);
    let mut b = b.chunks_exact_mut(8);
    for ((r, g), b) in (&mut r).zip(&mut g).zip(&mut b) {
        let vr = _mm256_loadu_ps(r.as_ptr());
        let vg = _mm256_loadu_ps(g.as_ptr());
        let vb = _mm256_loadu_ps(b.as_ptr());
        let vy = _mm256_fmadd_ps(vb, vlb, _mm256_fmadd_ps(vg, vlg, _mm256_mul_ps(vr, vlr)));
        let vy_pq = crate::tf::pq::linear_to_pq_x86_64_avx2(vy, intensity_target);
        let vy_mapped = crate::tf::rec2408::rec2408_eetf_x86_64_avx2(
            vy_pq,
            intensity_target,
            from_luminance_range,
            to_luminance_range,
        );
        let vy_mapped = crate::tf::pq::pq_to_linear_x86_64_avx2(vy_mapped, intensity_target);
        let is_small = _mm256_cmp_ps::<_CMP_LT_OQ>(vy, _mm256_set1_ps(1e-7));
        let vy = _mm256_blendv_ps(vy, _mm256_set1_ps(1.0), is_small);
        let ratio = _mm256_div_ps(_mm256_mul_ps(vy_mapped, vscale), vy);
        _mm256_storeu_ps(r.as_mut_ptr(), _mm256_mul_ps(vr, ratio));
        _mm256_storeu_ps(g.as_mut_ptr(), _mm256_mul_ps(vg, ratio));
        _mm256_storeu_ps(b.as_mut_ptr(), _mm256_mul_ps(vb, ratio));
    }

    tone_map_x86_64_fma(
        r.into_remainder(),
        g.into_remainder(),
        b.into_remainder(),
        luminances,
        intensity_target,
        from_luminance_range,
        to_luminance_range,
    );
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[target_feature(enable = "fma")]
unsafe fn tone_map_x86_64_fma(
    r: &mut [f32],
    g: &mut [f32],
    b: &mut [f32],
    luminances: [f32; 3],
    intensity_target: f32,
    from_luminance_range: (f32, f32),
    to_luminance_range: (f32, f32),
) {
    use std::arch::x86_64::*;

    assert_eq!(r.len(), g.len());
    assert_eq!(g.len(), b.len());
    let scale = intensity_target / to_luminance_range.1;
    let vscale = _mm_set1_ps(scale);

    let [lr, lg, lb] = luminances;
    let vlr = _mm_set1_ps(lr);
    let vlg = _mm_set1_ps(lg);
    let vlb = _mm_set1_ps(lb);
    let mut r = r.chunks_exact_mut(4);
    let mut g = g.chunks_exact_mut(4);
    let mut b = b.chunks_exact_mut(4);
    for ((r, g), b) in (&mut r).zip(&mut g).zip(&mut b) {
        let vr = _mm_loadu_ps(r.as_ptr());
        let vg = _mm_loadu_ps(g.as_ptr());
        let vb = _mm_loadu_ps(b.as_ptr());
        let vy = _mm_fmadd_ps(vb, vlb, _mm_fmadd_ps(vg, vlg, _mm_mul_ps(vr, vlr)));
        let vy_pq = crate::tf::pq::linear_to_pq_x86_64_fma(vy, intensity_target);
        let vy_mapped = crate::tf::rec2408::rec2408_eetf_x86_64_fma(
            vy_pq,
            intensity_target,
            from_luminance_range,
            to_luminance_range,
        );
        let vy_mapped = crate::tf::pq::pq_to_linear_x86_64_fma(vy_mapped, intensity_target);
        let is_small = _mm_cmplt_ps(vy, _mm_set1_ps(1e-7));
        let vy = _mm_blendv_ps(vy, _mm_set1_ps(1.0), is_small);
        let ratio = _mm_div_ps(_mm_mul_ps(vy_mapped, vscale), vy);
        _mm_storeu_ps(r.as_mut_ptr(), _mm_mul_ps(vr, ratio));
        _mm_storeu_ps(g.as_mut_ptr(), _mm_mul_ps(vg, ratio));
        _mm_storeu_ps(b.as_mut_ptr(), _mm_mul_ps(vb, ratio));
    }

    tone_map_generic(
        r.into_remainder(),
        g.into_remainder(),
        b.into_remainder(),
        luminances,
        intensity_target,
        from_luminance_range,
        to_luminance_range,
    );
}

#[cfg(target_arch = "x86_64")]
unsafe fn tone_map_x86_64_sse2(
    r: &mut [f32],
    g: &mut [f32],
    b: &mut [f32],
    luminances: [f32; 3],
    intensity_target: f32,
    from_luminance_range: (f32, f32),
    to_luminance_range: (f32, f32),
) {
    use std::arch::x86_64::*;

    assert_eq!(r.len(), g.len());
    assert_eq!(g.len(), b.len());
    let scale = intensity_target / to_luminance_range.1;
    let vscale = _mm_set1_ps(scale);

    let [lr, lg, lb] = luminances;
    let vlr = _mm_set1_ps(lr);
    let vlg = _mm_set1_ps(lg);
    let vlb = _mm_set1_ps(lb);
    let mut r = r.chunks_exact_mut(4);
    let mut g = g.chunks_exact_mut(4);
    let mut b = b.chunks_exact_mut(4);
    for ((r, g), b) in (&mut r).zip(&mut g).zip(&mut b) {
        let vr = _mm_loadu_ps(r.as_ptr());
        let vg = _mm_loadu_ps(g.as_ptr());
        let vb = _mm_loadu_ps(b.as_ptr());
        let vy = _mm_mul_ps(vr, vlr);
        let vy = _mm_add_ps(vy, _mm_mul_ps(vg, vlg));
        let vy = _mm_add_ps(vy, _mm_mul_ps(vb, vlb));
        let vy_pq = crate::tf::pq::linear_to_pq_x86_64_sse2(vy, intensity_target);
        let vy_mapped = crate::tf::rec2408::rec2408_eetf_x86_64_sse2(
            vy_pq,
            intensity_target,
            from_luminance_range,
            to_luminance_range,
        );
        let vy_mapped = crate::tf::pq::pq_to_linear_x86_64_sse2(vy_mapped, intensity_target);
        let is_small = _mm_cmplt_ps(vy, _mm_set1_ps(1e-7));
        let vy = _mm_or_ps(
            _mm_andnot_ps(is_small, vy),
            _mm_and_ps(is_small, _mm_set1_ps(1.0)),
        );
        let ratio = _mm_div_ps(_mm_mul_ps(vy_mapped, vscale), vy);
        _mm_storeu_ps(r.as_mut_ptr(), _mm_mul_ps(vr, ratio));
        _mm_storeu_ps(g.as_mut_ptr(), _mm_mul_ps(vg, ratio));
        _mm_storeu_ps(b.as_mut_ptr(), _mm_mul_ps(vb, ratio));
    }

    tone_map_generic(
        r.into_remainder(),
        g.into_remainder(),
        b.into_remainder(),
        luminances,
        intensity_target,
        from_luminance_range,
        to_luminance_range,
    );
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn tone_map_aarch64_neon(
    r: &mut [f32],
    g: &mut [f32],
    b: &mut [f32],
    luminances: [f32; 3],
    intensity_target: f32,
    from_luminance_range: (f32, f32),
    to_luminance_range: (f32, f32),
) {
    use std::arch::aarch64::*;

    assert_eq!(r.len(), g.len());
    assert_eq!(g.len(), b.len());
    let scale = intensity_target / to_luminance_range.1;

    let [lr, lg, lb] = luminances;
    let mut r = r.chunks_exact_mut(4);
    let mut g = g.chunks_exact_mut(4);
    let mut b = b.chunks_exact_mut(4);
    for ((r, g), b) in (&mut r).zip(&mut g).zip(&mut b) {
        let vr = vld1q_f32(r.as_ptr());
        let vg = vld1q_f32(g.as_ptr());
        let vb = vld1q_f32(b.as_ptr());
        let vy = vfmaq_n_f32(vfmaq_n_f32(vmulq_n_f32(vr, lr), vg, lg), vb, lb);
        let vy_pq = crate::tf::pq::linear_to_pq_aarch64_neon(vy, intensity_target);
        let vy_mapped = crate::tf::rec2408::rec2408_eetf_aarch64_neon(
            vy_pq,
            intensity_target,
            from_luminance_range,
            to_luminance_range,
        );
        let vy_mapped = crate::tf::pq::pq_to_linear_aarch64_neon(vy_mapped, intensity_target);

        let is_small = vcleq_f32(vy, vdupq_n_f32(1e-7));
        let vy = vbslq_f32(is_small, vdupq_n_f32(1.0), vy);
        let ratio = vdivq_f32(vmulq_n_f32(vy_mapped, scale), vy);

        vst1q_f32(r.as_mut_ptr(), vmulq_f32(vr, ratio));
        vst1q_f32(g.as_mut_ptr(), vmulq_f32(vg, ratio));
        vst1q_f32(b.as_mut_ptr(), vmulq_f32(vb, ratio));
    }

    tone_map_generic(
        r.into_remainder(),
        g.into_remainder(),
        b.into_remainder(),
        luminances,
        intensity_target,
        from_luminance_range,
        to_luminance_range,
    );
}

fn tone_map_luma_generic(
    luma: &mut [f32],
    intensity_target: f32,
    from_luminance_range: (f32, f32),
    to_luminance_range: (f32, f32),
) {
    let scale = intensity_target / to_luminance_range.1;

    for y in luma {
        let y_pq = crate::tf::pq::linear_to_pq_generic(*y, intensity_target);
        let y_mapped = crate::tf::rec2408::rec2408_eetf_generic(
            y_pq,
            intensity_target,
            from_luminance_range,
            to_luminance_range,
        );
        let y_mapped = crate::tf::pq::pq_to_linear_generic(y_mapped, intensity_target);
        *y = y_mapped * scale;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "sse4.1")]
#[target_feature(enable = "fma")]
unsafe fn tone_map_luma_x86_64_avx2(
    luma: &mut [f32],
    intensity_target: f32,
    from_luminance_range: (f32, f32),
    to_luminance_range: (f32, f32),
) {
    use std::arch::x86_64::*;

    let scale = intensity_target / to_luminance_range.1;
    let vscale = _mm256_set1_ps(scale);

    let mut luma = luma.chunks_exact_mut(8);
    for y in &mut luma {
        let vy = _mm256_loadu_ps(y.as_ptr());
        let vy_pq = crate::tf::pq::linear_to_pq_x86_64_avx2(vy, intensity_target);
        let vy_mapped = crate::tf::rec2408::rec2408_eetf_x86_64_avx2(
            vy_pq,
            intensity_target,
            from_luminance_range,
            to_luminance_range,
        );
        let vy_mapped = crate::tf::pq::pq_to_linear_x86_64_avx2(vy_mapped, intensity_target);
        _mm256_storeu_ps(y.as_mut_ptr(), _mm256_mul_ps(vy_mapped, vscale));
    }

    tone_map_luma_x86_64_fma(
        luma.into_remainder(),
        intensity_target,
        from_luminance_range,
        to_luminance_range,
    );
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[target_feature(enable = "fma")]
unsafe fn tone_map_luma_x86_64_fma(
    luma: &mut [f32],
    intensity_target: f32,
    from_luminance_range: (f32, f32),
    to_luminance_range: (f32, f32),
) {
    use std::arch::x86_64::*;

    let scale = intensity_target / to_luminance_range.1;
    let vscale = _mm_set1_ps(scale);

    let mut luma = luma.chunks_exact_mut(4);
    for y in &mut luma {
        let vy = _mm_loadu_ps(y.as_ptr());
        let vy_pq = crate::tf::pq::linear_to_pq_x86_64_fma(vy, intensity_target);
        let vy_mapped = crate::tf::rec2408::rec2408_eetf_x86_64_fma(
            vy_pq,
            intensity_target,
            from_luminance_range,
            to_luminance_range,
        );
        let vy_mapped = crate::tf::pq::pq_to_linear_x86_64_fma(vy_mapped, intensity_target);
        _mm_storeu_ps(y.as_mut_ptr(), _mm_mul_ps(vy_mapped, vscale));
    }

    tone_map_luma_generic(
        luma.into_remainder(),
        intensity_target,
        from_luminance_range,
        to_luminance_range,
    );
}

#[cfg(target_arch = "x86_64")]
unsafe fn tone_map_luma_x86_64_sse2(
    luma: &mut [f32],
    intensity_target: f32,
    from_luminance_range: (f32, f32),
    to_luminance_range: (f32, f32),
) {
    use std::arch::x86_64::*;

    let scale = intensity_target / to_luminance_range.1;
    let vscale = _mm_set1_ps(scale);

    let mut luma = luma.chunks_exact_mut(4);
    for y in &mut luma {
        let vy = _mm_loadu_ps(y.as_ptr());
        let vy_pq = crate::tf::pq::linear_to_pq_x86_64_sse2(vy, intensity_target);
        let vy_mapped = crate::tf::rec2408::rec2408_eetf_x86_64_sse2(
            vy_pq,
            intensity_target,
            from_luminance_range,
            to_luminance_range,
        );
        let vy_mapped = crate::tf::pq::pq_to_linear_x86_64_sse2(vy_mapped, intensity_target);
        _mm_storeu_ps(y.as_mut_ptr(), _mm_mul_ps(vy_mapped, vscale));
    }

    tone_map_luma_generic(
        luma.into_remainder(),
        intensity_target,
        from_luminance_range,
        to_luminance_range,
    );
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn tone_map_luma_aarch64_neon(
    luma: &mut [f32],
    intensity_target: f32,
    from_luminance_range: (f32, f32),
    to_luminance_range: (f32, f32),
) {
    use std::arch::aarch64::*;

    let scale = intensity_target / to_luminance_range.1;

    let mut luma = luma.chunks_exact_mut(4);
    for y in &mut luma {
        let vy = vld1q_f32(y.as_ptr());
        let vy_pq = crate::tf::pq::linear_to_pq_aarch64_neon(vy, intensity_target);
        let vy_mapped = crate::tf::rec2408::rec2408_eetf_aarch64_neon(
            vy_pq,
            intensity_target,
            from_luminance_range,
            to_luminance_range,
        );
        let vy_mapped = crate::tf::pq::pq_to_linear_aarch64_neon(vy_mapped, intensity_target);
        vst1q_f32(y.as_mut_ptr(), vmulq_n_f32(vy_mapped, scale));
    }

    tone_map_luma_generic(
        luma.into_remainder(),
        intensity_target,
        from_luminance_range,
        to_luminance_range,
    );
}

fn detect_peak_luminance(r: &[f32], g: &[f32], b: &[f32], luminances: [f32; 3]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                return detect_peak_luminance_avx2(r, g, b, luminances);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if is_aarch64_feature_detected!("neon") {
            unsafe {
                return detect_peak_luminance_neon(r, g, b, luminances);
            }
        }
    }

    detect_peak_luminance_generic(r, g, b, luminances)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
unsafe fn detect_peak_luminance_avx2(r: &[f32], g: &[f32], b: &[f32], luminances: [f32; 3]) -> f32 {
    use std::arch::x86_64::*;

    assert_eq!(r.len(), g.len());
    assert_eq!(g.len(), b.len());

    let mut r = r.chunks_exact(8);
    let mut g = g.chunks_exact(8);
    let mut b = b.chunks_exact(8);

    let [lr, lg, lb] = luminances;
    let vlr = _mm256_set1_ps(lr);
    let vlg = _mm256_set1_ps(lg);
    let vlb = _mm256_set1_ps(lb);
    let mut peak_luminance = _mm256_setzero_ps();
    for ((r, g), b) in (&mut r).zip(&mut g).zip(&mut b) {
        let vr = _mm256_loadu_ps(r.as_ptr());
        let vg = _mm256_loadu_ps(g.as_ptr());
        let vb = _mm256_loadu_ps(b.as_ptr());
        let vy = _mm256_fmadd_ps(vb, vlb, _mm256_fmadd_ps(vg, vlg, _mm256_mul_ps(vr, vlr)));
        peak_luminance = _mm256_max_ps(peak_luminance, vy);
    }

    let mut peak_luminance = _mm_max_ps(
        _mm256_extractf128_ps::<0>(peak_luminance),
        _mm256_extractf128_ps::<1>(peak_luminance),
    );
    let mut r = r.remainder();
    let mut g = g.remainder();
    let mut b = b.remainder();
    if r.len() >= 4 {
        let (vr, remainder_r) = r.split_at(4);
        let (vg, remainder_g) = g.split_at(4);
        let (vb, remainder_b) = b.split_at(4);
        let vlr = _mm256_extractf128_ps::<0>(vlr);
        let vlg = _mm256_extractf128_ps::<0>(vlg);
        let vlb = _mm256_extractf128_ps::<0>(vlb);
        let vr = _mm_loadu_ps(vr.as_ptr());
        let vg = _mm_loadu_ps(vg.as_ptr());
        let vb = _mm_loadu_ps(vb.as_ptr());
        let vy = _mm_fmadd_ps(vb, vlb, _mm_fmadd_ps(vg, vlg, _mm_mul_ps(vr, vlr)));
        peak_luminance = _mm_max_ps(peak_luminance, vy);

        r = remainder_r;
        g = remainder_g;
        b = remainder_b;
    }
    let mut peak_luminance_arr = [0f32; 4];
    _mm_storeu_ps(peak_luminance_arr.as_mut_ptr(), peak_luminance);

    let mut peak_luminance = peak_luminance_arr
        .into_iter()
        .fold(0f32, |max, v| max.max(v));
    for ((r, g), b) in r.iter().zip(g).zip(b) {
        let y = b.mul_add(lb, g.mul_add(lg, *r * lr));
        peak_luminance = peak_luminance.max(y);
    }

    if peak_luminance <= 0.0 {
        1.0
    } else {
        peak_luminance
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn detect_peak_luminance_neon(r: &[f32], g: &[f32], b: &[f32], luminances: [f32; 3]) -> f32 {
    use std::arch::aarch64::*;

    assert_eq!(r.len(), g.len());
    assert_eq!(g.len(), b.len());

    let mut r = r.chunks_exact(4);
    let mut g = g.chunks_exact(4);
    let mut b = b.chunks_exact(4);

    let [lr, lg, lb] = luminances;
    let mut peak_luminance = vdupq_n_f32(0.0);
    for ((r, g), b) in (&mut r).zip(&mut g).zip(&mut b) {
        let vr = vld1q_f32(r.as_ptr());
        let vg = vld1q_f32(g.as_ptr());
        let vb = vld1q_f32(b.as_ptr());
        let vy = vfmaq_n_f32(vfmaq_n_f32(vmulq_n_f32(vr, lr), vg, lg), vb, lb);
        peak_luminance = vmaxq_f32(peak_luminance, vy);
    }

    let mut peak_luminance = vmaxvq_f32(peak_luminance);
    for ((r, g), b) in r.remainder().iter().zip(g.remainder()).zip(b.remainder()) {
        let y = b.mul_add(lb, g.mul_add(lg, *r * lr));
        peak_luminance = peak_luminance.max(y);
    }

    if peak_luminance <= 0.0 {
        1.0
    } else {
        peak_luminance
    }
}

#[inline]
fn detect_peak_luminance_generic(r: &[f32], g: &[f32], b: &[f32], luminances: [f32; 3]) -> f32 {
    assert_eq!(r.len(), g.len());
    assert_eq!(g.len(), b.len());

    let [lr, lg, lb] = luminances;
    let mut peak_luminance = 0f32;
    for ((r, g), b) in r.iter().zip(g).zip(b) {
        let y = r * lr + g * lg + b * lb;
        peak_luminance = if peak_luminance < y {
            y
        } else {
            peak_luminance
        };
    }

    if peak_luminance <= 0.0 {
        1.0
    } else {
        peak_luminance
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tone_map_range() {
        let mut samples = [0f32; 10];
        for (idx, v) in samples.iter_mut().enumerate() {
            *v = (idx / 5) as f32 * 0.1;
        }
        let mut r = samples;
        let mut g = samples;
        let mut b = samples;

        let hdr_params = HdrParams {
            luminances: [0.2126, 0.7152, 0.0722],
            intensity_target: 10000.0,
            min_nits: 0.0,
        };
        tone_map(&mut r, &mut g, &mut b, &hdr_params, 255.0);

        dbg!(r);
        dbg!(g);
        dbg!(b);

        for (idx, ((r, g), b)) in r.into_iter().zip(b).zip(b).enumerate() {
            let expected = (idx / 5) as f32;
            assert!((r - expected).abs() < 2e-5);
            assert!((g - expected).abs() < 2e-5);
            assert!((b - expected).abs() < 2e-5);
        }
    }

    #[test]
    fn detect_peak() {
        let samples = [0f32, 0.05, 0.075, 0.1];
        let r = samples;
        let g = samples;
        let b = samples;
        let peak = detect_peak_luminance(&r, &g, &b, [0.2126, 0.7152, 0.0722]);
        assert!((peak - 0.1).abs() < 1e-6);
    }

    #[test]
    fn detect_peak_zero() {
        let samples = [0f32, 0.0];
        let r = samples;
        let g = samples;
        let b = samples;
        let peak = detect_peak_luminance(&r, &g, &b, [0.2126, 0.7152, 0.0722]);
        assert!(peak == 1.0);
    }
}
