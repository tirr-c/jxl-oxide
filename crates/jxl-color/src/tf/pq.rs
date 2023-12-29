//! Approximation of perceptual quantizer transfer functions, ported from libjxl.

#[cfg(target_arch = "aarch64")]
use std::arch::is_aarch64_feature_detected;
#[cfg(target_arch = "x86_64")]
use std::arch::is_x86_feature_detected;

use crate::fastmath::rational_poly;

const EOTF_P: [f32; 5] = [
    2.6297566e-4,
    -6.235531e-3,
    7.386023e-1,
    2.6455317,
    5.500349e-1,
];
const EOTF_Q: [f32; 5] = [
    4.213501e2,
    -4.2873682e2,
    1.7436467e2,
    -3.3907887e1,
    2.6771877,
];

const INV_EOTF_P: [f32; 5] = [1.351392e-2, -1.095778, 5.522776e1, 1.492516e2, 4.838434e1];
const INV_EOTF_Q: [f32; 5] = [1.012416, 2.016708e1, 9.26371e1, 1.120607e2, 2.590418e1];
const INV_EOTF_P_SMALL: [f32; 5] = [
    9.863406e-6,
    3.881234e-1,
    1.352821e2,
    6.889862e4,
    -2.864824e5,
];
const INV_EOTF_Q_SMALL: [f32; 5] = [3.371868e1, 1.477719e3, 1.608477e4, -4.389884e4, -2.072546e5];

/// Converts the linear samples with the PQ transfer function, where linear sample value of 1.0
/// represents `intensity_target` nits (PQ inverse EOTF).
#[allow(unused_mut)]
pub fn linear_to_pq(mut samples: &mut [f32], intensity_target: f32) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("fma") && is_x86_feature_detected!("sse4.1") {
        if is_x86_feature_detected!("avx2") {
            let mut it = samples.chunks_exact_mut(8);
            // SAFETY: feature is checked above.
            unsafe {
                for s in &mut it {
                    let v = std::arch::x86_64::_mm256_loadu_ps(s.as_ptr());
                    std::arch::x86_64::_mm256_storeu_ps(
                        s.as_mut_ptr(),
                        linear_to_pq_x86_64_avx2(v, intensity_target),
                    );
                }
            }
            samples = it.into_remainder();
        }

        let mut it = samples.chunks_exact_mut(4);
        // SAFETY: feature is checked above.
        unsafe {
            for s in &mut it {
                let v = std::arch::x86_64::_mm_loadu_ps(s.as_ptr());
                std::arch::x86_64::_mm_storeu_ps(
                    s.as_mut_ptr(),
                    linear_to_pq_x86_64_fma(v, intensity_target),
                );
            }
        }
        samples = it.into_remainder();
    } else {
        let mut it = samples.chunks_exact_mut(4);
        // SAFETY: x86_64 implies SSE2.
        unsafe {
            for s in &mut it {
                let v = std::arch::x86_64::_mm_loadu_ps(s.as_ptr());
                std::arch::x86_64::_mm_storeu_ps(
                    s.as_mut_ptr(),
                    linear_to_pq_x86_64_sse2(v, intensity_target),
                );
            }
        }
        samples = it.into_remainder();
    }

    #[cfg(target_arch = "aarch64")]
    if is_aarch64_feature_detected!("neon") {
        let mut it = samples.chunks_exact_mut(4);
        // SAFETY: feature is checked above.
        unsafe {
            for s in &mut it {
                let v = std::arch::aarch64::vld1q_f32(s.as_ptr());
                std::arch::aarch64::vst1q_f32(
                    s.as_mut_ptr(),
                    linear_to_pq_aarch64_neon(v, intensity_target),
                );
            }
        }
        samples = it.into_remainder();
    }

    for s in samples {
        *s = linear_to_pq_generic(*s, intensity_target);
    }
}

#[inline]
pub(crate) fn linear_to_pq_generic(s: f32, intensity_target: f32) -> f32 {
    let y_mult = intensity_target / 10000.0;
    let a = s.abs();
    let a_scaled = a * y_mult;
    let a_1_4 = a_scaled.sqrt().sqrt();

    let y = if a < 1e-4 {
        rational_poly::eval_generic(a_1_4, INV_EOTF_P_SMALL, INV_EOTF_Q_SMALL)
    } else {
        rational_poly::eval_generic(a_1_4, INV_EOTF_P, INV_EOTF_Q)
    };

    y.copysign(s)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "sse4.1")]
#[target_feature(enable = "fma")]
#[inline]
pub(crate) unsafe fn linear_to_pq_x86_64_avx2(
    v: std::arch::x86_64::__m256,
    intensity_target: f32,
) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;

    let y_mult = intensity_target / 10000.0;
    let v_mult = _mm256_set1_ps(y_mult);
    let sign_mask = _mm256_set1_ps(f32::from_bits(0x8000_0000));

    let sign = _mm256_and_ps(sign_mask, v);
    let a = _mm256_andnot_ps(sign_mask, v);
    let a_scaled = _mm256_mul_ps(a, v_mult);
    let a_1_4 = _mm256_sqrt_ps(_mm256_sqrt_ps(a_scaled));

    let v_small = rational_poly::eval_x86_64_avx2(a_1_4, INV_EOTF_P_SMALL, INV_EOTF_Q_SMALL);
    let v_large = rational_poly::eval_x86_64_avx2(a_1_4, INV_EOTF_P, INV_EOTF_Q);
    let is_small = _mm256_cmp_ps::<_CMP_LE_OQ>(a, _mm256_set1_ps(1e-4));
    let v = _mm256_blendv_ps(v_large, v_small, is_small);
    _mm256_or_ps(_mm256_andnot_ps(sign_mask, v), sign)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[target_feature(enable = "fma")]
#[inline]
pub(crate) unsafe fn linear_to_pq_x86_64_fma(
    v: std::arch::x86_64::__m128,
    intensity_target: f32,
) -> std::arch::x86_64::__m128 {
    use std::arch::x86_64::*;

    let y_mult = intensity_target / 10000.0;
    let v_mult = _mm_set1_ps(y_mult);
    let sign_mask = _mm_set1_ps(f32::from_bits(0x8000_0000));

    let sign = _mm_and_ps(sign_mask, v);
    let a = _mm_andnot_ps(sign_mask, v);
    let a_scaled = _mm_mul_ps(a, v_mult);
    let a_1_4 = _mm_sqrt_ps(_mm_sqrt_ps(a_scaled));

    let v_small = rational_poly::eval_x86_64_fma(a_1_4, INV_EOTF_P_SMALL, INV_EOTF_Q_SMALL);
    let v_large = rational_poly::eval_x86_64_fma(a_1_4, INV_EOTF_P, INV_EOTF_Q);
    let is_small = _mm_cmplt_ps(a, _mm_set1_ps(1e-4));
    let v = _mm_blendv_ps(v_large, v_small, is_small);
    _mm_or_ps(_mm_andnot_ps(sign_mask, v), sign)
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) unsafe fn linear_to_pq_x86_64_sse2(
    v: std::arch::x86_64::__m128,
    intensity_target: f32,
) -> std::arch::x86_64::__m128 {
    use std::arch::x86_64::*;

    let y_mult = intensity_target / 10000.0;
    let v_mult = _mm_set1_ps(y_mult);
    let sign_mask = _mm_set1_ps(f32::from_bits(0x8000_0000));

    let sign = _mm_and_ps(sign_mask, v);
    let a = _mm_andnot_ps(sign_mask, v);
    let a_scaled = _mm_mul_ps(a, v_mult);
    let a_1_4 = _mm_sqrt_ps(_mm_sqrt_ps(a_scaled));

    let v_small = rational_poly::eval_x86_64_sse2(a_1_4, INV_EOTF_P_SMALL, INV_EOTF_Q_SMALL);
    let v_large = rational_poly::eval_x86_64_sse2(a_1_4, INV_EOTF_P, INV_EOTF_Q);
    let is_small = _mm_cmplt_ps(a, _mm_set1_ps(1e-4));
    let v = _mm_or_ps(
        _mm_andnot_ps(is_small, v_large),
        _mm_and_ps(is_small, v_small),
    );
    _mm_or_ps(_mm_andnot_ps(sign_mask, v), sign)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub(crate) unsafe fn linear_to_pq_aarch64_neon(
    v: std::arch::aarch64::float32x4_t,
    intensity_target: f32,
) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;

    let y_mult = intensity_target / 10000.0;
    let sign_mask = vdupq_n_u32(0x8000_0000);

    let sign = vandq_u32(vreinterpretq_u32_f32(v), sign_mask);
    let a = vabsq_f32(v);
    let a_scaled = vmulq_n_f32(a, y_mult);
    let a_1_4 = vsqrtq_f32(vsqrtq_f32(a_scaled));

    let v_small = rational_poly::eval_aarch64_neon(a_1_4, INV_EOTF_P_SMALL, INV_EOTF_Q_SMALL);
    let v_large = rational_poly::eval_aarch64_neon(a_1_4, INV_EOTF_P, INV_EOTF_Q);
    let is_small = vcltq_f32(a, vdupq_n_f32(1e-4));
    let v = vabsq_f32(vbslq_f32(is_small, v_small, v_large));
    vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(v), sign))
}

/// Converts non-linear PQ signals to linear display luminance values, where luminance value of 1.0
/// represents `intensity_target` nits (PQ EOTF).
#[allow(unused_mut)]
pub fn pq_to_linear(mut samples: &mut [f32], intensity_target: f32) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("fma") && is_x86_feature_detected!("sse4.1") {
        if is_x86_feature_detected!("avx2") {
            let mut it = samples.chunks_exact_mut(8);
            // SAFETY: feature is checked above.
            unsafe {
                for s in &mut it {
                    let v = std::arch::x86_64::_mm256_loadu_ps(s.as_ptr());
                    std::arch::x86_64::_mm256_storeu_ps(
                        s.as_mut_ptr(),
                        pq_to_linear_x86_64_avx2(v, intensity_target),
                    );
                }
            }
            samples = it.into_remainder();
        }

        let mut it = samples.chunks_exact_mut(4);
        // SAFETY: feature is checked above.
        unsafe {
            for s in &mut it {
                let v = std::arch::x86_64::_mm_loadu_ps(s.as_ptr());
                std::arch::x86_64::_mm_storeu_ps(
                    s.as_mut_ptr(),
                    pq_to_linear_x86_64_fma(v, intensity_target),
                );
            }
        }
        samples = it.into_remainder();
    } else {
        let mut it = samples.chunks_exact_mut(4);
        // SAFETY: x86_64 implies SSE2.
        unsafe {
            for s in &mut it {
                let v = std::arch::x86_64::_mm_loadu_ps(s.as_ptr());
                std::arch::x86_64::_mm_storeu_ps(
                    s.as_mut_ptr(),
                    pq_to_linear_x86_64_sse2(v, intensity_target),
                );
            }
        }
        samples = it.into_remainder();
    }

    #[cfg(target_arch = "aarch64")]
    if is_aarch64_feature_detected!("neon") {
        let mut it = samples.chunks_exact_mut(4);
        // SAFETY: feature is checked above.
        unsafe {
            for s in &mut it {
                let v = std::arch::aarch64::vld1q_f32(s.as_ptr());
                std::arch::aarch64::vst1q_f32(
                    s.as_mut_ptr(),
                    pq_to_linear_aarch64_neon(v, intensity_target),
                );
            }
        }
        samples = it.into_remainder();
    }

    for s in samples {
        *s = pq_to_linear_generic(*s, intensity_target);
    }
}

#[inline]
pub(crate) fn pq_to_linear_generic(s: f32, intensity_target: f32) -> f32 {
    let y_mult = 10000.0 / intensity_target;
    let a = s.abs();
    let x = a.mul_add(a, a);
    let y = rational_poly::eval_generic(x, EOTF_P, EOTF_Q);
    (y * y_mult).copysign(s)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
#[inline]
pub(crate) unsafe fn pq_to_linear_x86_64_avx2(
    v: std::arch::x86_64::__m256,
    intensity_target: f32,
) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;

    let y_mult = 10000.0 / intensity_target;
    let v_mult = _mm256_set1_ps(y_mult);
    let sign_mask = _mm256_set1_ps(f32::from_bits(0x8000_0000));

    let sign = _mm256_and_ps(sign_mask, v);
    let a = _mm256_andnot_ps(sign_mask, v);
    let x = _mm256_fmadd_ps(a, a, a);

    let y = rational_poly::eval_x86_64_avx2(x, EOTF_P, EOTF_Q);
    let v = _mm256_mul_ps(y, v_mult);
    _mm256_or_ps(_mm256_andnot_ps(sign_mask, v), sign)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
#[inline]
pub(crate) unsafe fn pq_to_linear_x86_64_fma(
    v: std::arch::x86_64::__m128,
    intensity_target: f32,
) -> std::arch::x86_64::__m128 {
    use std::arch::x86_64::*;

    let y_mult = 10000.0 / intensity_target;
    let v_mult = _mm_set1_ps(y_mult);
    let sign_mask = _mm_set1_ps(f32::from_bits(0x8000_0000));

    let sign = _mm_and_ps(sign_mask, v);
    let a = _mm_andnot_ps(sign_mask, v);
    let x = _mm_fmadd_ps(a, a, a);

    let y = rational_poly::eval_x86_64_fma(x, EOTF_P, EOTF_Q);
    let v = _mm_mul_ps(y, v_mult);
    _mm_or_ps(_mm_andnot_ps(sign_mask, v), sign)
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) unsafe fn pq_to_linear_x86_64_sse2(
    v: std::arch::x86_64::__m128,
    intensity_target: f32,
) -> std::arch::x86_64::__m128 {
    use std::arch::x86_64::*;

    let y_mult = 10000.0 / intensity_target;
    let v_mult = _mm_set1_ps(y_mult);
    let sign_mask = _mm_set1_ps(f32::from_bits(0x8000_0000));

    let sign = _mm_and_ps(sign_mask, v);
    let a = _mm_andnot_ps(sign_mask, v);
    let x = _mm_add_ps(_mm_mul_ps(a, a), a);

    let y = rational_poly::eval_x86_64_sse2(x, EOTF_P, EOTF_Q);
    let v = _mm_mul_ps(y, v_mult);
    _mm_or_ps(_mm_andnot_ps(sign_mask, v), sign)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub(crate) unsafe fn pq_to_linear_aarch64_neon(
    v: std::arch::aarch64::float32x4_t,
    intensity_target: f32,
) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;

    let y_mult = 10000.0 / intensity_target;
    let sign_mask = vdupq_n_u32(0x8000_0000);

    let sign = vandq_u32(vreinterpretq_u32_f32(v), sign_mask);
    let a = vabsq_f32(v);
    let x = vfmaq_f32(a, a, a);

    let y = rational_poly::eval_aarch64_neon(x, EOTF_P, EOTF_Q);
    let v = vmulq_n_f32(y, y_mult);
    vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(v), sign))
}

pub(crate) fn pq_table(n: usize) -> Vec<u16> {
    const M1_RECIP_F64: f64 = 8192.0 / 1305.0;
    const M2_RECIP_F64: f64 = 32.0 / 2523.0;
    const C1_F64: f64 = 107.0 / 128.0;
    const C2_F64: f64 = 2413.0 / 128.0;
    const C3_F64: f64 = 2392.0 / 128.0;

    let mut out = vec![0u16; n];
    for (idx, out) in out.iter_mut().enumerate() {
        let e = idx as f64 / (n - 1) as f64;

        let e_pow = e.powf(M2_RECIP_F64);
        let numerator = (e_pow - C1_F64).max(0.0);
        let denominator = e_pow.mul_add(-C3_F64, C2_F64);
        let d = (numerator / denominator).powf(M1_RECIP_F64);
        *out = (d * 65535.0) as u16; // clamped
    }
    out
}

#[cfg(test)]
mod tests {
    const M1: f32 = 1305.0 / 8192.0;
    const M2: f32 = 2523.0 / 32.0;
    const C1: f32 = 107.0 / 128.0;
    const C2: f32 = 2413.0 / 128.0;
    const C3: f32 = 2392.0 / 128.0;

    #[test]
    fn pq_inverse_eotf_100k_generic() {
        let mut input = Vec::with_capacity(100000);
        for (idx, v) in input.iter_mut().enumerate() {
            *v = idx as f32 * 1e-5;
        }

        for v in &mut input {
            *v = super::linear_to_pq_generic(*v, 10000.0);
        }

        for (idx, v) in input.iter().enumerate() {
            let linear = idx as f32 * 1e-5;
            let y_m1 = linear.powf(M1);
            let expected = ((y_m1.mul_add(C2, C1)) / (y_m1.mul_add(C3, 1.0))).powf(M2);
            let diff = (expected - *v).abs();
            assert!(diff < 1e-6);
        }
    }

    #[test]
    fn pq_inverse_eotf_100k() {
        let mut input = Vec::with_capacity(100000);
        for (idx, v) in input.iter_mut().enumerate() {
            *v = idx as f32 * 1e-5;
        }

        super::linear_to_pq(&mut input, 10000.0);

        for (idx, v) in input.iter().enumerate() {
            let linear = idx as f32 * 1e-5;
            let y_m1 = linear.powf(M1);
            let expected = ((y_m1.mul_add(C2, C1)) / (y_m1.mul_add(C3, 1.0))).powf(M2);
            let diff = (expected - *v).abs();
            assert!(diff < 1e-6);
        }
    }

    #[test]
    fn pq_roundtrip_10k_generic() {
        let mut input = Vec::with_capacity(10000);
        for (idx, v) in input.iter_mut().enumerate() {
            *v = idx as f32 * 1e-5;
        }

        for v in &mut input {
            let t = super::linear_to_pq_generic(*v, 10000.0);
            *v = super::pq_to_linear_generic(t, 1000.0);
        }

        for (idx, v) in input.iter().enumerate() {
            let expected = idx as f32 * 1e-4;
            let diff = (expected - *v).abs();
            assert!(diff < 1e-5);
        }
    }

    #[test]
    fn pq_roundtrip_10k() {
        let mut input = Vec::with_capacity(10000);
        for (idx, v) in input.iter_mut().enumerate() {
            *v = idx as f32 * 1e-5;
        }

        super::linear_to_pq(&mut input, 10000.0);
        super::pq_to_linear(&mut input, 1000.0);

        for (idx, v) in input.iter().enumerate() {
            let expected = idx as f32 * 1e-4;
            let diff = (expected - *v).abs();
            assert!(diff < 1e-5);
        }
    }
}
