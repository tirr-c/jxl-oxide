#![allow(clippy::excessive_precision)]

const POW2F_NUMER_COEFFS: [f32; 3] = [1.01749063e1, 4.88687798e1, 9.85506591e1];
const POW2F_DENOM_COEFFS: [f32; 4] = [2.10242958e-1, -2.22328856e-2, -1.94414990e1, 9.85506633e1];

#[inline]
fn fast_pow2f_generic(x: f32) -> f32 {
    let x_floor = x.floor();
    let exp = f32::from_bits(((x_floor as i32 + 127) as u32) << 23);
    let frac = x - x_floor;

    let num = frac + POW2F_NUMER_COEFFS[0];
    let num = num * frac + POW2F_NUMER_COEFFS[1];
    let num = num * frac + POW2F_NUMER_COEFFS[2];
    let num = num * exp;

    let den = POW2F_DENOM_COEFFS[0] * frac + POW2F_DENOM_COEFFS[1];
    let den = den * frac + POW2F_DENOM_COEFFS[2];
    let den = den * frac + POW2F_DENOM_COEFFS[3];

    num / den
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn fast_pow2f_x86_64_sse2(x: std::arch::x86_64::__m128) -> std::arch::x86_64::__m128 {
    use std::arch::x86_64::*;

    unsafe {
        let x_floor = _mm_floor_ps(x);
        let exp = _mm_add_epi32(_mm_cvtps_epi32(x_floor), _mm_set1_epi32(127));
        let exp = _mm_castsi128_ps(_mm_slli_epi32::<23>(exp));
        let frac = _mm_sub_ps(x, x_floor);

        let num = _mm_add_ps(_mm_set1_ps(POW2F_NUMER_COEFFS[0]), frac);
        let num = _mm_add_ps(_mm_set1_ps(POW2F_NUMER_COEFFS[1]), _mm_mul_ps(frac, num));
        let num = _mm_add_ps(_mm_set1_ps(POW2F_NUMER_COEFFS[2]), _mm_mul_ps(frac, num));
        let num = _mm_mul_ps(exp, num);

        let den = _mm_add_ps(
            _mm_set1_ps(POW2F_DENOM_COEFFS[1]),
            _mm_mul_ps(frac, _mm_set1_ps(POW2F_DENOM_COEFFS[0])),
        );
        let den = _mm_add_ps(_mm_set1_ps(POW2F_DENOM_COEFFS[2]), _mm_mul_ps(frac, den));
        let den = _mm_add_ps(_mm_set1_ps(POW2F_DENOM_COEFFS[3]), _mm_mul_ps(frac, den));

        _mm_div_ps(num, den)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
#[inline]
unsafe fn fast_pow2f_x86_64_fma(x: std::arch::x86_64::__m128) -> std::arch::x86_64::__m128 {
    use std::arch::x86_64::*;

    let x_floor = _mm_floor_ps(x);
    let exp = _mm_add_epi32(_mm_cvtps_epi32(x_floor), _mm_set1_epi32(127));
    let exp = _mm_castsi128_ps(_mm_slli_epi32::<23>(exp));
    let frac = _mm_sub_ps(x, x_floor);

    let num = _mm_add_ps(_mm_set1_ps(POW2F_NUMER_COEFFS[0]), frac);
    let num = _mm_fmadd_ps(frac, num, _mm_set1_ps(POW2F_NUMER_COEFFS[1]));
    let num = _mm_fmadd_ps(frac, num, _mm_set1_ps(POW2F_NUMER_COEFFS[2]));
    let num = _mm_mul_ps(exp, num);

    let den = _mm_fmadd_ps(
        frac,
        _mm_set1_ps(POW2F_DENOM_COEFFS[0]),
        _mm_set1_ps(POW2F_DENOM_COEFFS[1]),
    );
    let den = _mm_fmadd_ps(frac, den, _mm_set1_ps(POW2F_DENOM_COEFFS[2]));
    let den = _mm_fmadd_ps(frac, den, _mm_set1_ps(POW2F_DENOM_COEFFS[3]));

    _mm_div_ps(num, den)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
#[inline]
unsafe fn fast_pow2f_x86_64_avx2(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;

    let x_floor = _mm256_floor_ps(x);
    let exp = _mm256_add_epi32(_mm256_cvtps_epi32(x_floor), _mm256_set1_epi32(127));
    let exp = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(exp));
    let frac = _mm256_sub_ps(x, x_floor);

    let num = _mm256_add_ps(_mm256_set1_ps(POW2F_NUMER_COEFFS[0]), frac);
    let num = _mm256_fmadd_ps(frac, num, _mm256_set1_ps(POW2F_NUMER_COEFFS[1]));
    let num = _mm256_fmadd_ps(frac, num, _mm256_set1_ps(POW2F_NUMER_COEFFS[2]));
    let num = _mm256_mul_ps(exp, num);

    let den = _mm256_fmadd_ps(
        frac,
        _mm256_set1_ps(POW2F_DENOM_COEFFS[0]),
        _mm256_set1_ps(POW2F_DENOM_COEFFS[1]),
    );
    let den = _mm256_fmadd_ps(frac, den, _mm256_set1_ps(POW2F_DENOM_COEFFS[2]));
    let den = _mm256_fmadd_ps(frac, den, _mm256_set1_ps(POW2F_DENOM_COEFFS[3]));

    _mm256_div_ps(num, den)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn fast_pow2f_aarch64_neon(
    x: std::arch::aarch64::float32x4_t,
) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;

    let x_floor = vrndmq_f32(x);
    let exp = vaddq_s32(vcvtq_s32_f32(x_floor), vdupq_n_s32(127));
    let exp = vreinterpretq_f32_s32(vshlq_n_s32(exp, 23));
    let frac = vsubq_f32(x, x_floor);

    let num = vaddq_f32(vdupq_n_f32(POW2F_NUMER_COEFFS[0]), frac);
    let num = vfmaq_f32(vdupq_n_f32(POW2F_NUMER_COEFFS[1]), frac, num);
    let num = vfmaq_f32(vdupq_n_f32(POW2F_NUMER_COEFFS[2]), frac, num);
    let num = vmulq_f32(exp, num);

    let den = vfmaq_n_f32(
        vdupq_n_f32(POW2F_DENOM_COEFFS[1]),
        frac,
        POW2F_DENOM_COEFFS[0],
    );
    let den = vfmaq_f32(vdupq_n_f32(POW2F_DENOM_COEFFS[2]), frac, den);
    let den = vfmaq_f32(vdupq_n_f32(POW2F_DENOM_COEFFS[3]), frac, den);

    vdivq_f32(num, den)
}

const LOG2F_P: [f32; 3] = [
    -1.8503833400518310e-6,
    1.4287160470083755,
    7.4245873327820566e-1,
];
const LOG2F_Q: [f32; 3] = [
    9.9032814277590719e-1,
    1.0096718572241148,
    1.7409343003366853e-1,
];

#[inline]
fn fast_log2f_generic(x: f32) -> f32 {
    let x_bits = x.to_bits() as i32;
    let exp_bits = x_bits - 0x3f2aaaab;
    let exp_shifted = exp_bits >> 23;
    let mantissa = f32::from_bits((x_bits - (exp_shifted << 23)) as u32);
    let exp_val = exp_shifted as f32;

    let x = mantissa - 1.0;
    super::rational_poly::eval_generic(x, LOG2F_P, LOG2F_Q) + exp_val
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn fast_log2f_x86_64_sse2(x: std::arch::x86_64::__m128) -> std::arch::x86_64::__m128 {
    use std::arch::x86_64::*;

    unsafe {
        let x_bits = _mm_castps_si128(x);
        let exp_bits = _mm_sub_epi32(x_bits, _mm_set1_epi32(0x3f2aaaab));
        let exp_shifted = _mm_srai_epi32::<23>(exp_bits);
        let mantissa = _mm_castsi128_ps(_mm_sub_epi32(x_bits, _mm_slli_epi32::<23>(exp_shifted)));
        let exp_val = _mm_cvtepi32_ps(exp_shifted);

        let x = _mm_sub_ps(mantissa, _mm_set1_ps(1.0));
        _mm_add_ps(
            super::rational_poly::eval_x86_64_sse2(x, LOG2F_P, LOG2F_Q),
            exp_val,
        )
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
#[inline]
unsafe fn fast_log2f_x86_64_fma(x: std::arch::x86_64::__m128) -> std::arch::x86_64::__m128 {
    use std::arch::x86_64::*;

    let x_bits = _mm_castps_si128(x);
    let exp_bits = _mm_sub_epi32(x_bits, _mm_set1_epi32(0x3f2aaaab));
    let exp_shifted = _mm_srai_epi32::<23>(exp_bits);
    let mantissa = _mm_castsi128_ps(_mm_sub_epi32(x_bits, _mm_slli_epi32::<23>(exp_shifted)));
    let exp_val = _mm_cvtepi32_ps(exp_shifted);

    let x = _mm_sub_ps(mantissa, _mm_set1_ps(1.0));
    _mm_add_ps(
        super::rational_poly::eval_x86_64_fma(x, LOG2F_P, LOG2F_Q),
        exp_val,
    )
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
#[inline]
unsafe fn fast_log2f_x86_64_avx2(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;

    let x_bits = _mm256_castps_si256(x);
    let exp_bits = _mm256_sub_epi32(x_bits, _mm256_set1_epi32(0x3f2aaaab));
    let exp_shifted = _mm256_srai_epi32::<23>(exp_bits);
    let mantissa = _mm256_castsi256_ps(_mm256_sub_epi32(
        x_bits,
        _mm256_slli_epi32::<23>(exp_shifted),
    ));
    let exp_val = _mm256_cvtepi32_ps(exp_shifted);

    let x = _mm256_sub_ps(mantissa, _mm256_set1_ps(1.0));
    _mm256_add_ps(
        super::rational_poly::eval_x86_64_avx2(x, LOG2F_P, LOG2F_Q),
        exp_val,
    )
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn fast_log2f_aarch64_neon(
    x: std::arch::aarch64::float32x4_t,
) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;

    let x_bits = vreinterpretq_s32_f32(x);
    let exp_bits = vsubq_s32(x_bits, vdupq_n_s32(0x3f2aaaab));
    let exp_shifted = vshrq_n_s32(exp_bits, 23);
    let mantissa = vreinterpretq_f32_s32(vsubq_s32(x_bits, vshlq_n_s32(exp_shifted, 23)));
    let exp_val = vcvtq_f32_s32(exp_shifted);

    let x = vsubq_f32(mantissa, vdupq_n_f32(1.0));
    vaddq_f32(
        super::rational_poly::eval_aarch64_neon(x, LOG2F_P, LOG2F_Q),
        exp_val,
    )
}

#[inline]
pub fn fast_powf_generic(base: f32, exp: f32) -> f32 {
    fast_pow2f_generic(fast_log2f_generic(base) * exp)
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub fn fast_powf_x86_64_sse2(
    base: std::arch::x86_64::__m128,
    exp: std::arch::x86_64::__m128,
) -> std::arch::x86_64::__m128 {
    unsafe {
        fast_pow2f_x86_64_sse2(std::arch::x86_64::_mm_mul_ps(
            fast_log2f_x86_64_sse2(base),
            exp,
        ))
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
#[inline]
pub unsafe fn fast_powf_x86_64_fma(
    base: std::arch::x86_64::__m128,
    exp: std::arch::x86_64::__m128,
) -> std::arch::x86_64::__m128 {
    fast_pow2f_x86_64_fma(std::arch::x86_64::_mm_mul_ps(
        fast_log2f_x86_64_fma(base),
        exp,
    ))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
#[inline]
pub unsafe fn fast_powf_x86_64_avx2(
    base: std::arch::x86_64::__m256,
    exp: std::arch::x86_64::__m256,
) -> std::arch::x86_64::__m256 {
    fast_pow2f_x86_64_avx2(std::arch::x86_64::_mm256_mul_ps(
        fast_log2f_x86_64_avx2(base),
        exp,
    ))
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn fast_powf_aarch64_neon(
    base: std::arch::aarch64::float32x4_t,
    exp: f32,
) -> std::arch::aarch64::float32x4_t {
    fast_pow2f_aarch64_neon(std::arch::aarch64::vmulq_n_f32(
        fast_log2f_aarch64_neon(base),
        exp,
    ))
}
