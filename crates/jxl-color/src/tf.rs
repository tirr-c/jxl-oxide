/// Applies gamma to samples.
pub fn apply_gamma(mut samples: &mut [f32], gamma: f32) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            let mut it = samples.chunks_exact_mut(4);
            for chunk in &mut it {
                unsafe {
                    let v = std::arch::aarch64::vld1q_f32(chunk.as_ptr());
                    let mask =
                        std::arch::aarch64::vcleq_f32(v, std::arch::aarch64::vdupq_n_f32(1e-5));
                    let exp = crate::fastmath::fast_powf_aarch64_neon(v, gamma);
                    let v = std::arch::aarch64::vbslq_f32(
                        mask,
                        std::arch::aarch64::vdupq_n_f32(0.0),
                        exp,
                    );
                    std::arch::aarch64::vst1q_f32(chunk.as_mut_ptr(), v);
                }
            }
            let remainder = it.into_remainder();
            samples = remainder;
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            samples = unsafe { linear_to_gamma_x86_64_avx2(samples, gamma) };
        } else {
            let mut it = samples.chunks_exact_mut(4);
            for chunk in &mut it {
                unsafe {
                    let v = std::arch::x86_64::_mm_loadu_ps(chunk.as_ptr());
                    let mask =
                        std::arch::x86_64::_mm_cmple_ps(v, std::arch::x86_64::_mm_set1_ps(1e-5));
                    let exp = crate::fastmath::fast_powf_x86_64_sse2(
                        v,
                        std::arch::x86_64::_mm_set1_ps(gamma),
                    );
                    let v = std::arch::x86_64::_mm_andnot_ps(mask, exp);
                    std::arch::x86_64::_mm_storeu_ps(chunk.as_mut_ptr(), v);
                }
            }
            let remainder = it.into_remainder();
            samples = remainder;
        }
    }

    for x in samples {
        let a = *x;
        *x = if a <= 1e-5 {
            0.0
        } else {
            crate::fastmath::fast_powf_generic(a, gamma)
        };
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
unsafe fn linear_to_gamma_x86_64_avx2(samples: &mut [f32], gamma: f32) -> &mut [f32] {
    use std::arch::x86_64::*;

    let mut it = samples.chunks_exact_mut(8);
    for chunk in &mut it {
        let v = _mm256_loadu_ps(chunk.as_ptr());
        let mask = _mm256_cmp_ps(v, _mm256_set1_ps(1e-5), _CMP_LE_OS);
        let exp = crate::fastmath::fast_powf_x86_64_avx2(v, _mm256_set1_ps(gamma));
        let v = _mm256_andnot_ps(mask, exp);
        _mm256_storeu_ps(chunk.as_mut_ptr(), v);
    }
    let remainder = it.into_remainder();
    if remainder.len() < 4 {
        return remainder;
    }

    let (chunk, remainder) = remainder.split_at_mut(4);
    let v = _mm_loadu_ps(chunk.as_ptr());
    let mask = _mm_cmple_ps(v, _mm_set1_ps(1e-5));
    let exp = crate::fastmath::fast_powf_x86_64_fma(v, _mm_set1_ps(gamma));
    let v = _mm_andnot_ps(mask, exp);
    _mm_storeu_ps(chunk.as_mut_ptr(), v);

    remainder
}

#[repr(align(16))]
struct Aligned([u8; 16]);

const SRGB_POWTABLE_UPPER: Aligned = Aligned([
    0x00, 0x0a, 0x19, 0x26, 0x32, 0x41, 0x4d, 0x5c, 0x68, 0x75, 0x83, 0x8f, 0xa0, 0xaa, 0xb9, 0xc6,
]);
const SRGB_POWTABLE_LOWER: Aligned = Aligned([
    0x00, 0xb7, 0x04, 0x0d, 0xcb, 0xe7, 0x41, 0x68, 0x51, 0xd1, 0xeb, 0xf2, 0x00, 0xb7, 0x04, 0x0d,
]);

/// Converts the linear samples with the sRGB transfer curve.
// Fast linear to sRGB conversion, ported from libjxl.
pub fn linear_to_srgb(samples: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { linear_to_srgb_avx2(samples) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { linear_to_srgb_aarch64_neon(samples) };
        }
    }

    for s in samples {
        let v = s.to_bits() & 0x7fff_ffff;
        let v_adj = f32::from_bits((v | 0x3e80_0000) & 0x3eff_ffff);
        let pow = 0.059914046f32;
        let pow = pow * v_adj - 0.10889456;
        let pow = pow * v_adj + 0.107963754;
        let pow = pow * v_adj + 0.018092343;

        // `mul` won't be used when `v` is small.
        let idx = (v >> 23).wrapping_sub(118) as usize & 0xf;
        let mul = 0x4000_0000
            | (u32::from(SRGB_POWTABLE_UPPER.0[idx]) << 18)
            | (u32::from(SRGB_POWTABLE_LOWER.0[idx]) << 10);

        let v = f32::from_bits(v);
        let small = v * 12.92;
        let acc = pow * f32::from_bits(mul) - 0.055;

        *s = if v <= 0.0031308 { small } else { acc }.copysign(*s);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
unsafe fn linear_to_srgb_avx2(samples: &mut [f32]) {
    use std::arch::x86_64::*;

    let powtable_upper = _mm256_castps_si256(_mm256_broadcast_ps(
        &*(SRGB_POWTABLE_UPPER.0.as_ptr() as *const _),
    ));
    let powtable_lower = _mm256_castps_si256(_mm256_broadcast_ps(
        &*(SRGB_POWTABLE_LOWER.0.as_ptr() as *const _),
    ));

    let mut chunks = samples.chunks_exact_mut(8);
    let sign_mask = _mm256_set1_ps(f32::from_bits(0x8000_0000));
    for chunk in &mut chunks {
        let v = _mm256_loadu_ps(chunk.as_ptr());
        let sign = _mm256_and_ps(sign_mask, v);
        let v = _mm256_andnot_ps(sign_mask, v);

        let v_adj = _mm256_and_ps(
            _mm256_set1_ps(f32::from_bits(0x3eff_ffff)),
            _mm256_or_ps(_mm256_set1_ps(f32::from_bits(0x3e80_0000)), v),
        );
        let pow = _mm256_set1_ps(0.059914046);
        let pow = _mm256_fmadd_ps(pow, v_adj, _mm256_set1_ps(-0.10889456));
        let pow = _mm256_fmadd_ps(pow, v_adj, _mm256_set1_ps(0.107963754));
        let pow = _mm256_fmadd_ps(pow, v_adj, _mm256_set1_ps(0.018092343));

        let exp_idx = _mm256_sub_epi32(
            _mm256_srai_epi32::<23>(_mm256_castps_si256(v)),
            _mm256_set1_epi32(118),
        );
        let pow_upper = _mm256_shuffle_epi8(powtable_upper, exp_idx);
        let pow_lower = _mm256_shuffle_epi8(powtable_lower, exp_idx);
        let mul = _mm256_or_si256(
            _mm256_or_si256(
                _mm256_slli_epi32::<18>(pow_upper),
                _mm256_slli_epi32::<10>(pow_lower),
            ),
            _mm256_set1_epi32(0x4000_0000),
        );
        let mul = _mm256_castsi256_ps(mul);

        let small = _mm256_mul_ps(v, _mm256_set1_ps(12.92));
        let acc = _mm256_fmadd_ps(pow, mul, _mm256_set1_ps(-0.055));

        let mask = _mm256_cmp_ps(v, _mm256_set1_ps(0.0031308), _CMP_LE_OS);
        let ret = _mm256_blendv_ps(acc, small, mask);
        let ret = _mm256_or_ps(ret, sign);
        _mm256_storeu_ps(chunk.as_mut_ptr(), ret);
    }

    for s in chunks.into_remainder() {
        let v = s.to_bits() & 0x7fff_ffff;
        let v_adj = f32::from_bits((v | 0x3e80_0000) & 0x3eff_ffff);
        let pow = 0.059914046f32;
        let pow = pow.mul_add(v_adj, -0.10889456);
        let pow = pow.mul_add(v_adj, 0.107963754);
        let pow = pow.mul_add(v_adj, 0.018092343);

        let idx = (v >> 23).wrapping_sub(118) as usize & 0xf;
        let mul = 0x4000_0000
            | (u32::from(SRGB_POWTABLE_UPPER.0[idx]) << 18)
            | (u32::from(SRGB_POWTABLE_LOWER.0[idx]) << 10);

        let v = f32::from_bits(v);
        let small = v * 12.92;
        let acc = pow.mul_add(f32::from_bits(mul), -0.055);

        *s = if v <= 0.0031308 { small } else { acc }.copysign(*s);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn linear_to_srgb_aarch64_neon(samples: &mut [f32]) {
    use std::arch::aarch64::*;

    let mut it = samples.chunks_exact_mut(4);
    for chunk in &mut it {
        let v = vld1q_u32(chunk.as_ptr() as *const _);
        let sign = vandq_u32(v, vdupq_n_u32(0x8000_0000));
        let v = vandq_u32(v, vdupq_n_u32(0x7fff_ffff));

        let v_adj = vandq_u32(
            vorrq_u32(v, vdupq_n_u32(0x3e80_0000)),
            vdupq_n_u32(0x3eff_ffff),
        );
        let v_adj = vreinterpretq_f32_u32(v_adj);
        let pow = vfmaq_n_f32(vdupq_n_f32(-0.10889456), v_adj, 0.059914046);
        let pow = vfmaq_f32(vdupq_n_f32(0.107963754), v_adj, pow);
        let pow = vfmaq_f32(vdupq_n_f32(0.018092343), v_adj, pow);

        let exp_idx = vsubq_u32(vshrq_n_u32::<23>(v), vdupq_n_u32(118));
        let pow_upper = vqtbl1q_u8(
            vld1q_u8(SRGB_POWTABLE_UPPER.0.as_ptr()),
            vreinterpretq_u8_u32(exp_idx),
        );
        let pow_lower = vqtbl1q_u8(
            vld1q_u8(SRGB_POWTABLE_LOWER.0.as_ptr()),
            vreinterpretq_u8_u32(exp_idx),
        );
        let mul = vorrq_u32(
            vorrq_u32(
                vshlq_n_u32::<18>(vreinterpretq_u32_u8(pow_upper)),
                vshlq_n_u32::<10>(vreinterpretq_u32_u8(pow_lower)),
            ),
            vdupq_n_u32(0x4000_0000),
        );
        let mul = vreinterpretq_f32_u32(mul);

        let v = vreinterpretq_f32_u32(v);
        let small = vmulq_n_f32(v, 12.92);
        let acc = vfmaq_f32(vdupq_n_f32(-0.055), mul, pow);

        let mask = vcleq_f32(v, vdupq_n_f32(0.0031308));
        let ret = vbslq_f32(mask, small, acc);
        let ret = vorrq_u32(vreinterpretq_u32_f32(ret), sign);
        vst1q_u32(chunk.as_mut_ptr() as *mut _, ret);
    }

    for s in it.into_remainder() {
        let v = s.to_bits() & 0x7fff_ffff;
        let v_adj = f32::from_bits((v | 0x3e80_0000) & 0x3eff_ffff);
        let pow = 0.059914046f32;
        let pow = pow * v_adj - 0.10889456;
        let pow = pow * v_adj + 0.107963754;
        let pow = pow * v_adj + 0.018092343;

        let idx = (v >> 23).wrapping_sub(118) as usize & 0xf;
        let mul = 0x4000_0000
            | (u32::from(SRGB_POWTABLE_UPPER.0[idx]) << 18)
            | (u32::from(SRGB_POWTABLE_LOWER.0[idx]) << 10);

        let v = f32::from_bits(v);
        let small = v * 12.92;
        let acc = pow * f32::from_bits(mul) - 0.055;

        *s = if v <= 0.0031308 { small } else { acc }.copysign(*s);
    }
}

/// Converts samples in sRGB transfer curve to linear. Inverse of `linear_to_srgb`.
pub fn srgb_to_linear(samples: &mut [f32]) {
    #[allow(clippy::excessive_precision)]
    const P: [f32; 5] = [
        2.200248328e-4,
        1.043637593e-2,
        1.624820318e-1,
        7.961564959e-1,
        8.210152774e-1,
    ];
    #[allow(clippy::excessive_precision)]
    const Q: [f32; 5] = [
        2.631846970e-1,
        1.076976492,
        4.987528350e-1,
        -5.512498495e-2,
        6.521209011e-3,
    ];

    for x in samples {
        let a = x.abs();
        *x = if a <= 0.04045 {
            a / 12.92
        } else {
            crate::fastmath::rational_poly::eval_generic(a, P, Q)
        }
        .copysign(*x);
    }
}

/// Converts the linear samples with the BT.709 transfer curve.
pub fn linear_to_bt709(mut samples: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            let mut it = samples.chunks_exact_mut(4);
            for chunk in &mut it {
                unsafe {
                    let v = std::arch::aarch64::vld1q_f32(chunk.as_ptr());
                    let mask =
                        std::arch::aarch64::vcleq_f32(v, std::arch::aarch64::vdupq_n_f32(0.018));
                    let lin = std::arch::aarch64::vmulq_n_f32(v, 4.5);
                    let exp = crate::fastmath::fast_powf_aarch64_neon(v, 0.45);
                    let exp = std::arch::aarch64::vfmaq_n_f32(
                        std::arch::aarch64::vdupq_n_f32(-0.099),
                        exp,
                        1.099,
                    );
                    let v = std::arch::aarch64::vbslq_f32(mask, lin, exp);
                    std::arch::aarch64::vst1q_f32(chunk.as_mut_ptr(), v);
                }
            }
            let remainder = it.into_remainder();
            samples = remainder;
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            samples = unsafe { linear_to_bt709_x86_64_avx2(samples) };
        } else {
            let mut it = samples.chunks_exact_mut(4);
            for chunk in &mut it {
                unsafe {
                    let v = std::arch::x86_64::_mm_loadu_ps(chunk.as_ptr());
                    let mask =
                        std::arch::x86_64::_mm_cmple_ps(v, std::arch::x86_64::_mm_set1_ps(0.018));
                    let lin = std::arch::x86_64::_mm_mul_ps(v, std::arch::x86_64::_mm_set1_ps(4.5));
                    let exp = crate::fastmath::fast_powf_x86_64_sse2(
                        v,
                        std::arch::x86_64::_mm_set1_ps(0.45),
                    );
                    let exp = std::arch::x86_64::_mm_add_ps(
                        std::arch::x86_64::_mm_mul_ps(exp, std::arch::x86_64::_mm_set1_ps(1.099)),
                        std::arch::x86_64::_mm_set1_ps(-0.099),
                    );
                    let v = std::arch::x86_64::_mm_or_ps(
                        std::arch::x86_64::_mm_and_ps(mask, lin),
                        std::arch::x86_64::_mm_andnot_ps(mask, exp),
                    );
                    std::arch::x86_64::_mm_storeu_ps(chunk.as_mut_ptr(), v);
                }
            }
            let remainder = it.into_remainder();
            samples = remainder;
        }
    }

    for x in samples {
        let a = *x;
        *x = if a <= 0.018 {
            4.5 * a
        } else {
            crate::fastmath::fast_powf_generic(a, 0.45).mul_add(1.099, -0.099)
        };
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
unsafe fn linear_to_bt709_x86_64_avx2(samples: &mut [f32]) -> &mut [f32] {
    use std::arch::x86_64::*;

    let mut it = samples.chunks_exact_mut(8);
    for chunk in &mut it {
        let v = _mm256_loadu_ps(chunk.as_ptr());
        let mask = _mm256_cmp_ps(v, _mm256_set1_ps(0.018), _CMP_LE_OS);
        let lin = _mm256_mul_ps(v, _mm256_set1_ps(4.5));
        let exp = crate::fastmath::fast_powf_x86_64_avx2(v, _mm256_set1_ps(0.45));
        let exp = _mm256_fmadd_ps(exp, _mm256_set1_ps(1.099), _mm256_set1_ps(-0.099));
        let v = _mm256_blendv_ps(exp, lin, mask);
        _mm256_storeu_ps(chunk.as_mut_ptr(), v);
    }
    let remainder = it.into_remainder();
    if remainder.len() < 4 {
        return remainder;
    }

    let (chunk, remainder) = remainder.split_at_mut(4);
    let v = _mm_loadu_ps(chunk.as_ptr());
    let mask = _mm_cmple_ps(v, _mm_set1_ps(0.018));
    let lin = _mm_mul_ps(v, _mm_set1_ps(4.5));
    let exp = crate::fastmath::fast_powf_x86_64_fma(v, _mm_set1_ps(0.45));
    let exp = _mm_fmadd_ps(exp, _mm_set1_ps(1.099), _mm_set1_ps(-0.099));
    let v = _mm_blendv_ps(exp, lin, mask);
    _mm_storeu_ps(chunk.as_mut_ptr(), v);

    remainder
}

pub fn bt709_to_linear(samples: &mut [f32]) {
    for x in samples {
        let a = *x;
        *x = if a <= 0.081 {
            a / 4.5
        } else {
            crate::fastmath::fast_powf_generic(a.mul_add(1.0 / 1.099, 0.099 / 1.099), 1.0 / 0.45)
        };
    }
}

/// Converts the linear samples with the PQ transfer function, where linear sample value of 1.0
/// represents `intensity_target` nits.
pub fn linear_to_pq(samples: &mut [f32], intensity_target: f32) {
    const M1: f32 = 1305.0 / 8192.0;
    const M2: f32 = 2523.0 / 32.0;
    const C1: f32 = 107.0 / 128.0;
    const C2: f32 = 2413.0 / 128.0;
    const C3: f32 = 2392.0 / 128.0;

    let y_mult = intensity_target / 10000.0;

    for s in samples {
        let a = s.abs();
        let y_m1 = (a * y_mult).powf(M1);
        *s = ((y_m1.mul_add(C2, C1)) / (y_m1.mul_add(C3, 1.0)))
            .powf(M2)
            .copysign(*s);
    }
}

pub(crate) fn pq_table(n: usize) -> Vec<u16> {
    const M1_RECIP: f64 = 8192.0 / 1305.0;
    const M2_RECIP: f64 = 32.0 / 2523.0;
    const C1: f64 = 107.0 / 128.0;
    const C2: f64 = 2413.0 / 128.0;
    const C3: f64 = 2392.0 / 128.0;

    let mut out = vec![0u16; n];
    for (idx, out) in out.iter_mut().enumerate() {
        let e = idx as f64 / (n - 1) as f64;

        let e_pow = e.powf(M2_RECIP);
        let numerator = (e_pow - C1).max(0.0);
        let denominator = e_pow.mul_add(-C3, C2);
        let d = (numerator / denominator).powf(M1_RECIP);
        *out = (d * 65535.0) as u16; // clamped
    }
    out
}

pub fn hlg_oo(
    [samples_r, samples_g, samples_b]: [&mut [f32]; 3],
    [lr, lg, lb]: [f32; 3],
    intensity_target: f32,
) {
    let gamma = 1.2f32 * 1.111f32.powf((intensity_target / 1e3).log2());
    // 1/g - 1
    let exp = gamma - 1.0;

    for ((r, g), b) in samples_r.iter_mut().zip(samples_g).zip(samples_b) {
        let mixed = r.mul_add(lr, g.mul_add(lg, *b * lb));
        let mult = mixed.powf(exp);
        *r *= mult;
        *g *= mult;
        *b *= mult;
    }
}

/// Converts the display-referred samples to scene-referred signals using the hybrid log-gamma
/// transfer function.
pub fn hlg_inverse_oo(
    [samples_r, samples_g, samples_b]: [&mut [f32]; 3],
    [lr, lg, lb]: [f32; 3],
    intensity_target: f32,
) {
    let gamma = 1.2f32 * 1.111f32.powf((intensity_target / 1e3).log2());
    // 1/g - 1
    let exp = (1.0 - gamma) / gamma;

    for ((r, g), b) in samples_r.iter_mut().zip(samples_g).zip(samples_b) {
        let mixed = r.mul_add(lr, g.mul_add(lg, *b * lb));
        let mult = mixed.powf(exp);
        *r *= mult;
        *g *= mult;
        *b *= mult;
    }
}

const HLG_A: f32 = 0.17883277;
const HLG_B: f32 = 0.28466892;
const HLG_C: f32 = 0.5599107;

/// Converts the scene-referred linear samples with the hybrid log-gamma transfer function.
pub fn linear_to_hlg(samples: &mut [f32]) {
    for s in samples {
        let a = s.abs();
        *s = if a <= 1.0 / 12.0 {
            (3.0 * a).sqrt()
        } else {
            HLG_A * a.mul_add(12.0, -HLG_B).ln() + HLG_C
        }
        .copysign(*s);
    }
}

/// HLG inverse OETF, maps non-linear HLG signal to scene-referred linear sample.
pub fn hlg_to_linear(samples: &mut [f32]) {
    for s in samples {
        let a = s.abs();
        *s = if a <= 0.5 {
            a * a / 3.0
        } else {
            (((a - HLG_C) / HLG_A).exp() + HLG_B) / 12.0
        }
        .copysign(*s);
    }
}

pub(crate) fn hlg_table(n: usize) -> Vec<u16> {
    const A: f64 = 0.17883277;
    const B: f64 = 0.28466892;
    const C: f64 = 0.5599107;

    let mut out = vec![0u16; n];
    for (idx, out) in out[..=(n - 1) / 2].iter_mut().enumerate() {
        let d = (idx * idx) as f64 / (3 * (n - 1) * (n - 1)) as f64;
        *out = (d * 65535.0) as u16; // clamped
    }
    for (idx, out) in out[(n - 1) / 2 + 1..].iter_mut().enumerate() {
        let idx = idx + (n - 1) / 2 + 1;
        let e = idx as f64 / (n - 1) as f64;
        let d = (((e - C) / A).exp() + B) / 12.0;
        *out = (d * 65535.0) as u16; // clamped
    }
    out
}
