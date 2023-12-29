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
