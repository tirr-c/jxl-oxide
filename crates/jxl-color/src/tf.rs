/// Converts the linear samples with the given gamma.
pub fn linear_to_gamma(samples: &mut [f32], gamma: f32) {
    for s in samples {
        let a = s.abs();
        *s = a.powf(gamma).copysign(*s);
    }
}

/// Converts the linear samples with the sRGB transfer curve.
// Fast linear to sRGB conversion, ported from libjxl.
pub fn linear_to_srgb(samples: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { linear_to_srgb_avx2(samples) };
        }
    }

    const POWTABLE_UPPER: [u8; 16] = [
        0x00, 0x0a, 0x19, 0x26, 0x32, 0x41, 0x4d, 0x5c,
        0x68, 0x75, 0x83, 0x8f, 0xa0, 0xaa, 0xb9, 0xc6,
    ];
    const POWTABLE_LOWER: [u8; 16] = [
        0x00, 0xb7, 0x04, 0x0d, 0xcb, 0xe7, 0x41, 0x68,
        0x51, 0xd1, 0xeb, 0xf2, 0x00, 0xb7, 0x04, 0x0d,
    ];

    for s in samples {
        let v = s.to_bits() & 0x7fff_ffff;
        let v_adj = f32::from_bits((v | 0x3e80_0000) & 0x3eff_ffff);
        let pow = 0.059914046f32;
        let pow = pow * v_adj - 0.10889456;
        let pow = pow * v_adj + 0.107963754;
        let pow = pow * v_adj + 0.018092343;

        // `mul` won't be used when `v` is small.
        let idx = (v >> 23).saturating_sub(118) as usize & 15;
        let mul = 0x4000_0000 | (u32::from(POWTABLE_UPPER[idx]) << 18) | (u32::from(POWTABLE_LOWER[idx]) << 10);

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

    #[repr(align(16))]
    struct Aligned([u8; 16]);

    const POWTABLE_UPPER: Aligned = Aligned([
        0x00, 0x0a, 0x19, 0x26, 0x32, 0x41, 0x4d, 0x5c,
        0x68, 0x75, 0x83, 0x8f, 0xa0, 0xaa, 0xb9, 0xc6,
    ]);
    const POWTABLE_LOWER: Aligned = Aligned([
        0x00, 0xb7, 0x04, 0x0d, 0xcb, 0xe7, 0x41, 0x68,
        0x51, 0xd1, 0xeb, 0xf2, 0x00, 0xb7, 0x04, 0x0d,
    ]);

    let powtable_upper = _mm256_castps_si256(
        _mm256_broadcast_ps(&*(POWTABLE_UPPER.0.as_ptr() as *const _))
    );
    let powtable_lower = _mm256_castps_si256(
        _mm256_broadcast_ps(&*(POWTABLE_LOWER.0.as_ptr() as *const _))
    );

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
        let pow_upper = _mm256_shuffle_epi8(
            powtable_upper,
            exp_idx,
        );
        let pow_lower = _mm256_shuffle_epi8(
            powtable_lower,
            exp_idx,
        );
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
        let ret = _mm256_or_ps(
            _mm256_and_ps(mask, small),
            _mm256_andnot_ps(mask, acc),
        );
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

        let idx = ((v >> 23) - 118) as usize;
        let mul = 0x4000_0000 | (u32::from(POWTABLE_UPPER.0[idx]) << 18) | (u32::from(POWTABLE_LOWER.0[idx]) << 10);

        let v = f32::from_bits(v);
        let small = v * 12.92;
        let acc = pow.mul_add(f32::from_bits(mul), -0.055);

        *s = if v <= 0.0031308 { small } else { acc }.copysign(*s);
    }
}

/// Converts the linear samples with the BT.709 transfer curve.
pub fn linear_to_bt709(samples: &mut [f32]) {
    for s in samples {
        let a = *s;
        *s = if a <= 0.018f32 {
            4.5 * a
        } else {
            a.powf(0.45).mul_add(1.099, -0.099)
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
        *s = ((y_m1.mul_add(C2, C1)) / (y_m1.mul_add(C3, 1.0))).powf(M2).copysign(*s);
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

/// Converts the display-referred samples to scene-referred signals using the hybrid log-gamma
/// transfer function.
pub fn hlg_inverse_oo(
    [samples_r, samples_g, samples_b]: [&mut [f32]; 3],
    [lr, lg, lb]: [f32; 3],
    intensity_target: f32,
) {
    let gamma = 1.2f32 * 1.111f32.powf((intensity_target / 1e3).log2());
    let exp = (1.0 - gamma) / gamma;

    for ((r, g), b) in samples_r.iter_mut().zip(samples_g).zip(samples_b) {
        let mixed = r.mul_add(lr, g.mul_add(lg, *b * lb));
        let mult = mixed.powf(exp);
        *r *= mult;
        *g *= mult;
        *b *= mult;
    }
}

/// Converts the scene-referred linear samples with the hybrid log-gamma transfer function.
pub fn linear_to_hlg(samples: &mut [f32]) {
    const A: f32 = 0.17883277;
    const B: f32 = 0.28466892;
    const C: f32 = 0.5599107;

    for s in samples {
        let a = s.abs();
        *s = if a <= 1.0 / 12.0 {
            (3.0 * a).sqrt()
        } else {
            A * a.mul_add(12.0, -B).ln() + C
        }.copysign(*s);
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
