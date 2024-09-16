mod bt709;
pub(crate) mod pq;
pub(crate) mod rec2408;
mod srgb;

pub use bt709::*;
pub use pq::*;
pub use srgb::*;

/// Applies gamma to samples.
pub fn apply_gamma(samples: &mut [f32], gamma: f32) {
    #[cfg(target_arch = "aarch64")]
    let samples = {
        if std::arch::is_aarch64_feature_detected!("neon") {
            let mut it = samples.chunks_exact_mut(4);
            for chunk in &mut it {
                unsafe {
                    let v = std::arch::aarch64::vld1q_f32(chunk.as_ptr());
                    let mask =
                        std::arch::aarch64::vcleq_f32(v, std::arch::aarch64::vdupq_n_f32(1e-7));
                    let exp = crate::fastmath::fast_powf_aarch64_neon(v, gamma);
                    let v = std::arch::aarch64::vbslq_f32(
                        mask,
                        std::arch::aarch64::vdupq_n_f32(0.0),
                        exp,
                    );
                    std::arch::aarch64::vst1q_f32(chunk.as_mut_ptr(), v);
                }
            }
            it.into_remainder()
        } else {
            samples
        }
    };

    #[cfg(target_arch = "x86_64")]
    let samples = {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            unsafe { linear_to_gamma_x86_64_avx2(samples, gamma) }
        } else {
            let mut it = samples.chunks_exact_mut(4);
            for chunk in &mut it {
                unsafe {
                    let v = std::arch::x86_64::_mm_loadu_ps(chunk.as_ptr());
                    let mask =
                        std::arch::x86_64::_mm_cmple_ps(v, std::arch::x86_64::_mm_set1_ps(1e-7));
                    let exp = crate::fastmath::fast_powf_x86_64_sse2(
                        v,
                        std::arch::x86_64::_mm_set1_ps(gamma),
                    );
                    let v = std::arch::x86_64::_mm_andnot_ps(mask, exp);
                    std::arch::x86_64::_mm_storeu_ps(chunk.as_mut_ptr(), v);
                }
            }
            it.into_remainder()
        }
    };

    for x in samples {
        let a = *x;
        *x = if a <= 1e-7 {
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
        let mask = _mm256_cmp_ps(v, _mm256_set1_ps(1e-7), _CMP_LE_OS);
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
    let mask = _mm_cmple_ps(v, _mm_set1_ps(1e-7));
    let exp = crate::fastmath::fast_powf_x86_64_fma(v, _mm_set1_ps(gamma));
    let v = _mm_andnot_ps(mask, exp);
    _mm_storeu_ps(chunk.as_mut_ptr(), v);

    remainder
}

/// Converts scene luminance values to display luminance values using the hybrid log-gamma
/// transfer function (HLG OOTF).
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
/// transfer function (HLG inverse OOTF).
pub fn hlg_inverse_oo(
    [samples_r, samples_g, samples_b]: [&mut [f32]; 3],
    [lr, lg, lb]: [f32; 3],
    intensity_target: f32,
) {
    // System gamma results to ~1 in this range.
    if (295.0..=305.0).contains(&intensity_target) {
        return;
    }

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
