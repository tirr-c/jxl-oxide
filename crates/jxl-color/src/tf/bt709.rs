/// Converts the linear samples with the BT.709 transfer curve.
pub fn linear_to_bt709(samples: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    let samples = {
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
            it.into_remainder()
        } else {
            samples
        }
    };

    #[cfg(target_arch = "x86_64")]
    let samples = {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            unsafe { linear_to_bt709_x86_64_avx2(samples) }
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
            it.into_remainder()
        }
    };

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
