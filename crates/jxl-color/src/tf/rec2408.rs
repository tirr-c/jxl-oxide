use super::pq;

#[inline]
pub(crate) fn rec2408_eetf_generic(
    from_pq_sample: f32,
    intensity_target: f32,
    from_luminance_range: (f32, f32),
    to_luminance_range: (f32, f32),
) -> f32 {
    // Lb, Lw, Lmin, Lmax
    let mut luminances = [
        from_luminance_range.0 / intensity_target,
        from_luminance_range.1 / intensity_target,
        to_luminance_range.0 / intensity_target,
        to_luminance_range.1 / intensity_target,
    ];
    for y in &mut luminances {
        *y = pq::linear_to_pq_generic(*y, intensity_target);
    }

    // Step 1
    let source_pq_diff = luminances[1] - luminances[0];
    let normalized_source_pq_sample = (from_pq_sample - luminances[0]) / source_pq_diff;
    let min_luminance = (luminances[2] - luminances[0]) / source_pq_diff;
    let max_luminance = (luminances[3] - luminances[0]) / source_pq_diff;

    // Step 2
    let ks = 1.5 * max_luminance - 0.5;
    let b = min_luminance;

    // Step 3
    let compressed_pq_sample = if normalized_source_pq_sample < ks {
        normalized_source_pq_sample
    } else {
        // Step 4
        let one_sub_ks = 1.0 - ks;
        let t = (normalized_source_pq_sample - ks) / one_sub_ks;
        let t_p2 = t * t;
        let t_p3 = t_p2 * t;
        (2.0 * t_p3 - 3.0 * t_p2 + 1.0) * ks
            + (t_p3 - 2.0 * t_p2 + t) * one_sub_ks
            + (-2.0 * t_p3 + 3.0 * t_p2) * max_luminance
    };

    let one_sub_compressed_p4 = {
        let x = 1f32 - compressed_pq_sample;
        x * x * x * x
    };
    let normalized_target_pq_sample = one_sub_compressed_p4 * b + compressed_pq_sample;

    // Step 5
    normalized_target_pq_sample * source_pq_diff + luminances[0]
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "sse4.1")]
#[target_feature(enable = "fma")]
#[inline]
pub(crate) unsafe fn rec2408_eetf_x86_64_avx2(
    from_pq_sample: std::arch::x86_64::__m256,
    intensity_target: f32,
    from_luminance_range: (f32, f32),
    to_luminance_range: (f32, f32),
) -> std::arch::x86_64::__m256 {
    use jxl_grid::SimdVector;
    use std::arch::x86_64::*;

    // Lb, Lw, Lmin, Lmax
    let mut luminances = [
        from_luminance_range.0 / intensity_target,
        from_luminance_range.1 / intensity_target,
        to_luminance_range.0 / intensity_target,
        to_luminance_range.1 / intensity_target,
    ];
    for y in &mut luminances {
        *y = pq::linear_to_pq_generic(*y, intensity_target);
    }
    let v_min_source_luminance = _mm256_set1_ps(luminances[0]);

    // Step 1
    let source_pq_diff = luminances[1] - luminances[0];
    let v_source_pq_diff = _mm256_set1_ps(source_pq_diff);
    let normalized_source_pq_sample = from_pq_sample
        .sub(v_min_source_luminance)
        .div(v_source_pq_diff);
    let min_luminance = (luminances[2] - luminances[0]) / source_pq_diff;
    let max_luminance = (luminances[3] - luminances[0]) / source_pq_diff;

    // Step 2
    let ks = 1.5 * max_luminance - 0.5;
    let v_ks = _mm256_set1_ps(ks);
    let b = min_luminance;
    let v_b = _mm256_set1_ps(b);

    // Step 4
    let one_sub_ks = 1.0 - ks;
    let v_one_sub_ks = _mm256_set1_ps(one_sub_ks);
    let t = normalized_source_pq_sample.sub(v_ks).div(v_one_sub_ks);
    let t_p2 = t.mul(t);
    let t_p3 = t_p2.mul(t);

    let compressed_pq_sample = v_ks.muladd(
        t_p3.muladd(
            _mm256_set1_ps(2.0),
            t_p2.muladd(_mm256_set1_ps(-3.0), _mm256_set1_ps(1.0)),
        ),
        v_one_sub_ks.muladd(
            t_p2.muladd(_mm256_set1_ps(-2.0), t_p3.add(t)),
            _mm256_set1_ps(max_luminance)
                .mul(t_p3.muladd(_mm256_set1_ps(-2.0), t_p2.mul(_mm256_set1_ps(3.0)))),
        ),
    );

    // Step 3
    let is_small = _mm256_cmp_ps::<_CMP_LT_OQ>(normalized_source_pq_sample, v_ks);
    let compressed_pq_sample =
        _mm256_blendv_ps(compressed_pq_sample, normalized_source_pq_sample, is_small);

    let one_sub_compressed_p4 = {
        let x = _mm256_set1_ps(1.0).sub(compressed_pq_sample);
        let x2 = x.mul(x);
        x2.mul(x2)
    };
    let normalized_target_pq_sample = v_b.muladd(one_sub_compressed_p4, compressed_pq_sample);

    // Step 5
    normalized_target_pq_sample.muladd(v_source_pq_diff, v_min_source_luminance)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[target_feature(enable = "fma")]
#[inline]
pub(crate) unsafe fn rec2408_eetf_x86_64_fma(
    from_pq_sample: std::arch::x86_64::__m128,
    intensity_target: f32,
    from_luminance_range: (f32, f32),
    to_luminance_range: (f32, f32),
) -> std::arch::x86_64::__m128 {
    use jxl_grid::SimdVector;
    use std::arch::x86_64::*;

    // Lb, Lw, Lmin, Lmax
    let mut luminances = [
        from_luminance_range.0 / intensity_target,
        from_luminance_range.1 / intensity_target,
        to_luminance_range.0 / intensity_target,
        to_luminance_range.1 / intensity_target,
    ];
    for y in &mut luminances {
        *y = pq::linear_to_pq_generic(*y, intensity_target);
    }
    let v_min_source_luminance = _mm_set1_ps(luminances[0]);

    // Step 1
    let source_pq_diff = luminances[1] - luminances[0];
    let v_source_pq_diff = _mm_set1_ps(source_pq_diff);
    let normalized_source_pq_sample = from_pq_sample
        .sub(v_min_source_luminance)
        .div(v_source_pq_diff);
    let min_luminance = (luminances[2] - luminances[0]) / source_pq_diff;
    let max_luminance = (luminances[3] - luminances[0]) / source_pq_diff;

    // Step 2
    let ks = 1.5 * max_luminance - 0.5;
    let v_ks = _mm_set1_ps(ks);
    let b = min_luminance;
    let v_b = _mm_set1_ps(b);

    // Step 4
    let one_sub_ks = 1.0 - ks;
    let v_one_sub_ks = _mm_set1_ps(one_sub_ks);
    let t = normalized_source_pq_sample.sub(v_ks).div(v_one_sub_ks);
    let t_p2 = t.mul(t);
    let t_p3 = t_p2.mul(t);

    let compressed_pq_sample = v_ks.muladd(
        t_p3.muladd(
            _mm_set1_ps(2.0),
            t_p2.muladd(_mm_set1_ps(-3.0), _mm_set1_ps(1.0)),
        ),
        v_one_sub_ks.muladd(
            t_p2.muladd(_mm_set1_ps(-2.0), t_p3.add(t)),
            _mm_set1_ps(max_luminance)
                .mul(t_p3.muladd(_mm_set1_ps(-2.0), t_p2.mul(_mm_set1_ps(3.0)))),
        ),
    );

    // Step 3
    let is_small = _mm_cmplt_ps(normalized_source_pq_sample, v_ks);
    let compressed_pq_sample =
        _mm_blendv_ps(compressed_pq_sample, normalized_source_pq_sample, is_small);

    let one_sub_compressed_p4 = {
        let x = _mm_set1_ps(1.0).sub(compressed_pq_sample);
        let x2 = x.mul(x);
        x2.mul(x2)
    };
    let normalized_target_pq_sample = v_b.muladd(one_sub_compressed_p4, compressed_pq_sample);

    // Step 5
    normalized_target_pq_sample.muladd(v_source_pq_diff, v_min_source_luminance)
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) unsafe fn rec2408_eetf_x86_64_sse2(
    from_pq_sample: std::arch::x86_64::__m128,
    intensity_target: f32,
    from_luminance_range: (f32, f32),
    to_luminance_range: (f32, f32),
) -> std::arch::x86_64::__m128 {
    use jxl_grid::SimdVector;
    use std::arch::x86_64::*;

    // Lb, Lw, Lmin, Lmax
    let mut luminances = [
        from_luminance_range.0 / intensity_target,
        from_luminance_range.1 / intensity_target,
        to_luminance_range.0 / intensity_target,
        to_luminance_range.1 / intensity_target,
    ];
    for y in &mut luminances {
        *y = pq::linear_to_pq_generic(*y, intensity_target);
    }
    let v_min_source_luminance = _mm_set1_ps(luminances[0]);

    // Step 1
    let source_pq_diff = luminances[1] - luminances[0];
    let v_source_pq_diff = _mm_set1_ps(source_pq_diff);
    let normalized_source_pq_sample = from_pq_sample
        .sub(v_min_source_luminance)
        .div(v_source_pq_diff);
    let min_luminance = (luminances[2] - luminances[0]) / source_pq_diff;
    let max_luminance = (luminances[3] - luminances[0]) / source_pq_diff;

    // Step 2
    let ks = 1.5 * max_luminance - 0.5;
    let v_ks = _mm_set1_ps(ks);
    let b = min_luminance;
    let v_b = _mm_set1_ps(b);

    // Step 4
    let one_sub_ks = 1.0 - ks;
    let v_one_sub_ks = _mm_set1_ps(one_sub_ks);
    let t = normalized_source_pq_sample.sub(v_ks).div(v_one_sub_ks);
    let t_p2 = t.mul(t);
    let t_p3 = t_p2.mul(t);

    let compressed_pq_sample = v_ks.muladd(
        t_p3.muladd(
            _mm_set1_ps(2.0),
            t_p2.muladd(_mm_set1_ps(-3.0), _mm_set1_ps(1.0)),
        ),
        v_one_sub_ks.muladd(
            t_p2.muladd(_mm_set1_ps(-2.0), t_p3.add(t)),
            _mm_set1_ps(max_luminance)
                .mul(t_p3.muladd(_mm_set1_ps(-2.0), t_p2.mul(_mm_set1_ps(3.0)))),
        ),
    );

    // Step 3
    let is_small = _mm_cmplt_ps(normalized_source_pq_sample, v_ks);
    let compressed_pq_sample = _mm_or_ps(
        _mm_andnot_ps(is_small, compressed_pq_sample),
        _mm_and_ps(is_small, normalized_source_pq_sample),
    );

    let one_sub_compressed_p4 = {
        let x = _mm_set1_ps(1.0).sub(compressed_pq_sample);
        let x2 = x.mul(x);
        x2.mul(x2)
    };
    let normalized_target_pq_sample = v_b.muladd(one_sub_compressed_p4, compressed_pq_sample);

    // Step 5
    normalized_target_pq_sample.muladd(v_source_pq_diff, v_min_source_luminance)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub(crate) unsafe fn rec2408_eetf_aarch64_neon(
    from_pq_sample: std::arch::aarch64::float32x4_t,
    intensity_target: f32,
    from_luminance_range: (f32, f32),
    to_luminance_range: (f32, f32),
) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;

    // Lb, Lw, Lmin, Lmax
    let mut luminances = [
        from_luminance_range.0 / intensity_target,
        from_luminance_range.1 / intensity_target,
        to_luminance_range.0 / intensity_target,
        to_luminance_range.1 / intensity_target,
    ];
    for y in &mut luminances {
        *y = pq::linear_to_pq_generic(*y, intensity_target);
    }

    // Step 1
    let source_pq_diff = luminances[1] - luminances[0];
    let normalized_source_pq_sample = vmulq_n_f32(
        vsubq_f32(from_pq_sample, vdupq_n_f32(luminances[0])),
        source_pq_diff.recip(),
    );
    let min_luminance = (luminances[2] - luminances[0]) / source_pq_diff;
    let max_luminance = (luminances[3] - luminances[0]) / source_pq_diff;

    // Step 2
    let ks = 1.5 * max_luminance - 0.5;
    let b = min_luminance;

    // Step 4
    let one_sub_ks = 1.0 - ks;
    let t = vmulq_n_f32(
        vsubq_f32(normalized_source_pq_sample, vdupq_n_f32(ks)),
        one_sub_ks.recip(),
    );
    let t_p2 = vmulq_f32(t, t);
    let t_p3 = vmulq_f32(t_p2, t);
    let compressed_pq_sample = vfmaq_n_f32(
        vfmaq_n_f32(
            vmulq_n_f32(
                vfmaq_n_f32(vmulq_n_f32(t_p2, 3.0), t_p3, -2.0),
                max_luminance,
            ),
            vfmaq_n_f32(vaddq_f32(t_p3, t), t_p2, -2.0),
            one_sub_ks,
        ),
        vfmaq_n_f32(vfmaq_n_f32(vdupq_n_f32(1.0), t_p2, -3.0), t_p3, 2.0),
        ks,
    );

    // Step 3
    let is_small = vcltq_f32(normalized_source_pq_sample, vdupq_n_f32(ks));
    let compressed_pq_sample =
        vbslq_f32(is_small, normalized_source_pq_sample, compressed_pq_sample);

    let one_sub_compressed_p4 = {
        let x = vsubq_f32(vdupq_n_f32(1.0), compressed_pq_sample);
        let x2 = vmulq_f32(x, x);
        vmulq_f32(x2, x2)
    };
    let normalized_target_pq_sample = vfmaq_n_f32(compressed_pq_sample, one_sub_compressed_p4, b);

    // Step 5
    vfmaq_n_f32(
        vdupq_n_f32(luminances[0]),
        normalized_target_pq_sample,
        source_pq_diff,
    )
}
