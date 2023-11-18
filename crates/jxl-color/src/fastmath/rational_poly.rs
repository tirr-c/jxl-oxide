#[inline]
pub fn eval_generic<const P: usize, const Q: usize>(x: f32, p: [f32; P], q: [f32; Q]) -> f32 {
    let yp = p.into_iter().rev().reduce(|yp, p| yp * x + p).unwrap();
    let yq = q.into_iter().rev().reduce(|yq, q| yq * x + q).unwrap();
    yp / yq
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub fn eval_x86_64_sse2<const P: usize, const Q: usize>(
    x: std::arch::x86_64::__m128,
    p: [f32; P],
    q: [f32; Q],
) -> std::arch::x86_64::__m128 {
    use std::arch::x86_64::*;

    unsafe {
        let yp = p
            .into_iter()
            .rev()
            .map(|p| _mm_set1_ps(p))
            .reduce(|yp, p| _mm_add_ps(_mm_mul_ps(yp, x), p))
            .unwrap();
        let yq = q
            .into_iter()
            .rev()
            .map(|q| _mm_set1_ps(q))
            .reduce(|yq, q| _mm_add_ps(_mm_mul_ps(yq, x), q))
            .unwrap();
        _mm_div_ps(yp, yq)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
#[inline]
pub unsafe fn eval_x86_64_fma<const P: usize, const Q: usize>(
    x: std::arch::x86_64::__m128,
    p: [f32; P],
    q: [f32; Q],
) -> std::arch::x86_64::__m128 {
    use std::arch::x86_64::*;

    let yp = p
        .into_iter()
        .rev()
        .map(|p| _mm_set1_ps(p))
        .reduce(|yp, p| _mm_fmadd_ps(yp, x, p))
        .unwrap();
    let yq = q
        .into_iter()
        .rev()
        .map(|q| _mm_set1_ps(q))
        .reduce(|yq, q| _mm_fmadd_ps(yq, x, q))
        .unwrap();
    _mm_div_ps(yp, yq)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
#[inline]
pub unsafe fn eval_x86_64_avx2<const P: usize, const Q: usize>(
    x: std::arch::x86_64::__m256,
    p: [f32; P],
    q: [f32; Q],
) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;

    let yp = p
        .into_iter()
        .rev()
        .map(|p| _mm256_set1_ps(p))
        .reduce(|yp, p| _mm256_fmadd_ps(yp, x, p))
        .unwrap();
    let yq = q
        .into_iter()
        .rev()
        .map(|q| _mm256_set1_ps(q))
        .reduce(|yq, q| _mm256_fmadd_ps(yq, x, q))
        .unwrap();
    _mm256_div_ps(yp, yq)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn eval_aarch64_neon<const P: usize, const Q: usize>(
    x: std::arch::aarch64::float32x4_t,
    p: [f32; P],
    q: [f32; Q],
) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;

    let yp = p
        .into_iter()
        .rev()
        .map(|p| vdupq_n_f32(p))
        .reduce(|yp, p| vfmaq_f32(p, yp, x))
        .unwrap();
    let yq = q
        .into_iter()
        .rev()
        .map(|q| vdupq_n_f32(q))
        .reduce(|yq, q| vfmaq_f32(q, yq, x))
        .unwrap();
    vdivq_f32(yp, yq)
}
