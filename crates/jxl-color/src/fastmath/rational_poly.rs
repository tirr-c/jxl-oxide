#[inline]
pub fn eval_generic<const P: usize, const Q: usize>(
    x: f32,
    p: [f32; P],
    q: [f32; Q],
) -> f32 {
    let yp = p.into_iter().rev().reduce(|yp, p| yp * x + p).unwrap();
    let yq = q.into_iter().rev().reduce(|yq, q| yq * x + q).unwrap();
    yp / yq
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

    let yp = p.into_iter().rev().map(|p| vdupq_n_f32(p)).reduce(|yp, p| vfmaq_f32(p, yp, x)).unwrap();
    let yq = q.into_iter().rev().map(|q| vdupq_n_f32(q)).reduce(|yq, q| vfmaq_f32(q, yq, x)).unwrap();
    vdivq_f32(yp, yq)
}
