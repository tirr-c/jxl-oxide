#![allow(clippy::excessive_precision)]

#[inline]
fn fast_pow2f_generic(x: f32) -> f32 {
    let x_floor = x.floor();
    let exp = f32::from_bits(((x_floor as i32 + 127) as u32) << 23);
    let frac = x - x_floor;

    let num = frac + 1.01749063e1;
    let num = num * frac + 4.88687798e1;
    let num = num * frac + 9.85506591e1;
    let num = num * exp;

    let den = 2.10242958e-1 * frac - 2.22328856e-2;
    let den = den * frac - 1.94414990e1;
    let den = den * frac + 9.85506633e1;

    num / den
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn fast_pow2f_aarch64_neon(x: std::arch::aarch64::float32x4_t) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;

    let x_floor = vrndmq_f32(x);
    let exp = vaddq_s32(vcvtq_s32_f32(x_floor), vdupq_n_s32(127));
    let exp = vreinterpretq_f32_s32(vshlq_n_s32(exp, 23));
    let frac = vsubq_f32(x, x_floor);

    let num = vaddq_f32(vdupq_n_f32(1.01749063e1), frac);
    let num = vfmaq_f32(vdupq_n_f32(4.88687798e1), frac, num);
    let num = vfmaq_f32(vdupq_n_f32(9.85506591e1), frac, num);
    let num = vmulq_f32(exp, num);

    let den = vfmaq_n_f32(vdupq_n_f32(-2.22328856e-2), frac, 2.10242958e-1);
    let den = vfmaq_f32(vdupq_n_f32(-1.94414990e1), frac, den);
    let den = vfmaq_f32(vdupq_n_f32(9.85506633e1), frac, den);

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

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn fast_log2f_aarch64_neon(x: std::arch::aarch64::float32x4_t) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;

    let x_bits = vreinterpretq_s32_f32(x);
    let exp_bits = vsubq_s32(x_bits, vdupq_n_s32(0x3f2aaaab));
    let exp_shifted = vshrq_n_s32(exp_bits, 23);
    let mantissa = vreinterpretq_f32_s32(
        vsubq_s32(x_bits, vshlq_n_s32(exp_shifted, 23))
    );
    let exp_val = vcvtq_f32_s32(exp_shifted);

    let x = vsubq_f32(mantissa, vdupq_n_f32(1.0));
    vaddq_f32(super::rational_poly::eval_aarch64_neon(x, LOG2F_P, LOG2F_Q), exp_val)
}

#[inline]
pub fn fast_powf_generic(base: f32, exp: f32) -> f32 {
    fast_pow2f_generic(fast_log2f_generic(base) * exp)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn fast_powf_aarch64_neon(base: std::arch::aarch64::float32x4_t, exp: f32) -> std::arch::aarch64::float32x4_t {
    fast_pow2f_aarch64_neon(
        std::arch::aarch64::vmulq_n_f32(fast_log2f_aarch64_neon(base), exp)
    )
}

/*
pub fn fast_powf(base_out: &mut [f32], exp: f32) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            let mut it = base_out.chunks_exact_mut(4);
            for chunk in &mut it {
                unsafe {
                    let v = std::arch::aarch64::vld1q_f32(chunk.as_ptr());
                    let v = fast_powf_aarch64_neon(v, exp);
                    std::arch::aarch64::vst1q_f32(chunk.as_mut_ptr(), v);
                }
            }

            for x in it.into_remainder() {
                *x = fast_powf_generic(*x, exp);
            }

            return;
        }
    }

    for x in base_out {
        *x = fast_powf_generic(*x, exp);
    }
}
*/
