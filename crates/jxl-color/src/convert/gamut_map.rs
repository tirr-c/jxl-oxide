#[cfg(target_arch = "aarch64")]
use std::arch::is_aarch64_feature_detected;
#[cfg(target_arch = "x86_64")]
use std::arch::is_x86_feature_detected;

pub(super) fn gamut_map(
    mut r: &mut [f32],
    mut g: &mut [f32],
    mut b: &mut [f32],
    luminances: [f32; 3],
    saturation_factor: f32,
) {
    assert_eq!(r.len(), g.len());
    assert_eq!(g.len(), b.len());

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("fma") && is_x86_feature_detected!("sse4.1") {
        if is_x86_feature_detected!("avx2") {
            // AVX2
        }

        // SSE4.1 + FMA
    } else {
        // SSE2
    }

    #[cfg(target_arch = "aarch64")]
    if is_aarch64_feature_detected!("neon") {
        // NEON
        // SAFETY: features are checked above.
        unsafe {
            let mut r_it = r.chunks_exact_mut(4);
            let mut g_it = g.chunks_exact_mut(4);
            let mut b_it = b.chunks_exact_mut(4);

            for ((r, g), b) in (&mut r_it).zip(&mut g_it).zip(&mut b_it) {
                let rgb = [
                    std::arch::aarch64::vld1q_f32(r.as_ptr()),
                    std::arch::aarch64::vld1q_f32(g.as_ptr()),
                    std::arch::aarch64::vld1q_f32(b.as_ptr()),
                ];
                let [vr, vg, vb] =
                    crate::gamut::map_gamut_aarch64_neon(rgb, luminances, saturation_factor);
                std::arch::aarch64::vst1q_f32(r.as_mut_ptr(), vr);
                std::arch::aarch64::vst1q_f32(g.as_mut_ptr(), vg);
                std::arch::aarch64::vst1q_f32(b.as_mut_ptr(), vb);
            }

            r = r_it.into_remainder();
            g = g_it.into_remainder();
            b = b_it.into_remainder();
        }
    }

    // generic
    for ((r, g), b) in r.iter_mut().zip(g).zip(b) {
        let mapped = crate::gamut::map_gamut_generic([*r, *g, *b], luminances, saturation_factor);
        *r = mapped[0];
        *g = mapped[1];
        *b = mapped[2];
    }
}
