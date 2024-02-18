#[cfg(target_arch = "aarch64")]
use std::arch::is_aarch64_feature_detected;
#[cfg(target_arch = "x86_64")]
use std::arch::is_x86_feature_detected;

pub(super) fn gamut_map(
    r: &mut [f32],
    g: &mut [f32],
    b: &mut [f32],
    luminances: [f32; 3],
    saturation_factor: f32,
) {
    assert_eq!(r.len(), g.len());
    assert_eq!(g.len(), b.len());

    #[cfg(target_arch = "x86_64")]
    let (r, g, b) = if is_x86_feature_detected!("fma") && is_x86_feature_detected!("sse4.1") {
        if is_x86_feature_detected!("avx2") {
            // AVX2
            // SAFETY: features are checked above.
            unsafe { gamut_map_x86_64_avx2(r, g, b, luminances, saturation_factor) }
        } else {
            // SSE4.1 + FMA
            // SAFETY: features are checked above.
            unsafe { gamut_map_x86_64_fma(r, g, b, luminances, saturation_factor) }
        }
    } else {
        // SAFETY: x86_64 implies SSE2.
        unsafe {
            let mut r_it = r.chunks_exact_mut(4);
            let mut g_it = g.chunks_exact_mut(4);
            let mut b_it = b.chunks_exact_mut(4);

            for ((r, g), b) in (&mut r_it).zip(&mut g_it).zip(&mut b_it) {
                let rgb = [
                    std::arch::x86_64::_mm_loadu_ps(r.as_ptr()),
                    std::arch::x86_64::_mm_loadu_ps(g.as_ptr()),
                    std::arch::x86_64::_mm_loadu_ps(b.as_ptr()),
                ];
                let [vr, vg, vb] =
                    crate::gamut::map_gamut_x86_64_sse2(rgb, luminances, saturation_factor);
                std::arch::x86_64::_mm_storeu_ps(r.as_mut_ptr(), vr);
                std::arch::x86_64::_mm_storeu_ps(g.as_mut_ptr(), vg);
                std::arch::x86_64::_mm_storeu_ps(b.as_mut_ptr(), vb);
            }

            let r = r_it.into_remainder();
            let g = g_it.into_remainder();
            let b = b_it.into_remainder();
            (r, g, b)
        }
    };

    #[cfg(target_arch = "aarch64")]
    let (r, g, b) = if is_aarch64_feature_detected!("neon") {
        // NEON
        // SAFETY: features are checked above.
        unsafe { gamut_map_aarch64_neon(r, g, b, luminances, saturation_factor) }
    } else {
        (r, g, b)
    };

    // generic
    for ((r, g), b) in r.iter_mut().zip(g).zip(b) {
        let mapped = crate::gamut::map_gamut_generic([*r, *g, *b], luminances, saturation_factor);
        *r = mapped[0];
        *g = mapped[1];
        *b = mapped[2];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
#[target_feature(enable = "sse4.1")]
pub(super) unsafe fn gamut_map_x86_64_avx2<'r, 'g, 'b>(
    r: &'r mut [f32],
    g: &'g mut [f32],
    b: &'b mut [f32],
    luminances: [f32; 3],
    saturation_factor: f32,
) -> (&'r mut [f32], &'g mut [f32], &'b mut [f32]) {
    let mut r_it = r.chunks_exact_mut(8);
    let mut g_it = g.chunks_exact_mut(8);
    let mut b_it = b.chunks_exact_mut(8);

    for ((r, g), b) in (&mut r_it).zip(&mut g_it).zip(&mut b_it) {
        let rgb = [
            std::arch::x86_64::_mm256_loadu_ps(r.as_ptr()),
            std::arch::x86_64::_mm256_loadu_ps(g.as_ptr()),
            std::arch::x86_64::_mm256_loadu_ps(b.as_ptr()),
        ];
        let [vr, vg, vb] = crate::gamut::map_gamut_x86_64_avx2(rgb, luminances, saturation_factor);
        std::arch::x86_64::_mm256_storeu_ps(r.as_mut_ptr(), vr);
        std::arch::x86_64::_mm256_storeu_ps(g.as_mut_ptr(), vg);
        std::arch::x86_64::_mm256_storeu_ps(b.as_mut_ptr(), vb);
    }

    (
        r_it.into_remainder(),
        g_it.into_remainder(),
        b_it.into_remainder(),
    )
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
#[target_feature(enable = "sse4.1")]
pub(super) unsafe fn gamut_map_x86_64_fma<'r, 'g, 'b>(
    r: &'r mut [f32],
    g: &'g mut [f32],
    b: &'b mut [f32],
    luminances: [f32; 3],
    saturation_factor: f32,
) -> (&'r mut [f32], &'g mut [f32], &'b mut [f32]) {
    let mut r_it = r.chunks_exact_mut(4);
    let mut g_it = g.chunks_exact_mut(4);
    let mut b_it = b.chunks_exact_mut(4);

    for ((r, g), b) in (&mut r_it).zip(&mut g_it).zip(&mut b_it) {
        let rgb = [
            std::arch::x86_64::_mm_loadu_ps(r.as_ptr()),
            std::arch::x86_64::_mm_loadu_ps(g.as_ptr()),
            std::arch::x86_64::_mm_loadu_ps(b.as_ptr()),
        ];
        let [vr, vg, vb] = crate::gamut::map_gamut_x86_64_fma(rgb, luminances, saturation_factor);
        std::arch::x86_64::_mm_storeu_ps(r.as_mut_ptr(), vr);
        std::arch::x86_64::_mm_storeu_ps(g.as_mut_ptr(), vg);
        std::arch::x86_64::_mm_storeu_ps(b.as_mut_ptr(), vb);
    }

    (
        r_it.into_remainder(),
        g_it.into_remainder(),
        b_it.into_remainder(),
    )
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(super) unsafe fn gamut_map_aarch64_neon<'r, 'g, 'b>(
    r: &'r mut [f32],
    g: &'g mut [f32],
    b: &'b mut [f32],
    luminances: [f32; 3],
    saturation_factor: f32,
) -> (&'r mut [f32], &'g mut [f32], &'b mut [f32]) {
    let mut r_it = r.chunks_exact_mut(4);
    let mut g_it = g.chunks_exact_mut(4);
    let mut b_it = b.chunks_exact_mut(4);

    for ((r, g), b) in (&mut r_it).zip(&mut g_it).zip(&mut b_it) {
        let rgb = [
            std::arch::aarch64::vld1q_f32(r.as_ptr()),
            std::arch::aarch64::vld1q_f32(g.as_ptr()),
            std::arch::aarch64::vld1q_f32(b.as_ptr()),
        ];
        let [vr, vg, vb] = crate::gamut::map_gamut_aarch64_neon(rgb, luminances, saturation_factor);
        std::arch::aarch64::vst1q_f32(r.as_mut_ptr(), vr);
        std::arch::aarch64::vst1q_f32(g.as_mut_ptr(), vg);
        std::arch::aarch64::vst1q_f32(b.as_mut_ptr(), vb);
    }

    (
        r_it.into_remainder(),
        g_it.into_remainder(),
        b_it.into_remainder(),
    )
}
