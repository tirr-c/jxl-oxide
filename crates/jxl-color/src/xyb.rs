pub(crate) fn run(
    xyb: [&mut [f32]; 3],
    ob: [f32; 3],
    intensity_target: f32,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma") {
            // SAFETY: Feature set is checked above.
            return unsafe {
                run_x86_64_avx2(xyb, ob, intensity_target)
            };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: Feature set is checked above.
            return unsafe {
                run_aarch64_neon(xyb, ob, intensity_target)
            };
        }
    }

    run_generic(xyb, ob, intensity_target)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
unsafe fn run_x86_64_avx2(
    xyb: [&mut [f32]; 3],
    ob: [f32; 3],
    intensity_target: f32,
) {
    run_generic(xyb, ob, intensity_target)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn run_aarch64_neon(
    xyb: [&mut [f32]; 3],
    ob: [f32; 3],
    intensity_target: f32,
) {
    run_generic(xyb, ob, intensity_target)
}

#[inline(always)]
fn run_generic(
    xyb: [&mut [f32]; 3],
    ob: [f32; 3],
    intensity_target: f32,
) {
    let itscale = 255.0 / intensity_target;

    let [x, y, b] = xyb;
    if x.len() != y.len() || y.len() != b.len() {
        panic!("Grid size mismatch");
    }
    let cbrt_ob = ob.map(|v| v.cbrt());

    for ((x, y), b) in x.iter_mut().zip(&mut *y).zip(&mut *b) {
        // matrix: [1, 1, 0, -1, 1, 0, 0, 0, 1]
        let g_l = *y + *x;
        let g_m = *y - *x;
        let g_s = *b;

        // bias: -cbrt_ob
        let g_l = g_l - cbrt_ob[0];
        let g_m = g_m - cbrt_ob[1];
        let g_s = g_s - cbrt_ob[2];

        // inverse tf: gamma3, bias: ob, matrix: id * itscale
        *x = (g_l * g_l).mul_add(g_l, ob[0]) * itscale;
        *y = (g_m * g_m).mul_add(g_m, ob[1]) * itscale;
        *b = (g_s * g_s).mul_add(g_s, ob[2]) * itscale;
    }
}
