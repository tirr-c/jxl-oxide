use jxl_grid::AlignedGrid;

/// Applies transform from YCbCr to RGB.
///
/// Channels are expected to be in CbYCr order.
pub fn ycbcr_to_rgb(fb_cbycr: [&mut AlignedGrid<f32>; 3]) {
    let [cb, y, cr] = fb_cbycr;
    let cb = cb.buf_mut();
    let y = y.buf_mut();
    let cr = cr.buf_mut();

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma") {
            // SAFETY: Feature set is checked above.
            return unsafe { run_x86_64_avx2([cb, y, cr]) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: Feature set is checked above.
            return unsafe { run_aarch64_neon([cb, y, cr]) };
        }
    }

    run_generic([cb, y, cr])
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
unsafe fn run_x86_64_avx2(buf_cbycr: [&mut [f32]; 3]) {
    run_generic(buf_cbycr)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn run_aarch64_neon(buf_cbycr: [&mut [f32]; 3]) {
    run_generic(buf_cbycr)
}

#[inline(always)]
fn run_generic([cb, y, cr]: [&mut [f32]; 3]) {
    if cb.len() != y.len() || y.len() != cr.len() {
        panic!("Grid size mismatch");
    }
    for ((r, g), b) in cb.iter_mut().zip(y).zip(cr) {
        let cb = *r;
        let y = *g + 128.0 / 255.0;
        let cr = *b;

        *r = cr.mul_add(1.402, y);
        *g = cb.mul_add(
            -0.114 * 1.772 / 0.587,
            cr.mul_add(-0.299 * 1.402 / 0.587, y),
        );
        *b = cb.mul_add(1.772, y);
    }
}
