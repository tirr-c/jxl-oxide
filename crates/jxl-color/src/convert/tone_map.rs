use super::HdrParams;

pub(super) fn tone_map(
    r: &mut [f32],
    g: &mut [f32],
    b: &mut [f32],
    hdr_params: &HdrParams,
    target_display_luminance: f32,
) {
    assert_eq!(r.len(), g.len());
    assert_eq!(g.len(), b.len());

    let luminances = hdr_params.luminances;
    let intensity_target = hdr_params.intensity_target;
    let min_nits = hdr_params.min_nits;
    let detected_peak_luminance = detect_peak_luminance(r, g, b, luminances) * intensity_target;
    let peak_luminance = intensity_target.min(detected_peak_luminance);

    tone_map_generic(
        r,
        g,
        b,
        luminances,
        intensity_target,
        (min_nits, peak_luminance),
        (0.0, target_display_luminance),
    );
}

fn tone_map_generic(
    r: &mut [f32],
    g: &mut [f32],
    b: &mut [f32],
    luminances: [f32; 3],
    intensity_target: f32,
    from_luminance_range: (f32, f32),
    to_luminance_range: (f32, f32),
) {
    assert_eq!(r.len(), g.len());
    assert_eq!(g.len(), b.len());
    let scale = intensity_target / to_luminance_range.1;

    let [lr, lg, lb] = luminances;
    for ((r, g), b) in r.iter_mut().zip(g).zip(b) {
        let y = *r * lr + *g * lg + *b * lb;
        let mut y_pq = y;
        crate::tf::linear_to_pq(std::slice::from_mut(&mut y_pq), intensity_target);
        let mut y_mapped = crate::tf::rec2408_eetf_generic(
            y_pq,
            intensity_target,
            from_luminance_range,
            to_luminance_range,
        );
        crate::tf::pq_to_linear(std::slice::from_mut(&mut y_mapped), intensity_target);
        let ratio = y_mapped / y * scale;
        *r *= ratio;
        *g *= ratio;
        *b *= ratio;
    }
}

fn detect_peak_luminance(r: &[f32], g: &[f32], b: &[f32], luminances: [f32; 3]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            unsafe {
                return detect_peak_luminance_avx2(r, g, b, luminances);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                return detect_peak_luminance_neon(r, g, b, luminances);
            }
        }
    }

    detect_peak_luminance_impl(r, g, b, luminances)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn detect_peak_luminance_avx2(r: &[f32], g: &[f32], b: &[f32], luminances: [f32; 3]) -> f32 {
    detect_peak_luminance_impl(r, g, b, luminances)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn detect_peak_luminance_neon(r: &[f32], g: &[f32], b: &[f32], luminances: [f32; 3]) -> f32 {
    detect_peak_luminance_impl(r, g, b, luminances)
}

fn detect_peak_luminance_impl(r: &[f32], g: &[f32], b: &[f32], luminances: [f32; 3]) -> f32 {
    assert_eq!(r.len(), g.len());
    assert_eq!(g.len(), b.len());

    let [lr, lg, lb] = luminances;
    let mut peak_luminance = 0f32;
    for ((r, g), b) in r.iter().zip(g).zip(b) {
        let y = r * lr + g * lg + b * lb;
        peak_luminance = peak_luminance.max(y);
    }

    if peak_luminance <= 0.0 {
        1.0
    } else {
        peak_luminance
    }
}
