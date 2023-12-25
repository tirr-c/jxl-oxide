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
    super::linear_to_pq(&mut luminances, intensity_target);

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
