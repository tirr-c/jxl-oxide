pub fn linear_to_srgb(samples: &mut [f32]) {
    for s in samples {
        let a = s.abs();
        *s = if a <= 0.0031308f32 {
            12.92 * a
        } else {
            1.055 * a.powf(1.0 / 2.4) - 0.055
        }.copysign(*s);
    }
}
