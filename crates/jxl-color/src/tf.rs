pub fn linear_to_gamma(samples: &mut [f32], gamma: f32) {
    for s in samples {
        let a = s.abs();
        *s = a.powf(gamma).copysign(*s);
    }
}

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

pub fn linear_to_bt709(samples: &mut [f32]) {
    for s in samples {
        let a = *s;
        *s = if a <= 0.018f32 {
            4.5 * a
        } else {
            1.099 * a.powf(0.45) - 0.099
        };
    }
}

pub fn linear_to_pq(samples: &mut [f32], intensity_target: f32) {
    const M1: f32 = 1305.0 / 8192.0;
    const M2: f32 = 2523.0 / 32.0;
    const C1: f32 = 107.0 / 128.0;
    const C2: f32 = 2413.0 / 128.0;
    const C3: f32 = 2392.0 / 128.0;

    let y_mult = intensity_target / 10000.0;

    for s in samples {
        let a = s.abs();
        let y_m1 = (a * y_mult).powf(M1);
        *s = ((C1 + C2 * y_m1) / (1.0 + C3 * y_m1)).powf(M2).copysign(*s);
    }
}

pub(crate) fn pq_table(n: usize) -> Vec<u16> {
    const M1_RECIP: f64 = 8192.0 / 1305.0;
    const M2_RECIP: f64 = 32.0 / 2523.0;
    const C1: f64 = 107.0 / 128.0;
    const C2: f64 = 2413.0 / 128.0;
    const C3: f64 = 2392.0 / 128.0;

    let mut out = vec![0u16; n];
    for (idx, out) in out.iter_mut().enumerate() {
        let e = idx as f64 / (n - 1) as f64;

        let e_pow = e.powf(M2_RECIP);
        let numerator = (e_pow - C1).max(0.0);
        let denominator = C2 - C3 * e_pow;
        let d = (numerator / denominator).powf(M1_RECIP);
        *out = (d * 65535.0) as u16; // clamped
    }
    out
}

pub fn hlg_inverse_oo(
    [samples_r, samples_g, samples_b]: [&mut [f32]; 3],
    [lr, lg, lb]: [f32; 3],
    intensity_target: f32,
) {
    let gamma = 1.2f32 * 1.111f32.powf((intensity_target / 1e3).log2());
    let exp = (1.0 - gamma) / gamma;

    for ((r, g), b) in samples_r.iter_mut().zip(samples_g).zip(samples_b) {
        let mixed = lr * *r + lg * *g + lb * *b;
        let mult = mixed.powf(exp);
        *r *= mult;
        *g *= mult;
        *b *= mult;
    }
}

pub fn linear_to_hlg(samples: &mut [f32]) {
    const A: f32 = 0.17883277;
    const B: f32 = 0.28466892;
    const C: f32 = 0.5599107;

    for s in samples {
        let a = s.abs();
        *s = if a <= 1.0 / 12.0 {
            (3.0 * a).sqrt()
        } else {
            A * (12.0 * a - B).ln() + C
        }.copysign(*s);
    }
}

pub(crate) fn hlg_table(n: usize) -> Vec<u16> {
    const A: f64 = 0.17883277;
    const B: f64 = 0.28466892;
    const C: f64 = 0.5599107;

    let mut out = vec![0u16; n];
    for (idx, out) in out[..=(n - 1) / 2].iter_mut().enumerate() {
        let d = (idx * idx) as f64 / (3 * (n - 1) * (n - 1)) as f64;
        *out = (d * 65535.0) as u16; // clamped
    }
    for (idx, out) in out[(n - 1) / 2 + 1..].iter_mut().enumerate() {
        let idx = idx + (n - 1) / 2 + 1;
        let e = idx as f64 / (n - 1) as f64;
        let d = (((e - C) / A).exp() + B) / 12.0;
        *out = (d * 65535.0) as u16; // clamped
    }
    out
}
