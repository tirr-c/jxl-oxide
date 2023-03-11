use std::{collections::BTreeMap, sync::Mutex};

const F1S2: f32 = std::f32::consts::FRAC_1_SQRT_2;

const COS_SIN_SMALL: [&[f32]; 4] = [
    &[1.0, 0.0, -1.0],
    &[1.0, F1S2, 0.0, -F1S2, -1.0, -F1S2],
    &[
        1.0,
        0.9238795,
        F1S2,
        0.38268343,
        0.0,
        -0.38268343,
        -F1S2,
        -0.9238795,
        -1.0,
        -0.9238795,
        -F1S2,
        -0.38268343,
    ],
    &[
        1.0,
        0.98078525,
        0.9238795,
        0.8314696,
        F1S2,
        0.55557024,
        0.38268343,
        0.19509032,

        0.0,
        -0.19509032,
        -0.38268343,
        -0.55557024,
        -F1S2,
        -0.8314696,
        -0.9238795,
        -0.98078525,

        -1.0,
        -0.98078525,
        -0.9238795,
        -0.8314696,
        -F1S2,
        -0.55557024,
        -0.38268343,
        -0.19509032,
    ],
];

pub fn cos_sin(n: usize) -> &'static [f32] {
    let idx = n.trailing_zeros() as usize - 2;

    static COS_SIN_LARGE: Mutex<BTreeMap<usize, &'static [f32]>> = Mutex::new(BTreeMap::new());

    if let Some(idx) = idx.checked_sub(4) {
        let mut map = COS_SIN_LARGE.lock().unwrap();
        map.entry(idx)
            .or_insert_with(|| {
                let mut cos_sin_table = vec![0f32; n / 4 * 3];
                for (k, cos) in cos_sin_table.iter_mut().enumerate() {
                    let theta = (2 * k) as f32 / n as f32 * std::f32::consts::PI;
                    *cos = theta.cos();
                }
                &*cos_sin_table.leak()
            })
    } else {
        COS_SIN_SMALL[idx]
    }
}

pub const fn cos_sin_small(n: usize) -> &'static [f32] {
    let idx = n.trailing_zeros() as usize - 2;
    COS_SIN_SMALL[idx]
}
