use std::{collections::BTreeMap, sync::Mutex};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum DctDirection {
    Forward,
    Inverse,
}

#[allow(clippy::excessive_precision)]
const SEC_HALF_SMALL: [&[f32]; 4] = [
    // n = 4
    &[0.541196100146197, 1.3065629648763764],
    // n = 8
    &[
        0.5097955791041592,
        0.6013448869350453,
        0.8999762231364156,
        2.5629154477415055,
    ],
    // n = 16
    &[
        0.5024192861881557,
        0.5224986149396889,
        0.5669440348163577,
        0.6468217833599901,
        0.7881546234512502,
        1.060677685990347,
        1.7224470982383342,
        5.101148618689155,
    ],
    // n = 32
    &[
        0.5006029982351963,
        0.5054709598975436,
        0.5154473099226246,
        0.5310425910897841,
        0.5531038960344445,
        0.5829349682061339,
        0.6225041230356648,
        0.6748083414550057,
        0.7445362710022984,
        0.8393496454155268,
        0.9725682378619608,
        1.1694399334328847,
        1.4841646163141662,
        2.057781009953411,
        3.407608418468719,
        10.190008123548033,
    ],
];

pub fn sec_half(n: usize) -> &'static [f32] {
    let idx = n.trailing_zeros() as usize - 2;

    static SEC_HALF_LARGE: Mutex<BTreeMap<usize, &'static [f32]>> = Mutex::new(BTreeMap::new());

    if let Some(idx) = idx.checked_sub(4) {
        let mut map = SEC_HALF_LARGE.lock().unwrap();
        map.entry(idx).or_insert_with(|| {
            let mut table = vec![0f32; n / 2];
            for (k, val) in table.iter_mut().enumerate() {
                let theta = (2 * k + 1) as f32 / (2 * n) as f32 * std::f32::consts::PI;
                *val = theta.cos().recip() / 2.0;
            }
            &*table.leak()
        })
    } else {
        SEC_HALF_SMALL[idx]
    }
}

pub const fn sec_half_small(n: usize) -> &'static [f32] {
    let idx = n.trailing_zeros() as usize - 2;
    SEC_HALF_SMALL[idx]
}
