use jxl_grid::SimpleGrid;
use jxl_vardct::TransformType;

use crate::dct::idct_2d_in_place;

fn aux_idct2_in_place(block: &mut SimpleGrid<f32>, size: usize) {
    debug_assert!(size.is_power_of_two());

    let stride = block.width();
    let block = block.buf_mut();

    let num_2x2 = size / 2;
    let mut scratch = vec![0.0f32; size * size];
    for y in 0..num_2x2 {
        for x in 0..num_2x2 {
            let c00 = block[y * stride + x];
            let c01 = block[y * stride + x + num_2x2];
            let c10 = block[(y + num_2x2) * stride + x];
            let c11 = block[(y + num_2x2) * stride + x + num_2x2];

            let base_idx = 2 * (y * size + x);
            scratch[base_idx] = c00 + c01 + c10 + c11;
            scratch[base_idx + 1] = c00 + c01 - c10 - c11;
            scratch[base_idx + size] = c00 - c01 + c10 - c11;
            scratch[base_idx + size + 1] = c00 - c01 - c10 + c11;
        }
    }

    for y in 0..size {
        block[y * stride..][..size].copy_from_slice(&scratch[y * size..][..size]);
    }
}

fn transform_dct2(coeff: &mut SimpleGrid<f32>) {
    aux_idct2_in_place(coeff, 2);
    aux_idct2_in_place(coeff, 4);
    aux_idct2_in_place(coeff, 8);
}

fn transform_dct4(coeff: &mut SimpleGrid<f32>) {
    aux_idct2_in_place(coeff, 2);
    let coeff = coeff.buf_mut();

    let mut scratch = [0.0f32; 64];
    for y in 0..2 {
        for x in 0..2 {
            let scratch = &mut scratch[(y * 2 + x) * 16..][..16];
            for iy in 0..4 {
                for ix in 0..4 {
                    scratch[iy * 4 + ix] = coeff[(y + iy * 2) * 8 + x + ix * 2];
                }
            }
            idct_2d_in_place(scratch, 4, 4);
        }
    }

    for y in 0..2 {
        for x in 0..2 {
            let scratch = &scratch[(y * 2 + x) * 16..][..16];
            for iy in 0..4 {
                for ix in 0..4 {
                    coeff[(y * 4 + iy) * 8 + x * 4 + ix] = scratch[iy * 4 + ix];
                }
            }
        }
    }
}

fn transform_hornuss(coeff: &mut SimpleGrid<f32>) {
    aux_idct2_in_place(coeff, 2);
    let coeff = coeff.buf_mut();

    let mut scratch = [0.0f32; 64];
    for y in 0..2 {
        for x in 0..2 {
            let scratch = &mut scratch[(y * 2 + x) * 16..][..16];
            for iy in 0..4 {
                for ix in 0..4 {
                    scratch[iy * 4 + ix] = coeff[(y + iy * 2) * 8 + x + ix * 2];
                }
            }
            let residual_sum: f32 = scratch[1..].iter().copied().sum();
            let avg = scratch[0] - residual_sum / 16.0;
            scratch[0] = scratch[5] + avg;
            scratch[5] = avg;
            for (idx, s) in scratch.iter_mut().enumerate() {
                if idx == 0 || idx == 5 {
                    continue;
                }
                *s += avg;
            }
        }
    }

    for y in 0..2 {
        for x in 0..2 {
            let scratch = &mut scratch[(y * 2 + x) * 16..][..16];
            for iy in 0..4 {
                for ix in 0..4 {
                    coeff[(y * 4 + iy) * 8 + x * 4 + ix] = scratch[iy * 4 + ix];
                }
            }
        }
    }
}

fn transform_dct4x8(coeff: &mut SimpleGrid<f32>, transpose: bool) {
    let coeff = coeff.buf_mut();
    let coeff0 = coeff[0];
    let coeff1 = coeff[8];
    coeff[0] = coeff0 + coeff1;
    coeff[8] = coeff0 - coeff1;

    let (w, h) = if transpose {
        (4, 8)
    } else {
        (8, 4)
    };

    let mut scratch = [0.0f32; 64];
    for idx in [0, 1] {
        let scratch = &mut scratch[(idx * 32)..][..32];
        for iy in 0..4 {
            for ix in 0..8 {
                scratch[iy * 8 + ix] = coeff[(iy * 2 + idx) * 8 + ix];
            }
        }
        idct_2d_in_place(scratch, w, h);
    }

    if transpose {
        for idx in [0, 4] {
            for y in 0..8 {
                for x in 0..4 {
                    coeff[y * 8 + x + idx] = scratch[y * 4 + x + idx * 8];
                }
            }
        }
    } else {
        coeff.copy_from_slice(&scratch);
    }
}

fn transform_afv<const N: usize>(coeff: &mut SimpleGrid<f32>) {
    assert!(N < 4);
    let flip_x = N % 2;
    let flip_y = N / 2;
    let coeff = coeff.buf_mut();

    let mut coeff_afv = [0.0f32; 16];
    coeff_afv[0] = (coeff[0] + coeff[1] + coeff[8]) * 4.0;
    for (idx, v) in coeff_afv.iter_mut().enumerate().skip(1) {
        let iy = idx / 4;
        let ix = idx % 4;
        *v = coeff[2 * iy * 8 + 2 * ix];
    }

    let mut samples_afv = [0.0f32; 16];
    for (sample, basis) in samples_afv.iter_mut().zip(AFV_BASIS) {
        *sample = coeff_afv
            .into_iter()
            .zip(basis)
            .map(|(coeff, basis)| coeff * basis)
            .sum();
    }

    let mut scratch_4x4 = [0.0f32; 16];
    let mut scratch_4x8 = [0.0f32; 32];

    scratch_4x4[0] = coeff[0] - coeff[1] + coeff[8];
    for iy in 0..4 {
        for ix in 0..4 {
            if ix | iy == 0 {
                continue;
            }
            scratch_4x4[iy * 4 + ix] = coeff[2 * iy * 8 + 2 * ix + 1];
        }
    }
    idct_2d_in_place(&mut scratch_4x4, 4, 4);

    scratch_4x8[0] = coeff[0] - coeff[8];
    for iy in 0..4 {
        for ix in 0..8 {
            if ix | iy == 0 {
                continue;
            }
            scratch_4x8[iy * 8 + ix] = coeff[(2 * iy + 1) * 8 + ix];
        }
    }
    idct_2d_in_place(&mut scratch_4x8, 8, 4);

    for iy in 0..4 {
        let afv_y = if flip_y == 0 { iy } else { 3 - iy };
        for ix in 0..4 {
            let afv_x = if flip_x == 0 { ix } else { 3 - ix };
            coeff[(flip_y * 4 + iy) * 8 + flip_x * 4 + ix] = samples_afv[afv_y * 4 + afv_x];
        }
    }

    for iy in 0..4 {
        let y = flip_y * 4 + iy;
        for ix in 0..4 {
            let x = (1 - flip_x) * 4 + ix;
            coeff[y * 8 + x] = scratch_4x4[iy * 4 + ix];
        }
    }

    for iy in 0..4 {
        let y = (1 - flip_y) * 4 + iy;
        for ix in 0..8 {
            coeff[y * 8 + ix] = scratch_4x8[iy * 8 + ix];
        }
    }
}

fn transform_dct(coeff: &mut SimpleGrid<f32>, target_width: u32, target_height: u32) {
    idct_2d_in_place(coeff.buf_mut(), target_width as usize, target_height as usize);

    if coeff.width() != target_width as usize {
        coeff.reinterpret_transposed();
    }
}

pub fn transform(coeff: &mut SimpleGrid<f32>, dct_select: TransformType) {
    use TransformType::*;

    match dct_select {
        Dct2 => transform_dct2(coeff),
        Dct4 => transform_dct4(coeff),
        Hornuss => transform_hornuss(coeff),
        Dct4x8 => transform_dct4x8(coeff, false),
        Dct8x4 => transform_dct4x8(coeff, true),
        Afv0 => transform_afv::<0>(coeff),
        Afv1 => transform_afv::<1>(coeff),
        Afv2 => transform_afv::<2>(coeff),
        Afv3 => transform_afv::<3>(coeff),
        _ => {
            let (w8, h8) = dct_select.dct_select_size();
            transform_dct(coeff, w8 * 8, h8 * 8)
        },
    }
}

const AFV_BASIS: [[f32; 16]; 16] = [
    [
        0.25,
        0.87690294,
        0.0,
        0.0,
        0.0,
        -0.41053775,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        0.25,
        0.2206518,
        0.0,
        0.0,
        -0.70710677,
        0.6235485,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        0.25,
        -0.1014005,
        0.40670076,
        -0.21255748,
        0.0,
        -0.06435072,
        -0.45175567,
        -0.30468476,
        0.30179295,
        0.4082483,
        0.1747867,
        -0.21105601,
        -0.14266084,
        -0.1381354,
        -0.17437603,
        0.11354987,
    ],
    [
        0.25,
        -0.1014005,
        0.44444817,
        0.3085497,
        0.0,
        -0.06435072,
        0.15854503,
        0.51126164,
        0.25792363,
        0.0,
        0.08126112,
        0.1856718,
        -0.34164467,
        0.33022827,
        0.07027907,
        -0.074175045,
    ],
    [
        0.25,
        0.2206518,
        0.0,
        0.0,
        std::f32::consts::FRAC_1_SQRT_2,
        0.6235485,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        0.25,
        -0.1014005,
        0.0,
        0.47067022,
        0.0,
        -0.06435072,
        -0.040385153,
        0.0,
        0.1627234,
        0.0,
        0.0,
        0.0,
        0.73674977,
        0.08755115,
        -0.29210266,
        0.19402893,
    ],
    [
        0.25,
        -0.1014005,
        0.195744,
        -0.16212052,
        0.0,
        -0.06435072,
        0.0074182265,
        -0.29048014,
        0.095200226,
        0.0,
        -0.3675398,
        0.4921586,
        0.24627107,
        -0.079467066,
        0.36238173,
        -0.4351905,
    ],
    [
        0.25,
        -0.1014005,
        0.29291,
        0.0,
        0.0,
        -0.06435072,
        0.39351034,
        -0.06578702,
        0.0,
        -0.4082483,
        -0.30788222,
        -0.38525015,
        -0.08574019,
        -0.46133748,
        0.0,
        0.21918684,
    ],
    [
        0.25,
        -0.1014005,
        -0.40670076,
        -0.21255748,
        0.0,
        -0.06435072,
        -0.45175567,
        0.30468476,
        0.30179295,
        -0.4082483,
        -0.1747867,
        0.21105601,
        -0.14266084,
        -0.1381354,
        -0.17437603,
        0.11354987,
    ],
    [
        0.25,
        -0.1014005,
        -0.195744,
        -0.16212052,
        0.0,
        -0.06435072,
        0.0074182265,
        0.29048014,
        0.095200226,
        0.0,
        0.3675398,
        -0.4921586,
        0.24627107,
        -0.079467066,
        0.36238173,
        -0.4351905,
    ],
    [
        0.25,
        -0.1014005,
        0.0,
        -0.47067022,
        0.0,
        -0.06435072,
        0.11074166,
        0.0,
        -0.1627234,
        0.0,
        0.0,
        0.0,
        0.14883399,
        0.49724647,
        0.29210266,
        0.55504435,
    ],
    [
        0.25,
        -0.1014005,
        0.11379074,
        -0.14642918,
        0.0,
        -0.06435072,
        0.08298163,
        -0.23889774,
        -0.35312384,
        -0.4082483,
        0.4826689,
        0.17419413,
        -0.047686804,
        0.12538059,
        -0.4326608,
        -0.25468278,
    ],
    [
        0.25,
        -0.1014005,
        -0.44444817,
        0.3085497,
        0.0,
        -0.06435072,
        0.15854503,
        -0.51126164,
        0.25792363,
        0.0,
        -0.08126112,
        -0.1856718,
        -0.34164467,
        0.33022827,
        0.07027907,
        -0.074175045,
    ],
    [
        0.25,
        -0.1014005,
        -0.29291,
        0.0,
        0.0,
        -0.06435072,
        0.39351034,
        0.06578702,
        0.0,
        0.4082483,
        0.30788222,
        0.38525015,
        -0.08574019,
        -0.46133748,
        0.0,
        0.21918684,
    ],
    [
        0.25,
        -0.1014005,
        -0.11379074,
        -0.14642918,
        0.0,
        -0.06435072,
        0.08298163,
        0.23889774,
        -0.35312384,
        0.4082483,
        -0.4826689,
        -0.17419413,
        -0.047686804,
        0.12538059,
        -0.4326608,
        -0.25468278,
    ],
    [
        0.25,
        -0.1014005,
        0.0,
        0.42511496,
        0.0,
        -0.06435072,
        -0.45175567,
        0.0,
        -0.6035859,
        0.0,
        0.0,
        0.0,
        -0.14266084,
        -0.1381354,
        0.34875205,
        0.11354987,
    ],
];
