#![allow(dead_code)]
use jxl_grid::{MutableSubgrid, SharedSubgrid};
use jxl_modular::ChannelShift;
use jxl_vardct::{BlockInfo, TransformType};

use crate::vardct::{
    dct_common::DctDirection,
    transform_common::{transform_varblocks_inner, AFV_BASIS},
};

use super::dct_2d;

#[inline(always)]
pub(crate) fn aux_idct2_in_place_2(block: &mut MutableSubgrid<'_>) {
    let c00 = block.get(0, 0);
    let c01 = block.get(1, 0);
    let c10 = block.get(0, 1);
    let c11 = block.get(1, 1);
    *block.get_mut(0, 0) = c00 + c01 + c10 + c11;
    *block.get_mut(1, 0) = c00 + c01 - c10 - c11;
    *block.get_mut(0, 1) = c00 - c01 + c10 - c11;
    *block.get_mut(1, 1) = c00 - c01 - c10 + c11;
}

#[inline(always)]
pub(crate) fn aux_idct2_in_place<const SIZE: usize>(block: &mut MutableSubgrid<'_>) {
    debug_assert!(SIZE.is_power_of_two());

    let num_2x2 = SIZE / 2;
    let mut scratch = [[0.0f32; SIZE]; SIZE];
    for y in 0..num_2x2 {
        for x in 0..num_2x2 {
            let c00 = block.get(x, y);
            let c01 = block.get(x + num_2x2, y);
            let c10 = block.get(x, y + num_2x2);
            let c11 = block.get(x + num_2x2, y + num_2x2);

            scratch[2 * y][2 * x] = c00 + c01 + c10 + c11;
            scratch[2 * y][2 * x + 1] = c00 + c01 - c10 - c11;
            scratch[2 * y + 1][2 * x] = c00 - c01 + c10 - c11;
            scratch[2 * y + 1][2 * x + 1] = c00 - c01 - c10 + c11;
        }
    }

    for (y, scratch_row) in scratch.into_iter().enumerate() {
        block.get_row_mut(y)[..SIZE].copy_from_slice(&scratch_row);
    }
}

pub(crate) fn transform_dct2(coeff: &mut MutableSubgrid<'_>) {
    aux_idct2_in_place::<2>(coeff);
    aux_idct2_in_place::<4>(coeff);
    aux_idct2_in_place::<8>(coeff);
}

pub(crate) fn transform_dct4(coeff: &mut MutableSubgrid<'_>) {
    aux_idct2_in_place::<2>(coeff);

    let mut scratch = [0.0f32; 64];
    for y in 0..2 {
        for x in 0..2 {
            let mut scratch = MutableSubgrid::from_buf(&mut scratch[(y * 2 + x) * 16..], 4, 4, 4);
            for iy in 0..4 {
                for ix in 0..4 {
                    *scratch.get_mut(iy, ix) = coeff.get(x + ix * 2, y + iy * 2);
                }
            }
            dct_2d(&mut scratch, DctDirection::Inverse);
        }
    }

    for y in 0..2 {
        for x in 0..2 {
            let scratch = &scratch[(y * 2 + x) * 16..][..16];
            for iy in 0..4 {
                for ix in 0..4 {
                    *coeff.get_mut(x * 4 + ix, y * 4 + iy) = scratch[iy * 4 + ix];
                }
            }
        }
    }
}

pub(crate) fn transform_hornuss(coeff: &mut MutableSubgrid<'_>) {
    aux_idct2_in_place::<2>(coeff);

    let mut scratch = [0.0f32; 64];
    for y in 0..2 {
        for x in 0..2 {
            let scratch = &mut scratch[(y * 2 + x) * 16..][..16];
            for iy in 0..4 {
                for ix in 0..4 {
                    scratch[iy * 4 + ix] = coeff.get(x + ix * 2, y + iy * 2);
                }
            }
            let residual_sum: f32 = scratch[1..].iter().copied().sum();
            let avg = scratch[0] - residual_sum / 16.0;
            scratch[0] = scratch[5];
            scratch[5] = 0.0;
            for s in scratch.iter_mut() {
                *s += avg;
            }
        }
    }

    for y in 0..2 {
        for x in 0..2 {
            let scratch = &mut scratch[(y * 2 + x) * 16..][..16];
            for iy in 0..4 {
                for ix in 0..4 {
                    *coeff.get_mut(x * 4 + ix, y * 4 + iy) = scratch[iy * 4 + ix];
                }
            }
        }
    }
}

pub(crate) fn transform_dct4x8<const TR: bool>(coeff: &mut MutableSubgrid<'_>) {
    let coeff0 = coeff.get(0, 0);
    let coeff1 = coeff.get(0, 1);
    *coeff.get_mut(0, 0) = coeff0 + coeff1;
    *coeff.get_mut(0, 1) = coeff0 - coeff1;

    let mut scratch = [0.0f32; 64];
    for idx in [0, 1] {
        let mut scratch = MutableSubgrid::from_buf(&mut scratch[(idx * 32)..], 8, 4, 8);
        for iy in 0..4 {
            for ix in 0..8 {
                *scratch.get_mut(ix, iy) = coeff.get(ix, iy * 2 + idx);
            }
        }
        dct_2d(&mut scratch, DctDirection::Inverse);
    }

    if TR {
        for y in 0..8 {
            for x in 0..8 {
                *coeff.get_mut(y, x) = scratch[y * 8 + x];
            }
        }
    } else {
        for y in 0..8 {
            coeff.get_row_mut(y)[..8].copy_from_slice(&scratch[y * 8..][..8]);
        }
    }
}

pub(crate) fn transform_afv<const N: usize>(coeff: &mut MutableSubgrid<'_>) {
    assert!(N < 4);
    let flip_x = N % 2;
    let flip_y = N / 2;

    let mut coeff_afv = [0.0f32; 16];
    coeff_afv[0] = (coeff.get(0, 0) + coeff.get(1, 0) + coeff.get(0, 1)) * 4.0;
    for (idx, v) in coeff_afv.iter_mut().enumerate().skip(1) {
        let iy = idx / 4;
        let ix = idx % 4;
        *v = coeff.get(2 * ix, 2 * iy);
    }

    let mut samples_afv = [0.0f32; 16];
    for (coeff, basis) in coeff_afv.into_iter().zip(AFV_BASIS) {
        for (sample, basis) in samples_afv.iter_mut().zip(basis) {
            *sample = coeff.mul_add(basis, *sample);
        }
    }

    let mut scratch_4x4 = [0.0f32; 16];
    let mut scratch_4x8 = [0.0f32; 32];

    scratch_4x4[0] = coeff.get(0, 0) - coeff.get(1, 0) + coeff.get(0, 1);
    for iy in 0..4 {
        for ix in 0..4 {
            if ix | iy == 0 {
                continue;
            }
            scratch_4x4[ix * 4 + iy] = coeff.get(2 * ix + 1, 2 * iy);
        }
    }
    dct_2d(
        &mut MutableSubgrid::from_buf(&mut scratch_4x4, 4, 4, 4),
        DctDirection::Inverse,
    );

    scratch_4x8[0] = coeff.get(0, 0) - coeff.get(0, 1);
    for iy in 0..4 {
        for ix in 0..8 {
            if ix | iy == 0 {
                continue;
            }
            scratch_4x8[iy * 8 + ix] = coeff.get(ix, 2 * iy + 1);
        }
    }
    dct_2d(
        &mut MutableSubgrid::from_buf(&mut scratch_4x8, 8, 4, 8),
        DctDirection::Inverse,
    );

    for iy in 0..4 {
        let afv_y = if flip_y == 0 { iy } else { 3 - iy };
        for ix in 0..4 {
            let afv_x = if flip_x == 0 { ix } else { 3 - ix };
            *coeff.get_mut(flip_x * 4 + ix, flip_y * 4 + iy) = samples_afv[afv_y * 4 + afv_x];
        }
    }

    for iy in 0..4 {
        let y = flip_y * 4 + iy;
        for ix in 0..4 {
            let x = (1 - flip_x) * 4 + ix;
            *coeff.get_mut(x, y) = scratch_4x4[iy * 4 + ix];
        }
    }

    for iy in 0..4 {
        let y = (1 - flip_y) * 4 + iy;
        coeff.get_row_mut(y)[..8].copy_from_slice(&scratch_4x8[iy * 8..][..8]);
    }
}

fn transform_dct(coeff: &mut MutableSubgrid<'_>) {
    dct_2d(coeff, DctDirection::Inverse);
}

fn transform(coeff: &mut MutableSubgrid<'_>, dct_select: TransformType) {
    use TransformType::*;

    match dct_select {
        Dct2 => transform_dct2(coeff),
        Dct4 => transform_dct4(coeff),
        Hornuss => transform_hornuss(coeff),
        Dct4x8 => transform_dct4x8::<false>(coeff),
        Dct8x4 => transform_dct4x8::<true>(coeff),
        Afv0 => transform_afv::<0>(coeff),
        Afv1 => transform_afv::<1>(coeff),
        Afv2 => transform_afv::<2>(coeff),
        Afv3 => transform_afv::<3>(coeff),
        _ => transform_dct(coeff),
    }
}

pub fn transform_varblocks(
    lf: &[SharedSubgrid<f32>; 3],
    coeff_out: &mut [MutableSubgrid<'_, f32>; 3],
    shifts_cbycr: [ChannelShift; 3],
    block_info: &SharedSubgrid<BlockInfo>,
) {
    unsafe {
        transform_varblocks_inner(
            lf,
            coeff_out,
            shifts_cbycr,
            block_info,
            super::dct::dct_2d,
            transform,
        );
    }
}
