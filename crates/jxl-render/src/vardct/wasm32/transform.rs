use std::arch::wasm32::*;

use jxl_grid::{MutableSubgrid, SharedSubgrid, SimdVector};
use jxl_modular::ChannelShift;
use jxl_vardct::{BlockInfo, TransformType};

use crate::vardct::{dct_common::DctDirection, transform_common::transform_varblocks_inner};

use super::generic;

unsafe fn transform_dct4_wasm32_simd128(coeff: &mut MutableSubgrid<'_>) {
    generic::aux_idct2_in_place_2(coeff);

    let mut scratch_0 = [v128::zero(); 4];
    let mut scratch_1 = [v128::zero(); 4];
    for y2 in 0..4 {
        let row_ptr = coeff.get_row(y2 * 2).as_ptr();
        let l = v128::load(row_ptr);
        let r = v128::load(row_ptr.add(4));
        scratch_0[y2] = i32x4_shuffle::<0, 2, 4, 6>(l, r);
        scratch_1[y2] = i32x4_shuffle::<1, 3, 5, 7>(l, r);
    }

    let scratch_0 = super::dct::transpose_lane(&scratch_0);
    let mut scratch_0 = super::dct::dct4_inverse(scratch_0);
    let scratch_1 = super::dct::transpose_lane(&scratch_1);
    let mut scratch_1 = super::dct::dct4_inverse(scratch_1);
    for y2 in 0..4 {
        let row_ptr = coeff.get_row_mut(y2).as_mut_ptr();
        super::dct::dct4_vec_inverse(scratch_0[y2]).store(row_ptr);
        super::dct::dct4_vec_inverse(scratch_1[y2]).store(row_ptr.add(4));

        let row_ptr = coeff.get_row(y2 * 2 + 1).as_ptr();
        let l = v128::load(row_ptr);
        let r = v128::load(row_ptr.add(4));
        scratch_0[y2] = i32x4_shuffle::<0, 2, 4, 6>(l, r);
        scratch_1[y2] = i32x4_shuffle::<1, 3, 5, 7>(l, r);
    }

    let scratch_0 = super::dct::transpose_lane(&scratch_0);
    let scratch_0 = super::dct::dct4_inverse(scratch_0);
    let scratch_1 = super::dct::transpose_lane(&scratch_1);
    let scratch_1 = super::dct::dct4_inverse(scratch_1);
    for y in 0..4 {
        let row_ptr = coeff.get_row_mut(y + 4).as_mut_ptr();
        super::dct::dct4_vec_inverse(scratch_0[y]).store(row_ptr);
        super::dct::dct4_vec_inverse(scratch_1[y]).store(row_ptr.add(4));
    }
}

unsafe fn transform_dct4x8_wasm32_simd128<const TR: bool>(coeff: &mut MutableSubgrid<'_>) {
    let coeff0 = coeff.get(0, 0);
    let coeff1 = coeff.get(0, 1);
    *coeff.get_mut(0, 0) = coeff0 + coeff1;
    *coeff.get_mut(0, 1) = coeff0 - coeff1;

    if TR {
        let mut scratch_0 = [v128::zero(); 4];
        let mut scratch_1 = [v128::zero(); 4];
        for y2 in 0..4 {
            let row_ptr = coeff.get_row(y2 * 2).as_ptr();
            let a = v128::load(row_ptr);
            let b = v128::load(row_ptr.add(4));
            let (l, r) = super::dct::dct8_vec_inverse(a, b);
            scratch_0[y2] = l;
            scratch_1[y2] = r;
        }

        let scratch_0 = super::dct::dct4_inverse(scratch_0);
        let scratch_1 = super::dct::dct4_inverse(scratch_1);
        let mut scratch_0 = super::dct::transpose_lane(&scratch_0);
        let mut scratch_1 = super::dct::transpose_lane(&scratch_1);
        for y2 in 0..4 {
            let y = [1, 5, 3, 7][y2];
            let row_ptr = coeff.get_row(y).as_ptr();
            let a = v128::load(row_ptr);
            let b = v128::load(row_ptr.add(4));
            let (l, r) = super::dct::dct8_vec_inverse(a, b);

            let row_ptr = coeff.get_row_mut(y2).as_mut_ptr();
            scratch_0[y2].store(row_ptr);
            let row_ptr = coeff.get_row_mut(y2 + 4).as_mut_ptr();
            scratch_1[y2].store(row_ptr);

            scratch_0[y2] = l;
            scratch_1[y2] = r;
        }
        scratch_0.swap(1, 2);
        scratch_1.swap(1, 2);

        let scratch_0 = super::dct::dct4_inverse(scratch_0);
        let scratch_1 = super::dct::dct4_inverse(scratch_1);
        let scratch_0 = super::dct::transpose_lane(&scratch_0);
        let scratch_1 = super::dct::transpose_lane(&scratch_1);
        for y in 0..4 {
            let row_ptr = coeff.get_row_mut(y).as_mut_ptr().add(4);
            scratch_0[y].store(row_ptr);
            let row_ptr = coeff.get_row_mut(y + 4).as_mut_ptr().add(4);
            scratch_1[y].store(row_ptr);
        }
    } else {
        let mut scratch_0 = [v128::zero(); 4];
        let mut scratch_1 = [v128::zero(); 4];
        for y2 in 0..4 {
            let row_ptr = coeff.get_row(y2 * 2).as_ptr();
            let a = v128::load(row_ptr);
            let b = v128::load(row_ptr.add(4));
            let (l, r) = super::dct::dct8_vec_inverse(a, b);
            scratch_0[y2] = l;
            scratch_1[y2] = r;
        }

        let mut scratch_0 = super::dct::dct4_inverse(scratch_0);
        let mut scratch_1 = super::dct::dct4_inverse(scratch_1);
        for y2 in 0..4 {
            let row_ptr = coeff.get_row_mut(y2).as_mut_ptr();
            scratch_0[y2].store(row_ptr);
            scratch_1[y2].store(row_ptr.add(4));

            let row_ptr = coeff.get_row(y2 * 2 + 1).as_ptr();
            let a = v128::load(row_ptr);
            let b = v128::load(row_ptr.add(4));
            let (l, r) = super::dct::dct8_vec_inverse(a, b);
            scratch_0[y2] = l;
            scratch_1[y2] = r;
        }

        let scratch_0 = super::dct::dct4_inverse(scratch_0);
        let scratch_1 = super::dct::dct4_inverse(scratch_1);
        for y in 0..4 {
            let row_ptr = coeff.get_row_mut(y + 4).as_mut_ptr();
            scratch_0[y].store(row_ptr);
            scratch_1[y].store(row_ptr.add(4));
        }
    }
}

unsafe fn transform_dct_wasm32_simd128(coeff: &mut MutableSubgrid<'_>) {
    super::dct::dct_2d_wasm32_simd128(coeff, DctDirection::Inverse);
}

unsafe fn transform_wasm32_simd128(coeff: &mut MutableSubgrid<'_>, dct_select: TransformType) {
    use TransformType::*;

    match dct_select {
        Dct2 => generic::transform_dct2(coeff),
        Dct4 => transform_dct4_wasm32_simd128(coeff),
        Hornuss => generic::transform_hornuss(coeff),
        Dct4x8 => transform_dct4x8_wasm32_simd128::<false>(coeff),
        Dct8x4 => transform_dct4x8_wasm32_simd128::<true>(coeff),
        Afv0 => generic::transform_afv::<0>(coeff),
        Afv1 => generic::transform_afv::<1>(coeff),
        Afv2 => generic::transform_afv::<2>(coeff),
        Afv3 => generic::transform_afv::<3>(coeff),
        _ => transform_dct_wasm32_simd128(coeff),
    }
}

unsafe fn transform_varblocks_wasm32_simd128(
    lf: &[SharedSubgrid<f32>; 3],
    coeff_out: &mut [MutableSubgrid<'_, f32>; 3],
    shifts_cbycr: [ChannelShift; 3],
    block_info: &SharedSubgrid<BlockInfo>,
) {
    transform_varblocks_inner(
        lf,
        coeff_out,
        shifts_cbycr,
        block_info,
        super::dct::dct_2d_wasm32_simd128,
        transform_wasm32_simd128,
    );
}

#[inline]
pub fn transform_varblocks(
    lf: &[SharedSubgrid<f32>; 3],
    coeff_out: &mut [MutableSubgrid<'_, f32>; 3],
    shifts_cbycr: [ChannelShift; 3],
    block_info: &SharedSubgrid<BlockInfo>,
) {
    unsafe { transform_varblocks_wasm32_simd128(lf, coeff_out, shifts_cbycr, block_info) }
}
