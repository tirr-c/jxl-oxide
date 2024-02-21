use std::arch::is_x86_feature_detected;
use std::arch::x86_64::*;

use jxl_grid::{CutGrid, SharedSubgrid};
use jxl_modular::ChannelShift;
use jxl_vardct::{BlockInfo, TransformType};

use crate::vardct::{
    dct_common::{DctDirection, scale_f},
    VarblockInfo,
};

use super::generic;

fn transform_dct4_x86_64_sse2(coeff: &mut CutGrid<'_>) {
    generic::aux_idct2_in_place_2(coeff);

    unsafe {
        let mut scratch_0 = [_mm_setzero_ps(); 4];
        let mut scratch_1 = [_mm_setzero_ps(); 4];
        for y2 in 0..4 {
            let row_ptr = coeff.get_row(y2 * 2).as_ptr();
            let a = _mm_loadu_ps(row_ptr);
            let b = _mm_loadu_ps(row_ptr.add(4));
            scratch_0[y2] = _mm_shuffle_ps::<0b10001000>(a, b);
            scratch_1[y2] = _mm_shuffle_ps::<0b11011101>(a, b);
        }

        super::dct::transpose_lane(&mut scratch_0);
        super::dct::transpose_lane(&mut scratch_1);
        let mut scratch_0 = super::dct::dct4_inverse(scratch_0);
        let mut scratch_1 = super::dct::dct4_inverse(scratch_1);
        for y2 in 0..4 {
            let row_ptr = coeff.get_row_mut(y2).as_mut_ptr();
            _mm_storeu_ps(row_ptr, super::dct::dct4_vec_inverse(scratch_0[y2]));
            _mm_storeu_ps(row_ptr.add(4), super::dct::dct4_vec_inverse(scratch_1[y2]));

            let row_ptr = coeff.get_row(y2 * 2 + 1).as_ptr();
            let a = _mm_loadu_ps(row_ptr);
            let b = _mm_loadu_ps(row_ptr.add(4));
            scratch_0[y2] = _mm_shuffle_ps::<0b10001000>(a, b);
            scratch_1[y2] = _mm_shuffle_ps::<0b11011101>(a, b);
        }

        super::dct::transpose_lane(&mut scratch_0);
        super::dct::transpose_lane(&mut scratch_1);
        let scratch_0 = super::dct::dct4_inverse(scratch_0);
        let scratch_1 = super::dct::dct4_inverse(scratch_1);
        for y in 0..4 {
            let row_ptr = coeff.get_row_mut(y + 4).as_mut_ptr();
            _mm_storeu_ps(row_ptr, super::dct::dct4_vec_inverse(scratch_0[y]));
            _mm_storeu_ps(row_ptr.add(4), super::dct::dct4_vec_inverse(scratch_1[y]));
        }
    }
}

fn transform_dct4x8_x86_64_sse2<const TR: bool>(coeff: &mut CutGrid<'_>) {
    let coeff0 = coeff.get(0, 0);
    let coeff1 = coeff.get(0, 1);
    *coeff.get_mut(0, 0) = coeff0 + coeff1;
    *coeff.get_mut(0, 1) = coeff0 - coeff1;

    unsafe {
        if TR {
            let mut scratch_0 = [_mm_setzero_ps(); 4];
            let mut scratch_1 = [_mm_setzero_ps(); 4];
            for y2 in 0..4 {
                let row_ptr = coeff.get_row(y2 * 2).as_ptr();
                let a = _mm_loadu_ps(row_ptr);
                let b = _mm_loadu_ps(row_ptr.add(4));
                let (l, r) = super::dct::dct8_vec_inverse(a, b);
                scratch_0[y2] = l;
                scratch_1[y2] = r;
            }

            let mut scratch_0 = super::dct::dct4_inverse(scratch_0);
            let mut scratch_1 = super::dct::dct4_inverse(scratch_1);
            super::dct::transpose_lane(&mut scratch_0);
            super::dct::transpose_lane(&mut scratch_1);
            for y2 in 0..4 {
                let y = [1, 5, 3, 7][y2];
                let row_ptr = coeff.get_row(y).as_ptr();
                let a = _mm_loadu_ps(row_ptr);
                let b = _mm_loadu_ps(row_ptr.add(4));
                let (l, r) = super::dct::dct8_vec_inverse(a, b);

                let row_ptr = coeff.get_row_mut(y2).as_mut_ptr();
                _mm_storeu_ps(row_ptr, scratch_0[y2]);
                let row_ptr = coeff.get_row_mut(y2 + 4).as_mut_ptr();
                _mm_storeu_ps(row_ptr, scratch_1[y2]);

                scratch_0[y2] = l;
                scratch_1[y2] = r;
            }
            scratch_0.swap(1, 2);
            scratch_1.swap(1, 2);

            let mut scratch_0 = super::dct::dct4_inverse(scratch_0);
            let mut scratch_1 = super::dct::dct4_inverse(scratch_1);
            super::dct::transpose_lane(&mut scratch_0);
            super::dct::transpose_lane(&mut scratch_1);
            for y in 0..4 {
                let row_ptr = coeff.get_row_mut(y).as_mut_ptr().add(4);
                _mm_storeu_ps(row_ptr, scratch_0[y]);
                let row_ptr = coeff.get_row_mut(y + 4).as_mut_ptr().add(4);
                _mm_storeu_ps(row_ptr, scratch_1[y]);
            }
        } else {
            let mut scratch_0 = [_mm_setzero_ps(); 4];
            let mut scratch_1 = [_mm_setzero_ps(); 4];
            for y2 in 0..4 {
                let row_ptr = coeff.get_row(y2 * 2).as_ptr();
                let a = _mm_loadu_ps(row_ptr);
                let b = _mm_loadu_ps(row_ptr.add(4));
                let (l, r) = super::dct::dct8_vec_inverse(a, b);
                scratch_0[y2] = l;
                scratch_1[y2] = r;
            }

            let mut scratch_0 = super::dct::dct4_inverse(scratch_0);
            let mut scratch_1 = super::dct::dct4_inverse(scratch_1);
            for y2 in 0..4 {
                let row_ptr = coeff.get_row_mut(y2).as_mut_ptr();
                _mm_storeu_ps(row_ptr, scratch_0[y2]);
                _mm_storeu_ps(row_ptr.add(4), scratch_1[y2]);

                let row_ptr = coeff.get_row(y2 * 2 + 1).as_ptr();
                let a = _mm_loadu_ps(row_ptr);
                let b = _mm_loadu_ps(row_ptr.add(4));
                let (l, r) = super::dct::dct8_vec_inverse(a, b);
                scratch_0[y2] = l;
                scratch_1[y2] = r;
            }

            let scratch_0 = super::dct::dct4_inverse(scratch_0);
            let scratch_1 = super::dct::dct4_inverse(scratch_1);
            for y in 0..4 {
                let row_ptr = coeff.get_row_mut(y + 4).as_mut_ptr();
                _mm_storeu_ps(row_ptr, scratch_0[y]);
                _mm_storeu_ps(row_ptr.add(4), scratch_1[y]);
            }
        }
    }
}

fn transform_dct(coeff: &mut CutGrid<'_>) {
    super::dct::dct_2d_x86_64_sse2(coeff, DctDirection::Inverse);
}

pub fn transform(coeff: &mut CutGrid<'_>, dct_select: TransformType) {
    use TransformType::*;

    match dct_select {
        Dct2 => generic::transform_dct2(coeff),
        Dct4 => transform_dct4_x86_64_sse2(coeff),
        Hornuss => generic::transform_hornuss(coeff),
        Dct4x8 => transform_dct4x8_x86_64_sse2::<false>(coeff),
        Dct8x4 => transform_dct4x8_x86_64_sse2::<true>(coeff),
        Afv0 => generic::transform_afv::<0>(coeff),
        Afv1 => generic::transform_afv::<1>(coeff),
        Afv2 => generic::transform_afv::<2>(coeff),
        Afv3 => generic::transform_afv::<3>(coeff),
        _ => transform_dct(coeff),
    }
}

#[inline]
fn transform_varblocks_x86_64_sse2(
    lf: &[SharedSubgrid<f32>; 3],
    coeff_out: &mut [CutGrid<'_, f32>; 3],
    shifts_cbycr: [ChannelShift; 3],
    block_info: &SharedSubgrid<BlockInfo>,
) {
    use TransformType::*;

    for (channel, (coeff, lf)) in coeff_out.iter_mut().zip(lf).enumerate() {
        let shift = shifts_cbycr[channel];
        crate::vardct::for_each_varblocks(
            block_info,
            shift,
            |VarblockInfo {
                 shifted_bx,
                 shifted_by,
                 dct_select,
                 ..
             }| {
                let (bw, bh) = dct_select.dct_select_size();
                let left = shifted_bx * 8;
                let top = shifted_by * 8;

                let bw = bw as usize;
                let bh = bh as usize;
                let logbw = bw.trailing_zeros() as usize;
                let logbh = bh.trailing_zeros() as usize;

                let mut out = coeff.subgrid_mut(left..(left + bw), top..(top + bh));
                if matches!(
                    dct_select,
                    Hornuss | Dct2 | Dct4 | Dct8x4 | Dct4x8 | Dct8 | Afv0 | Afv1 | Afv2 | Afv3
                ) {
                    debug_assert_eq!(bw * bh, 1);
                    *out.get_mut(0, 0) = *lf.get(shifted_bx, shifted_by);
                } else {
                    for y in 0..bh {
                        for x in 0..bw {
                            *out.get_mut(x, y) = *lf.get(shifted_bx + x, shifted_by + y);
                        }
                    }
                    super::dct::dct_2d_x86_64_sse2(&mut out, DctDirection::Forward);
                    for y in 0..bh {
                        for x in 0..bw {
                            *out.get_mut(x, y) /= scale_f(y, 5 - logbh) * scale_f(x, 5 - logbw);
                        }
                    }
                }

                let mut block = coeff.subgrid_mut(left..(left + bw * 8), top..(top + bh * 8));
                transform(&mut block, dct_select);
            },
        );
    }
}

#[target_feature(enable = "avx2")]
unsafe fn transform_varblocks_x86_64_avx2(
    lf: &[SharedSubgrid<f32>; 3],
    coeff_out: &mut [CutGrid<'_, f32>; 3],
    shifts_cbycr: [ChannelShift; 3],
    block_info: &SharedSubgrid<BlockInfo>,
) {
    transform_varblocks_x86_64_sse2(lf, coeff_out, shifts_cbycr, block_info);
}

pub fn transform_varblocks(
    lf: &[SharedSubgrid<f32>; 3],
    coeff_out: &mut [CutGrid<'_, f32>; 3],
    shifts_cbycr: [ChannelShift; 3],
    block_info: &SharedSubgrid<BlockInfo>,
) {
    if is_x86_feature_detected!("avx2") {
        unsafe {
            return transform_varblocks_x86_64_avx2(lf, coeff_out, shifts_cbycr, block_info);
        }
    }

    transform_varblocks_x86_64_sse2(lf, coeff_out, shifts_cbycr, block_info);
}
