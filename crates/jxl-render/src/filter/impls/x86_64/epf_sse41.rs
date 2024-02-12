use std::arch::x86_64::*;

use jxl_grid::SimdVector;

use crate::filter::epf::*;

type Vector = __m128;

#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn weight_sse41(scaled_distance: Vector, sigma: f32, step_multiplier: Vector) -> Vector {
    let neg_inv_sigma = Vector::splat_f32(6.6 * (std::f32::consts::FRAC_1_SQRT_2 - 1.0) / sigma)
        .mul(step_multiplier);
    let result = scaled_distance.mul(neg_inv_sigma).add(_mm_set1_ps(1.0));
    _mm_max_ps(result, Vector::zero())
}

#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn epf_row_x86_64_sse41<const STEP: usize>(epf_row: EpfRow) {
    let EpfRow {
        merged_input_rows,
        output_rows,
        width,
        x,
        y,
        sigma_row,
        epf_params,
        ..
    } = epf_row;
    let iwidth = width as isize;
    let merged_input_rows = merged_input_rows.unwrap();
    let kernel_offsets = epf_kernel_offsets::<STEP>();

    let step_multiplier = if STEP == 0 {
        epf_params.sigma.pass0_sigma_scale
    } else if STEP == 2 {
        epf_params.sigma.pass2_sigma_scale
    } else {
        1.0
    };
    let border_sad_mul = epf_params.sigma.border_sad_mul;
    let channel_scale = epf_params.channel_scale;

    let padding = 3 - STEP;
    if width < padding * 2 {
        return;
    }

    let simd_range = {
        let start = (x + padding + 7) & !7;
        let end = (x + width - padding) & !7;
        if start > end {
            let start = start - x;
            start..start
        } else {
            let start = start - x;
            let end = end - x;
            start..end
        }
    };

    let is_y_border = (y + 1) & 0b110 == 0;
    let sm = if is_y_border {
        let sm = _mm_set1_ps(step_multiplier * border_sad_mul);
        [sm, sm]
    } else {
        [
            _mm_set_ps(
                step_multiplier,
                step_multiplier,
                step_multiplier,
                step_multiplier * border_sad_mul,
            ),
            _mm_set_ps(
                step_multiplier * border_sad_mul,
                step_multiplier,
                step_multiplier,
                step_multiplier,
            ),
        ]
    };

    for dx in simd_range.step_by(4) {
        let sigma_x = x + dx;
        let sm = sm[(sigma_x / 4) & 1];
        let sigma_x = sigma_x / 8 - x / 8;

        let input_base_idx = 3 * width + dx;
        let sigma_val = sigma_row[sigma_x];

        let originals: [_; 3] = std::array::from_fn(|c| unsafe {
            _mm_loadu_ps(merged_input_rows[c].as_ptr().add(input_base_idx))
        });

        if sigma_val < 0.3 {
            for (c, val) in originals.into_iter().enumerate() {
                _mm_storeu_ps(output_rows[c].as_mut_ptr().add(dx), val);
            }
            continue;
        }

        let mut sum_weights = _mm_set1_ps(1.0);
        let mut sum_channels = originals;

        if STEP == 1 {
            // (0, -1), (0, 1)
            {
                let mut dist0 = _mm_setzero_ps();
                let mut dist1 = _mm_setzero_ps();
                for c in 0..3 {
                    let scale = _mm_set1_ps(channel_scale[c]);
                    let rows_ptr = merged_input_rows[c].as_ptr();

                    let v0 = _mm_loadu_ps(rows_ptr.add(input_base_idx - 2 * width));
                    let v1 = _mm_loadu_ps(rows_ptr.add(input_base_idx - width));
                    let v2 = _mm_loadu_ps(rows_ptr.add(input_base_idx));
                    let v3 = _mm_loadu_ps(rows_ptr.add(input_base_idx + width));
                    let v4 = _mm_loadu_ps(rows_ptr.add(input_base_idx + 2 * width));
                    let tmp = v1.sub(v2).abs().add(v3.sub(v2).abs());
                    let mut acc0 = tmp.add(v1.sub(v0).abs());
                    let mut acc1 = tmp.add(v3.sub(v4).abs());

                    let v1_left = *rows_ptr.add(input_base_idx - width - 1);
                    let v2_left = *rows_ptr.add(input_base_idx - 1);
                    let v3_left = *rows_ptr.add(input_base_idx + width - 1);
                    let v1_right = *rows_ptr.add(input_base_idx - width + 4);
                    let v2_right = *rows_ptr.add(input_base_idx + 4);
                    let v3_right = *rows_ptr.add(input_base_idx + width + 4);

                    let v1_left = _mm_castsi128_ps(_mm_insert_epi32::<0>(
                        _mm_slli_si128::<4>(_mm_castps_si128(v1)),
                        v1_left.to_bits() as i32,
                    ));
                    let v2_left = _mm_castsi128_ps(_mm_insert_epi32::<0>(
                        _mm_slli_si128::<4>(_mm_castps_si128(v2)),
                        v2_left.to_bits() as i32,
                    ));
                    let v3_left = _mm_castsi128_ps(_mm_insert_epi32::<0>(
                        _mm_slli_si128::<4>(_mm_castps_si128(v3)),
                        v3_left.to_bits() as i32,
                    ));
                    acc0 = acc0.add(v1_left.sub(v2_left).abs());
                    acc1 = acc1.add(v3_left.sub(v2_left).abs());

                    let v1_right = _mm_castsi128_ps(_mm_insert_epi32::<3>(
                        _mm_srli_si128::<4>(_mm_castps_si128(v1)),
                        v1_right.to_bits() as i32,
                    ));
                    let v2_right = _mm_castsi128_ps(_mm_insert_epi32::<3>(
                        _mm_srli_si128::<4>(_mm_castps_si128(v2)),
                        v2_right.to_bits() as i32,
                    ));
                    let v3_right = _mm_castsi128_ps(_mm_insert_epi32::<3>(
                        _mm_srli_si128::<4>(_mm_castps_si128(v3)),
                        v3_right.to_bits() as i32,
                    ));
                    acc0 = acc0.add(v1_right.sub(v2_right).abs());
                    acc1 = acc1.add(v3_right.sub(v2_right).abs());

                    dist0 = scale.mul(acc0).add(dist0);
                    dist1 = scale.mul(acc1).add(dist1);
                }

                let weight0 = weight_sse41(dist0, sigma_val, sm);
                let weight1 = weight_sse41(dist1, sigma_val, sm);
                sum_weights = sum_weights.add(weight0).add(weight1);

                for (c, sum) in sum_channels.iter_mut().enumerate() {
                    let rows_ptr = merged_input_rows[c].as_ptr();
                    let weighted0 = weight0.mul(_mm_loadu_ps(rows_ptr.add(input_base_idx - width)));
                    let weighted1 = weight1.mul(_mm_loadu_ps(rows_ptr.add(input_base_idx + width)));
                    *sum = sum.add(weighted0).add(weighted1);
                }
            }

            // (-1, 0), (1, 0)
            {
                let mut dist0 = _mm_setzero_ps();
                let mut dist1 = _mm_setzero_ps();
                for c in 0..3 {
                    let scale = _mm_set1_ps(channel_scale[c]);
                    let rows_ptr = merged_input_rows[c].as_ptr();

                    let v0r = _mm_loadu_ps(rows_ptr.add(input_base_idx - width + 1));
                    let v0 = _mm_castsi128_ps(_mm_insert_epi32::<0>(
                        _mm_slli_si128::<4>(_mm_castps_si128(v0r)),
                        (*rows_ptr.add(input_base_idx - width)).to_bits() as i32,
                    ));
                    let v0l = _mm_castsi128_ps(_mm_insert_epi32::<0>(
                        _mm_slli_si128::<4>(_mm_castps_si128(v0)),
                        (*rows_ptr.add(input_base_idx - width - 1)).to_bits() as i32,
                    ));
                    let mut acc0 = v0l.sub(v0).abs();
                    let mut acc1 = v0r.sub(v0).abs();

                    let v1rr = _mm_loadu_ps(rows_ptr.add(input_base_idx + 2));
                    let v1r = _mm_castsi128_ps(_mm_insert_epi32::<0>(
                        _mm_slli_si128::<4>(_mm_castps_si128(v1rr)),
                        (*rows_ptr.add(input_base_idx + 1)).to_bits() as i32,
                    ));
                    let v1 = _mm_castsi128_ps(_mm_insert_epi32::<0>(
                        _mm_slli_si128::<4>(_mm_castps_si128(v1r)),
                        (*rows_ptr.add(input_base_idx)).to_bits() as i32,
                    ));
                    let v1l = _mm_castsi128_ps(_mm_insert_epi32::<0>(
                        _mm_slli_si128::<4>(_mm_castps_si128(v1)),
                        (*rows_ptr.add(input_base_idx - 1)).to_bits() as i32,
                    ));
                    let v1ll = _mm_castsi128_ps(_mm_insert_epi32::<0>(
                        _mm_slli_si128::<4>(_mm_castps_si128(v1l)),
                        (*rows_ptr.add(input_base_idx - 2)).to_bits() as i32,
                    ));
                    acc0 = acc0.add(v1ll.sub(v1l).abs());
                    acc0 = acc0.add(v1.sub(v1l).abs());
                    acc0 = acc0.add(v1.sub(v1r).abs());
                    acc1 = acc1.add(v1.sub(v1l).abs());
                    acc1 = acc1.add(v1.sub(v1r).abs());
                    acc1 = acc1.add(v1rr.sub(v1r).abs());

                    let v2r = _mm_loadu_ps(rows_ptr.add(input_base_idx + width + 1));
                    let v2 = _mm_castsi128_ps(_mm_insert_epi32::<0>(
                        _mm_slli_si128::<4>(_mm_castps_si128(v2r)),
                        (*rows_ptr.add(input_base_idx + width)).to_bits() as i32,
                    ));
                    let v2l = _mm_castsi128_ps(_mm_insert_epi32::<0>(
                        _mm_slli_si128::<4>(_mm_castps_si128(v2)),
                        (*rows_ptr.add(input_base_idx + width - 1)).to_bits() as i32,
                    ));
                    acc0 = acc0.add(v2l.sub(v2).abs());
                    acc1 = acc1.add(v2r.sub(v2).abs());

                    dist0 = scale.mul(acc0).add(dist0);
                    dist1 = scale.mul(acc1).add(dist1);
                }

                let weight0 = weight_sse41(dist0, sigma_val, sm);
                let weight1 = weight_sse41(dist1, sigma_val, sm);
                sum_weights = sum_weights.add(weight0).add(weight1);

                for (c, sum) in sum_channels.iter_mut().enumerate() {
                    let rows_ptr = merged_input_rows[c].as_ptr();
                    let weighted0 = weight0.mul(_mm_loadu_ps(rows_ptr.add(input_base_idx - 1)));
                    let weighted1 = weight1.mul(_mm_loadu_ps(rows_ptr.add(input_base_idx + 1)));
                    *sum = sum.add(weighted0).add(weighted1);
                }
            }
        } else {
            for &(kx, ky) in kernel_offsets {
                let input_kernel_idx = input_base_idx.wrapping_add_signed(ky * iwidth + kx);
                let mut dist = _mm_setzero_ps();
                for c in 0..3 {
                    let scale = _mm_set1_ps(channel_scale[c]);
                    let rows_ptr = merged_input_rows[c].as_ptr();
                    if STEP == 0 {
                        let vk0 = _mm_loadu_ps(rows_ptr.add(input_kernel_idx - width));
                        let vb0 = _mm_loadu_ps(rows_ptr.add(input_base_idx - width));
                        let mut acc = vk0.sub(vb0).abs();

                        let vk1r = _mm_loadu_ps(rows_ptr.add(input_kernel_idx + 1));
                        let vb1r = _mm_loadu_ps(rows_ptr.add(input_base_idx + 1));
                        acc = acc.add(vk1r.sub(vb1r).abs());

                        let vk1 = _mm_castsi128_ps(_mm_insert_epi32::<0>(
                            _mm_slli_si128::<4>(_mm_castps_si128(vk1r)),
                            (*rows_ptr.add(input_kernel_idx)).to_bits() as i32,
                        ));
                        let vb1 = _mm_castsi128_ps(_mm_insert_epi32::<0>(
                            _mm_slli_si128::<4>(_mm_castps_si128(vb1r)),
                            (*rows_ptr.add(input_base_idx)).to_bits() as i32,
                        ));
                        acc = acc.add(vk1.sub(vb1).abs());

                        let vk1l = _mm_castsi128_ps(_mm_insert_epi32::<0>(
                            _mm_slli_si128::<4>(_mm_castps_si128(vk1)),
                            (*rows_ptr.add(input_kernel_idx - 1)).to_bits() as i32,
                        ));
                        let vb1l = _mm_castsi128_ps(_mm_insert_epi32::<0>(
                            _mm_slli_si128::<4>(_mm_castps_si128(vb1)),
                            (*rows_ptr.add(input_base_idx - 1)).to_bits() as i32,
                        ));
                        acc = acc.add(vk1l.sub(vb1l).abs());

                        let vk2 = _mm_loadu_ps(rows_ptr.add(input_kernel_idx + width));
                        let vb2 = _mm_loadu_ps(rows_ptr.add(input_base_idx + width));
                        acc = acc.add(vk2.sub(vb2).abs());

                        dist = scale.mul(acc).add(dist);
                    } else {
                        let v0 = _mm_loadu_ps(rows_ptr.add(input_kernel_idx));
                        let v1 = _mm_loadu_ps(rows_ptr.add(input_base_idx));
                        dist = scale.mul(v0.sub(v1).abs()).add(dist);
                    }
                }

                let weight = weight_sse41(dist, sigma_val, sm);
                sum_weights = sum_weights.add(weight);

                for (c, sum) in sum_channels.iter_mut().enumerate() {
                    *sum = weight
                        .mul(_mm_loadu_ps(
                            merged_input_rows[c].as_ptr().add(input_kernel_idx),
                        ))
                        .add(*sum);
                }
            }
        }

        for (c, sum) in sum_channels.into_iter().enumerate() {
            _mm_storeu_ps(output_rows[c].as_mut_ptr().add(dx), sum.div(sum_weights));
        }
    }
}
