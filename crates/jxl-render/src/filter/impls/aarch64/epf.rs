use std::arch::aarch64::*;

use jxl_grid::SimdVector;

use crate::filter::epf::*;

type Vector = float32x4_t;

#[target_feature(enable = "neon")]
#[inline]
unsafe fn weight_neon(scaled_distance: Vector, sigma: f32, step_multiplier: Vector) -> Vector {
    let neg_inv_sigma =
        vdupq_n_f32(6.6 * (std::f32::consts::FRAC_1_SQRT_2 - 1.0) / sigma).mul(step_multiplier);
    let result = scaled_distance.mul(neg_inv_sigma).add(vdupq_n_f32(1.0));
    vmaxq_f32(result, Vector::zero())
}

pub(crate) unsafe fn epf_row_aarch64_neon<const STEP: usize>(epf_row: EpfRow) {
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
        let sm = vdupq_n_f32(step_multiplier * border_sad_mul);
        [sm, sm]
    } else {
        let base_sm = vdupq_n_f32(step_multiplier);
        [
            vsetq_lane_f32::<0>(step_multiplier * border_sad_mul, base_sm),
            vsetq_lane_f32::<3>(step_multiplier * border_sad_mul, base_sm),
        ]
    };

    for dx in simd_range.step_by(4) {
        let sigma_x = x + dx;
        let sm = sm[(sigma_x / 4) & 1];
        let sigma_x = sigma_x / 8 - x / 8;

        let sigma_val = sigma_row[sigma_x];

        let originals: [_; 3] = std::array::from_fn(|c| unsafe {
            vld1q_f32(merged_input_rows[c].get_row(3).as_ptr().add(dx))
        });

        if sigma_val < 0.3 {
            for (c, val) in originals.into_iter().enumerate() {
                vst1q_f32(output_rows[c].as_mut_ptr().add(dx), val);
            }
            continue;
        }

        let mut sum_weights = vdupq_n_f32(1.0);
        let mut sum_channels = originals;

        if STEP == 1 {
            // (0, -1), (0, 1)
            {
                let mut dist0 = vdupq_n_f32(0.0);
                let mut dist1 = vdupq_n_f32(0.0);
                for c in 0..3 {
                    let scale = channel_scale[c];
                    let input_rows = &merged_input_rows[c];

                    let v0 = vld1q_f32(input_rows.get_row(1).as_ptr().add(dx));

                    let v1r = vld1q_f32(input_rows.get_row(2).as_ptr().add(dx + 1));
                    let v1lc = vld1_f32(input_rows.get_row(2).as_ptr().add(dx - 1));
                    let v1lcq = vcombine_f32(vdup_n_f32(0.0), v1lc);
                    let v1 = vextq_f32::<3>(v1lcq, v1r);
                    let v1l = vcombine_f32(v1lc, vget_low_f32(v1r));

                    let v2r = vld1q_f32(input_rows.get_row(3).as_ptr().add(dx + 1));
                    let v2lc = vld1_f32(input_rows.get_row(3).as_ptr().add(dx - 1));
                    let v2lcq = vcombine_f32(vdup_n_f32(0.0), v2lc);
                    let v2 = vextq_f32::<3>(v2lcq, v2r);
                    let v2l = vcombine_f32(v2lc, vget_low_f32(v2r));

                    let v3r = vld1q_f32(input_rows.get_row(4).as_ptr().add(dx + 1));
                    let v3lc = vld1_f32(input_rows.get_row(4).as_ptr().add(dx - 1));
                    let v3lcq = vcombine_f32(vdup_n_f32(0.0), v3lc);
                    let v3 = vextq_f32::<3>(v3lcq, v3r);
                    let v3l = vcombine_f32(v3lc, vget_low_f32(v3r));

                    let v4 = vld1q_f32(input_rows.get_row(5).as_ptr().add(dx));

                    let tmp = v1.sub(v2).abs().add(v3.sub(v2).abs());
                    let mut acc0 = tmp.add(v1.sub(v0).abs());
                    let mut acc1 = tmp.add(v3.sub(v4).abs());

                    acc0 = acc0.add(v1l.sub(v2l).abs());
                    acc1 = acc1.add(v3l.sub(v2l).abs());

                    acc0 = acc0.add(v1r.sub(v2r).abs());
                    acc1 = acc1.add(v3r.sub(v2r).abs());

                    dist0 = vmulq_n_f32(acc0, scale).add(dist0);
                    dist1 = vmulq_n_f32(acc1, scale).add(dist1);
                }

                let weight0 = weight_neon(dist0, sigma_val, sm);
                let weight1 = weight_neon(dist1, sigma_val, sm);
                sum_weights = sum_weights.add(weight0).add(weight1);

                for (c, sum) in sum_channels.iter_mut().enumerate() {
                    let input_rows = &merged_input_rows[c];
                    let weighted0 = weight0.mul(vld1q_f32(input_rows.get_row(2).as_ptr().add(dx)));
                    let weighted1 = weight1.mul(vld1q_f32(input_rows.get_row(4).as_ptr().add(dx)));
                    *sum = sum.add(weighted0).add(weighted1);
                }
            }

            // (-1, 0), (1, 0)
            {
                let mut dist0 = vdupq_n_f32(0.0);
                let mut dist1 = vdupq_n_f32(0.0);
                for c in 0..3 {
                    let scale = channel_scale[c];
                    let input_rows = &merged_input_rows[c];

                    let v0r = vld1q_f32(input_rows.get_row(2).as_ptr().add(dx + 1));
                    let v0lc = vld1_f32(input_rows.get_row(2).as_ptr().add(dx - 1));
                    let v0lcq = vcombine_f32(vdup_n_f32(0.0), v0lc);
                    let v0 = vextq_f32::<3>(v0lcq, v0r);
                    let v0l = vcombine_f32(v0lc, vget_low_f32(v0r));
                    let mut acc0 = v0l.sub(v0).abs();
                    let mut acc1 = v0r.sub(v0).abs();

                    let v1rr = vld1q_f32(input_rows.get_row(3).as_ptr().add(dx + 2));
                    let v1ll = vld1q_f32(input_rows.get_row(3).as_ptr().add(dx - 2));
                    let v1r = vextq_f32::<3>(v1ll, v1rr);
                    let v1 = vextq_f32::<2>(v1ll, v1rr);
                    let v1l = vextq_f32::<1>(v1ll, v1rr);
                    acc0 = acc0.add(v1ll.sub(v1l).abs());
                    acc0 = acc0.add(v1.sub(v1l).abs());
                    acc0 = acc0.add(v1.sub(v1r).abs());
                    acc1 = acc1.add(v1.sub(v1l).abs());
                    acc1 = acc1.add(v1.sub(v1r).abs());
                    acc1 = acc1.add(v1rr.sub(v1r).abs());

                    let v2r = vld1q_f32(input_rows.get_row(4).as_ptr().add(dx + 1));
                    let v2lc = vld1_f32(input_rows.get_row(4).as_ptr().add(dx - 1));
                    let v2lcq = vcombine_f32(vdup_n_f32(0.0), v2lc);
                    let v2 = vextq_f32::<3>(v2lcq, v2r);
                    let v2l = vcombine_f32(v2lc, vget_low_f32(v2r));
                    acc0 = acc0.add(v2l.sub(v2).abs());
                    acc1 = acc1.add(v2r.sub(v2).abs());

                    dist0 = vmulq_n_f32(acc0, scale).add(dist0);
                    dist1 = vmulq_n_f32(acc1, scale).add(dist1);
                }

                let weight0 = weight_neon(dist0, sigma_val, sm);
                let weight1 = weight_neon(dist1, sigma_val, sm);
                sum_weights = sum_weights.add(weight0).add(weight1);

                for (c, sum) in sum_channels.iter_mut().enumerate() {
                    let input_rows = &merged_input_rows[c];
                    let weighted0 =
                        weight0.mul(vld1q_f32(input_rows.get_row(3).as_ptr().add(dx - 1)));
                    let weighted1 =
                        weight1.mul(vld1q_f32(input_rows.get_row(3).as_ptr().add(dx + 1)));
                    *sum = sum.add(weighted0).add(weighted1);
                }
            }
        } else {
            for &(kx, ky) in kernel_offsets {
                let ky = 3usize.wrapping_add_signed(ky);
                let kx = dx.wrapping_add_signed(kx);
                let mut dist = vdupq_n_f32(0.0);
                for c in 0..3 {
                    let scale = channel_scale[c];
                    let input_rows = &merged_input_rows[c];
                    if STEP == 0 {
                        let vk0 = vld1q_f32(input_rows.get_row(ky - 1).as_ptr().add(kx));
                        let vb0 = vld1q_f32(input_rows.get_row(2).as_ptr().add(dx));
                        let mut acc = vk0.sub(vb0).abs();

                        let vk1r = vld1q_f32(input_rows.get_row(ky).as_ptr().add(kx + 1));
                        let vb1r = vld1q_f32(input_rows.get_row(3).as_ptr().add(dx + 1));
                        acc = acc.add(vk1r.sub(vb1r).abs());

                        let vk1lc = vld1_f32(input_rows.get_row(ky).as_ptr().add(kx - 1));
                        let vb1lc = vld1_f32(input_rows.get_row(3).as_ptr().add(dx - 1));

                        let vk1lcq = vcombine_f32(vdup_n_f32(0.0), vk1lc);
                        let vb1lcq = vcombine_f32(vdup_n_f32(0.0), vb1lc);
                        let vk1 = vextq_f32::<3>(vk1lcq, vk1r);
                        let vb1 = vextq_f32::<3>(vb1lcq, vb1r);
                        acc = acc.add(vk1.sub(vb1).abs());

                        let vk1l = vcombine_f32(vk1lc, vget_low_f32(vk1r));
                        let vb1l = vcombine_f32(vb1lc, vget_low_f32(vb1r));
                        acc = acc.add(vk1l.sub(vb1l).abs());

                        let vk2 = vld1q_f32(input_rows.get_row(ky + 1).as_ptr().add(kx));
                        let vb2 = vld1q_f32(input_rows.get_row(4).as_ptr().add(dx));
                        acc = acc.add(vk2.sub(vb2).abs());

                        dist = vmulq_n_f32(acc, scale).add(dist);
                    } else {
                        let v0 = vld1q_f32(input_rows.get_row(ky).as_ptr().add(kx));
                        let v1 = vld1q_f32(input_rows.get_row(3).as_ptr().add(dx));
                        dist = vmulq_n_f32(v0.sub(v1).abs(), scale).add(dist);
                    }
                }

                let weight = weight_neon(dist, sigma_val, sm);
                sum_weights = sum_weights.add(weight);

                for (c, sum) in sum_channels.iter_mut().enumerate() {
                    *sum = weight
                        .mul(vld1q_f32(merged_input_rows[c].get_row(ky).as_ptr().add(kx)))
                        .add(*sum);
                }
            }
        }

        for (c, sum) in sum_channels.into_iter().enumerate() {
            vst1q_f32(output_rows[c].as_mut_ptr().add(dx), sum.div(sum_weights));
        }
    }
}
