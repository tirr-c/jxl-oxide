use std::arch::wasm32::*;

use jxl_grid::SimdVector;

use crate::filter::epf::*;

type Vector = v128;

#[target_feature(enable = "simd128")]
unsafe fn weight_wasm32_simd128(
    scaled_distance: Vector,
    sigma: f32,
    step_multiplier: Vector,
) -> Vector {
    let neg_inv_sigma = Vector::splat_f32(6.6 * (std::f32::consts::FRAC_1_SQRT_2 - 1.0) / sigma)
        .mul(step_multiplier);
    let result = scaled_distance
        .mul(neg_inv_sigma)
        .add(Vector::splat_f32(1.0));
    f32x4_max(result, Vector::zero())
}

#[target_feature(enable = "simd128")]
pub(crate) unsafe fn epf_row_wasm32_simd128<const STEP: usize>(epf_row: EpfRow) {
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
        let sm = Vector::splat_f32(step_multiplier * border_sad_mul);
        [sm, sm]
    } else {
        let base_sm = Vector::splat_f32(step_multiplier);
        [
            f32x4_replace_lane::<0>(base_sm, step_multiplier * border_sad_mul),
            f32x4_replace_lane::<3>(base_sm, step_multiplier * border_sad_mul),
        ]
    };

    for dx in simd_range.step_by(4) {
        let sigma_x = x + dx;
        let sm = sm[(sigma_x / 4) & 1];
        let sigma_x = sigma_x / 8 - x / 8;

        let sigma_val = sigma_row[sigma_x];

        let originals: [_; 3] = std::array::from_fn(|c| unsafe {
            Vector::load(merged_input_rows[c].get_row(3).as_ptr().add(dx))
        });

        if sigma_val < 0.3 {
            for (c, val) in originals.into_iter().enumerate() {
                val.store(output_rows[c].as_mut_ptr().add(dx));
            }
            continue;
        }

        let mut sum_weights = Vector::splat_f32(1.0);
        let mut sum_channels = originals;

        if STEP == 1 {
            // (0, -1), (0, 1)
            {
                let mut dist0 = Vector::zero();
                let mut dist1 = Vector::zero();
                for c in 0..3 {
                    let scale = Vector::splat_f32(channel_scale[c]);
                    let input_rows = &merged_input_rows[c];

                    let v0 = Vector::load(input_rows.get_row(1).as_ptr().add(dx));

                    //      0 1 2 3
                    // v1l: A B C D
                    //      4 5 6 7
                    // v1r: C D E F
                    //      1 2 3 6
                    // v1 : B C D E
                    let v1r = Vector::load(input_rows.get_row(2).as_ptr().add(dx + 1));
                    let v1l = Vector::load(input_rows.get_row(2).as_ptr().add(dx - 1));
                    let v1 = i32x4_shuffle::<1, 2, 3, 6>(v1l, v1r);

                    let v2r = Vector::load(input_rows.get_row(3).as_ptr().add(dx + 1));
                    let v2l = Vector::load(input_rows.get_row(3).as_ptr().add(dx - 1));
                    let v2 = i32x4_shuffle::<1, 2, 3, 6>(v2l, v2r);

                    let v3r = Vector::load(input_rows.get_row(4).as_ptr().add(dx + 1));
                    let v3l = Vector::load(input_rows.get_row(4).as_ptr().add(dx - 1));
                    let v3 = i32x4_shuffle::<1, 2, 3, 6>(v3l, v3r);

                    let v4 = Vector::load(input_rows.get_row(5).as_ptr().add(dx));

                    let tmp = v1.sub(v2).abs().add(v3.sub(v2).abs());
                    let mut acc0 = tmp.add(v1.sub(v0).abs());
                    let mut acc1 = tmp.add(v3.sub(v4).abs());

                    acc0 = acc0.add(v1l.sub(v2l).abs());
                    acc1 = acc1.add(v3l.sub(v2l).abs());

                    acc0 = acc0.add(v1r.sub(v2r).abs());
                    acc1 = acc1.add(v3r.sub(v2r).abs());

                    dist0 = acc0.mul(scale).add(dist0);
                    dist1 = acc1.mul(scale).add(dist1);
                }

                let weight0 = weight_wasm32_simd128(dist0, sigma_val, sm);
                let weight1 = weight_wasm32_simd128(dist1, sigma_val, sm);
                sum_weights = sum_weights.add(weight0).add(weight1);

                for (c, sum) in sum_channels.iter_mut().enumerate() {
                    let input_rows = &merged_input_rows[c];
                    let weighted0 =
                        weight0.mul(Vector::load(input_rows.get_row(2).as_ptr().add(dx)));
                    let weighted1 =
                        weight1.mul(Vector::load(input_rows.get_row(4).as_ptr().add(dx)));
                    *sum = sum.add(weighted0).add(weighted1);
                }
            }

            // (-1, 0), (1, 0)
            {
                let mut dist0 = Vector::zero();
                let mut dist1 = Vector::zero();
                for c in 0..3 {
                    let scale = Vector::splat_f32(channel_scale[c]);
                    let input_rows = &merged_input_rows[c];

                    let v0r = Vector::load(input_rows.get_row(2).as_ptr().add(dx + 1));
                    let v0l = Vector::load(input_rows.get_row(2).as_ptr().add(dx - 1));
                    let v0 = i32x4_shuffle::<1, 2, 3, 6>(v0l, v0r);
                    let mut acc0 = v0l.sub(v0).abs();
                    let mut acc1 = v0r.sub(v0).abs();

                    let v1rr = Vector::load(input_rows.get_row(3).as_ptr().add(dx + 2));
                    let v1ll = Vector::load(input_rows.get_row(3).as_ptr().add(dx - 2));
                    let v1l = i32x4_shuffle::<1, 2, 3, 4>(v1ll, v1rr);
                    let v1 = i32x4_shuffle::<2, 3, 4, 5>(v1ll, v1rr);
                    let v1r = i32x4_shuffle::<3, 4, 5, 6>(v1ll, v1rr);
                    acc0 = acc0.add(v1ll.sub(v1l).abs());
                    acc0 = acc0.add(v1.sub(v1l).abs());
                    acc0 = acc0.add(v1.sub(v1r).abs());
                    acc1 = acc1.add(v1.sub(v1l).abs());
                    acc1 = acc1.add(v1.sub(v1r).abs());
                    acc1 = acc1.add(v1rr.sub(v1r).abs());

                    let v2r = Vector::load(input_rows.get_row(4).as_ptr().add(dx + 1));
                    let v2l = Vector::load(input_rows.get_row(4).as_ptr().add(dx - 1));
                    let v2 = i32x4_shuffle::<1, 2, 3, 6>(v2l, v2r);
                    acc0 = acc0.add(v2l.sub(v2).abs());
                    acc1 = acc1.add(v2r.sub(v2).abs());

                    dist0 = scale.mul(acc0).add(dist0);
                    dist1 = scale.mul(acc1).add(dist1);
                }

                let weight0 = weight_wasm32_simd128(dist0, sigma_val, sm);
                let weight1 = weight_wasm32_simd128(dist1, sigma_val, sm);
                sum_weights = sum_weights.add(weight0).add(weight1);

                for (c, sum) in sum_channels.iter_mut().enumerate() {
                    let input_rows = &merged_input_rows[c];
                    let weighted0 =
                        weight0.mul(Vector::load(input_rows.get_row(3).as_ptr().add(dx - 1)));
                    let weighted1 =
                        weight1.mul(Vector::load(input_rows.get_row(3).as_ptr().add(dx + 1)));
                    *sum = sum.add(weighted0).add(weighted1);
                }
            }
        } else {
            for &(kx, ky) in kernel_offsets {
                let ky = 3usize.wrapping_add_signed(ky);
                let kx = dx.wrapping_add_signed(kx);
                let mut dist = Vector::zero();
                for c in 0..3 {
                    let scale = Vector::splat_f32(channel_scale[c]);
                    let input_rows = &merged_input_rows[c];
                    if STEP == 0 {
                        let vk0 = Vector::load(input_rows.get_row(ky - 1).as_ptr().add(kx));
                        let vb0 = Vector::load(input_rows.get_row(2).as_ptr().add(dx));
                        let mut acc = vk0.sub(vb0).abs();

                        let vk1r = Vector::load(input_rows.get_row(ky).as_ptr().add(kx + 1));
                        let vb1r = Vector::load(input_rows.get_row(3).as_ptr().add(dx + 1));
                        acc = acc.add(vk1r.sub(vb1r).abs());

                        let vk1l = Vector::load(input_rows.get_row(ky).as_ptr().add(kx - 1));
                        let vb1l = Vector::load(input_rows.get_row(3).as_ptr().add(dx - 1));
                        let vk1 = i32x4_shuffle::<1, 2, 3, 6>(vk1l, vk1r);
                        let vb1 = i32x4_shuffle::<1, 2, 3, 6>(vb1l, vb1r);
                        acc = acc.add(vk1.sub(vb1).abs());
                        acc = acc.add(vk1l.sub(vb1l).abs());

                        let vk2 = Vector::load(input_rows.get_row(ky + 1).as_ptr().add(kx));
                        let vb2 = Vector::load(input_rows.get_row(4).as_ptr().add(dx));
                        acc = acc.add(vk2.sub(vb2).abs());

                        dist = acc.mul(scale).add(dist);
                    } else {
                        let v0 = Vector::load(input_rows.get_row(ky).as_ptr().add(kx));
                        let v1 = Vector::load(input_rows.get_row(3).as_ptr().add(dx));
                        dist = v0.sub(v1).abs().mul(scale).add(dist);
                    }
                }

                let weight = weight_wasm32_simd128(dist, sigma_val, sm);
                sum_weights = sum_weights.add(weight);

                for (c, sum) in sum_channels.iter_mut().enumerate() {
                    *sum = weight
                        .mul(Vector::load(
                            merged_input_rows[c].get_row(ky).as_ptr().add(kx),
                        ))
                        .add(*sum);
                }
            }
        }

        for (c, sum) in sum_channels.into_iter().enumerate() {
            sum.div(sum_weights)
                .store(output_rows[c].as_mut_ptr().add(dx));
        }
    }
}
