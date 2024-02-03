use std::arch::x86_64::*;

use jxl_grid::SimdVector;

use crate::filter::impls::generic::EpfRow;

type Vector = __m256;

#[inline]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
unsafe fn weight_avx2(scaled_distance: Vector, sigma: Vector, step_multiplier: Vector) -> Vector {
    let neg_inv_sigma = Vector::splat_f32(6.6 * (std::f32::consts::FRAC_1_SQRT_2 - 1.0))
        .div(sigma)
        .mul(step_multiplier);
    let result = _mm256_fmadd_ps(scaled_distance, neg_inv_sigma, Vector::splat_f32(1.0));
    _mm256_max_ps(result, Vector::zero())
}

#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
pub(crate) unsafe fn epf_row_x86_64_avx2<const STEP: usize>(epf_row: EpfRow<'_, '_>) {
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
    let (kernel_offsets, dist_offsets) = super::super::generic::epf_kernel::<STEP>();

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
    let right_edge_width = ((width - padding * 2) & 7) + padding;
    let right_edge_start = width - right_edge_width;

    let simd_width = right_edge_start - padding;
    assert!(simd_width % 8 == 0);

    let is_y_border = (y + 1) & 0b110 == 0;
    let sm = if is_y_border {
        [step_multiplier * border_sad_mul; 8]
    } else {
        let x = x + padding;
        let neg_x = 8 - (x & 7);
        let mut sm = [step_multiplier; 8];
        sm[neg_x & 7] *= border_sad_mul;
        sm[(neg_x + 7) & 7] *= border_sad_mul;
        sm
    };
    let sm = _mm256_loadu_ps(sm.as_ptr());

    for dx8 in 0..simd_width / 8 {
        let dx = dx8 * 8 + padding;
        let input_base_idx = 3 * width + dx;
        let sigma_val = _mm256_loadu_ps(sigma_row.as_ptr().add(dx));
        let mask = _mm256_cmp_ps::<_CMP_LT_OS>(sigma_val, _mm256_set1_ps(0.3));

        let originals: [_; 3] = std::array::from_fn(|c| {
            unsafe {
                _mm256_loadu_ps(merged_input_rows[c].as_ptr().add(input_base_idx))
            }
        });

        let mut sum_weights = _mm256_set1_ps(1.0);
        let mut sum_channels = originals;

        for &(kx, ky) in kernel_offsets {
            let input_kernel_idx = input_base_idx.wrapping_add_signed(ky * iwidth + kx);
            let mut dist = _mm256_setzero_ps();
            for c in 0..3 {
                let scale = _mm256_set1_ps(channel_scale[c]);
                for &(ix, iy) in dist_offsets {
                    let offset = iy * iwidth + ix;
                    let input_kernel_idx = input_kernel_idx.wrapping_add_signed(offset);
                    let input_base_idx = input_base_idx.wrapping_add_signed(offset);

                    let v0 = _mm256_loadu_ps(merged_input_rows[c].as_ptr().add(input_kernel_idx));
                    let v1 = _mm256_loadu_ps(merged_input_rows[c].as_ptr().add(input_base_idx));
                    dist = _mm256_fmadd_ps(
                        scale,
                        v0.sub(v1).abs(),
                        dist,
                    );
                }
            }

            let weight = weight_avx2(
                dist,
                sigma_val,
                sm,
            );
            sum_weights = sum_weights.add(weight);

            for (c, sum) in sum_channels.iter_mut().enumerate() {
                *sum = _mm256_fmadd_ps(
                    weight,
                    _mm256_loadu_ps(merged_input_rows[c].as_ptr().add(input_kernel_idx)),
                    *sum,
                );
            }
        }

        for (c, sum) in sum_channels.into_iter().enumerate() {
            let output = _mm256_blendv_ps(
                sum.div(sum_weights),
                originals[c],
                mask,
            );
            _mm256_storeu_ps(output_rows[c].as_mut_ptr().add(dx), output);
        }
    }
}
