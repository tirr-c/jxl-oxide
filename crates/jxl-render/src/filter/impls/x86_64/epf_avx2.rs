use std::arch::x86_64::*;

use jxl_grid::SimdVector;

use crate::filter::impls::generic::EpfRow;

type Vector = __m256;

#[inline]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
unsafe fn weight_avx2(scaled_distance: Vector, sigma: f32, step_multiplier: Vector) -> Vector {
    let neg_inv_sigma = Vector::splat_f32(6.6 * (std::f32::consts::FRAC_1_SQRT_2 - 1.0) / sigma).mul(step_multiplier);
    let result = _mm256_fmadd_ps(
        scaled_distance,
        neg_inv_sigma,
        Vector::splat_f32(1.0),
    );
    _mm256_max_ps(result, Vector::zero())
}

macro_rules! define_epf_avx2 {
    { $($v:vis unsafe fn $name:ident ($width:ident, $kernel_offsets:expr, $dist_offsets:expr $(,)?); )* } => {
        $(
            #[target_feature(enable = "avx2")]
            #[target_feature(enable = "fma")]
            $v unsafe fn $name(epf_row: EpfRow<'_>) {
                let EpfRow {
                    input_buf,
                    mut output_buf_rows,
                    width,
                    y8,
                    dy,
                    sigma_grid,
                    channel_scale,
                    border_sad_mul,
                    step_multiplier,
                } = epf_row;

                let y = y8 * 8 + dy;
                let $width = width as isize;

                let is_y_border = dy == 0 || dy == 7;
                let sm = if is_y_border {
                    Vector::splat_f32(border_sad_mul * step_multiplier)
                } else {
                    Vector::set([
                        border_sad_mul * step_multiplier,
                        step_multiplier,
                        step_multiplier,
                        step_multiplier,
                        step_multiplier,
                        step_multiplier,
                        step_multiplier,
                        border_sad_mul * step_multiplier,
                    ])
                };

                for x8 in 1..width / 8 {
                    let Some(&sigma) = sigma_grid.get(x8 - 1, y8) else { break; };

                    let base_x = x8 * 8;
                    let base_idx = (y + 3) * width + base_x;

                    // SAFETY: Indexing doesn't go out of bounds since we have padding after image region.
                    // SAFETY: Every row is aligned to 32 bytes.
                    let mut sum_weights = Vector::splat_f32(1.0);
                    let mut sum_channels = input_buf.map(|buf| {
                        Vector::load_aligned(buf.as_ptr().add(base_idx))
                    });

                    if sigma < 0.3 {
                        for (buf, sum) in output_buf_rows.iter_mut().zip(sum_channels) {
                            sum.store_aligned(buf.as_mut_ptr().add(base_x));
                        }
                        continue;
                    }

                    for offset in $kernel_offsets {
                        let kernel_idx = base_idx.wrapping_add_signed(offset);
                        let mut dist = Vector::zero();
                        for (buf, scale) in input_buf.into_iter().zip(channel_scale) {
                            let scale = Vector::splat_f32(scale);
                            for offset in $dist_offsets {
                                let base_idx = base_idx.wrapping_add_signed(offset);
                                let kernel_idx = kernel_idx.wrapping_add_signed(offset);
                                dist = _mm256_fmadd_ps(
                                    scale,
                                    Vector::load(buf.as_ptr().add(base_idx)).sub(
                                        Vector::load(buf.as_ptr().add(kernel_idx))
                                    ).abs(),
                                    dist,
                                );
                            }
                        }

                        let weight = weight_avx2(
                            dist,
                            sigma,
                            sm,
                        );
                        sum_weights = sum_weights.add(weight);

                        for (sum, buf) in sum_channels.iter_mut().zip(input_buf) {
                            *sum = _mm256_fmadd_ps(
                                weight,
                                Vector::load(buf.as_ptr().add(kernel_idx)),
                                *sum,
                            );
                        }
                    }

                    for (buf, sum) in output_buf_rows.iter_mut().zip(sum_channels) {
                        let val = sum.div(sum_weights);
                        val.store_aligned(buf.as_mut_ptr().add(base_x));
                    }
                }
            }
        )*
    };
}

define_epf_avx2! {
    pub(crate) unsafe fn epf_row_step0_avx2(
        width,
        [
            -2 * width,
            -1 - width, -width, 1 - width,
            -2, -1, 1, 2,
            width - 1, width, width + 1,
            2 * width,
        ],
        [-width, -1, 0, 1, width],
    );
    pub(crate) unsafe fn epf_row_step1_avx2(
        width,
        [-width, -1, 1, width],
        [-width, -1, 0, 1, width],
    );
    pub(crate) unsafe fn epf_row_step2_avx2(
        width,
        [-width, -1, 1, width],
        [0isize],
    );
}
