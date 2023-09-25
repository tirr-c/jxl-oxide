use std::arch::x86_64::*;

use jxl_grid::SimpleGrid;
use jxl_grid::SimdVector;

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
    { $($v:vis unsafe fn $name:ident ($width:ident, $kernel_diff:expr, $dist_diff:expr $(,)?); )* } => {
        $(
            #[target_feature(enable = "avx2")]
            #[target_feature(enable = "fma")]
            $v unsafe fn $name(
                input: &[SimpleGrid<f32>; 3],
                output: &mut [SimpleGrid<f32>; 3],
                sigma_grid: &SimpleGrid<f32>,
                channel_scale: [f32; 3],
                border_sad_mul: f32,
                step_multiplier: f32,
            ) {
                let width = input[0].width();
                let $width = width as isize;
                let height = input[0].height();
                assert_eq!(input[1].width(), width);
                assert_eq!(input[2].width(), width);
                assert_eq!(input[1].height(), height);
                assert_eq!(input[2].height(), height);
                assert_eq!(output[0].width(), width);
                assert_eq!(output[1].width(), width);
                assert_eq!(output[2].width(), width);
                assert_eq!(output[0].height(), height);
                assert_eq!(output[1].height(), height);
                assert_eq!(output[2].height(), height);

                let input_buf = [input[0].buf(), input[1].buf(), input[2].buf()];
                let mut output_buf = {
                    let [a, b, c] = output;
                    [a.buf_mut(), b.buf_mut(), c.buf_mut()]
                };

                let sm_y_edge = Vector::splat_f32(border_sad_mul * step_multiplier);
                let sm = Vector::set([
                    border_sad_mul * step_multiplier, step_multiplier, step_multiplier, step_multiplier,
                    step_multiplier, step_multiplier, step_multiplier, border_sad_mul * step_multiplier,
                ]);

                for y in 3..height - 4 {
                    let sigma_row = &sigma_grid.buf()[(y - 3) / 8 * sigma_grid.width()..][..(width + 1) / 8];
                    for (vx, &sigma) in sigma_row.iter().enumerate() {
                        let x = 3 + vx * 8;
                        let idx_base = y * width + x;

                        // SAFETY: Indexing doesn't go out of bounds since we have padding after image region.
                        let mut sum_weights = Vector::splat_f32(1.0);
                        let mut sum_channels = input_buf.map(|buf| {
                            Vector::load(buf.as_ptr().add(idx_base))
                        });

                        if sigma < 0.3 {
                            for (buf, sum) in output_buf.iter_mut().zip(sum_channels) {
                                sum.store(buf.as_mut_ptr().add(idx_base));
                            }
                            continue;
                        }

                        for kdiff in $kernel_diff {
                            let kernel_base = idx_base.wrapping_add_signed(kdiff);
                            let mut dist = Vector::zero();
                            for (buf, scale) in input_buf.into_iter().zip(channel_scale) {
                                let scale = Vector::splat_f32(scale);
                                for diff in $dist_diff {
                                    dist = _mm256_fmadd_ps(
                                        scale,
                                        Vector::load(buf.as_ptr().add(idx_base.wrapping_add_signed(diff))).sub(
                                            Vector::load(buf.as_ptr().add(kernel_base.wrapping_add_signed(diff)))
                                        ).abs(),
                                        dist,
                                    );
                                }
                            }

                            let weight = weight_avx2(
                                dist,
                                sigma,
                                if (y - 3) % 8 == 0 || (y - 3) % 8 == 7 {
                                    sm_y_edge
                                } else {
                                    sm
                                },
                            );
                            sum_weights = sum_weights.add(weight);

                            for (sum, buf) in sum_channels.iter_mut().zip(input_buf) {
                                *sum = _mm256_fmadd_ps(
                                    weight,
                                    Vector::load(buf.as_ptr().add(kernel_base)),
                                    *sum,
                                );
                            }
                        }

                        for (buf, sum) in output_buf.iter_mut().zip(sum_channels) {
                            let val = sum.div(sum_weights);
                            val.store(buf.as_mut_ptr().add(idx_base));
                        }
                    }
                }
            }
        )*
    };
}

define_epf_avx2! {
    pub unsafe fn epf_step0_avx2(
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
    pub unsafe fn epf_step1_avx2(
        width,
        [-width, -1, 1, width],
        [-width, -1, 0, 1, width],
    );
    pub unsafe fn epf_step2_avx2(
        width,
        [-width, -1, 1, width],
        [0isize],
    );
}
