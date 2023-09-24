use std::arch::x86_64::*;

use jxl_grid::SimpleGrid;
use jxl_grid::SimdVector;

type Vector = __m128;

#[inline]
fn weight_sse2(scaled_distance: Vector, sigma: f32, step_multiplier: Vector) -> Vector {
    let neg_inv_sigma = Vector::splat_f32(6.6 * (std::f32::consts::FRAC_1_SQRT_2 - 1.0) / sigma).mul(step_multiplier);
    let result = scaled_distance.muladd(neg_inv_sigma, Vector::splat_f32(1.0));
    unsafe { _mm_max_ps(result, Vector::zero()) }
}

macro_rules! define_epf_sse2 {
    { $($v:vis fn $name:ident ($width:ident, $kernel_diff:expr, $dist_diff:expr $(,)?); )* } => {
        $(
            $v fn $name(
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
                let sm_left = Vector::set([border_sad_mul * step_multiplier, step_multiplier, step_multiplier, step_multiplier]);
                let sm_right = Vector::set([step_multiplier, step_multiplier, step_multiplier, border_sad_mul * step_multiplier]);

                for y in 3..height - 4 {
                    for x in (3..width - 3).step_by(Vector::SIZE) {
                        let sigma = *sigma_grid.get((x - 3) / 8, (y - 3) / 8).unwrap();
                        let idx_base = y * width + x;

                        if sigma < 0.3 {
                            for (input, output) in input_buf.into_iter().zip(&mut output_buf) {
                                output[idx_base..][..Vector::SIZE].copy_from_slice(&input[idx_base..][..Vector::SIZE]);
                            }
                            continue;
                        }

                        // SAFETY: Indexing doesn't go out of bounds since we have padding after image region.
                        let mut sum_weights = Vector::splat_f32(1.0);
                        let mut sum_channels = input_buf.map(|buf| {
                            unsafe { Vector::load(buf.as_ptr().add(idx_base)) }
                        });

                        for kdiff in $kernel_diff {
                            let kernel_base = idx_base.wrapping_add_signed(kdiff);
                            let mut dist = Vector::zero();
                            for (buf, scale) in input_buf.into_iter().zip(channel_scale) {
                                let scale = Vector::splat_f32(scale);
                                for diff in $dist_diff {
                                    unsafe {
                                        dist = scale.muladd(
                                            Vector::load(buf.as_ptr().add(idx_base.wrapping_add_signed(diff))).sub(
                                                Vector::load(buf.as_ptr().add(kernel_base.wrapping_add_signed(diff)))
                                            ).abs(),
                                            dist,
                                        );
                                    }
                                }
                            }

                            let weight = weight_sse2(
                                dist,
                                sigma,
                                if (y - 3) % 8 == 0 || (y - 3) % 8 == 7 {
                                    sm_y_edge
                                } else if (x - 3) % 8 == 0 {
                                    sm_left
                                } else {
                                    sm_right
                                },
                            );
                            sum_weights = sum_weights.add(weight);

                            for (sum, buf) in sum_channels.iter_mut().zip(input_buf) {
                                *sum = weight.muladd(unsafe { Vector::load(buf.as_ptr().add(kernel_base)) }, *sum);
                            }
                        }

                        for (buf, sum) in output_buf.iter_mut().zip(sum_channels) {
                            let val = sum.div(sum_weights);
                            unsafe { val.store(buf.as_mut_ptr().add(idx_base)); }
                        }
                    }
                }
            }
        )*
    };
}

define_epf_sse2! {
    pub fn epf_step0_sse2(
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
    pub fn epf_step1_sse2(
        width,
        [-width, -1, 1, width],
        [-width, -1, 0, 1, width],
    );
    pub fn epf_step2_sse2(
        width,
        [-width, -1, 1, width],
        [0isize],
    );
}
