use std::arch::aarch64::*;

use jxl_grid::SimpleGrid;
use jxl_grid::SimdVector;

type Vector = float32x4_t;

#[target_feature(enable = "neon")]
#[inline]
unsafe fn weight_neon(scaled_distance: Vector, sigma: f32, step_multiplier: Vector) -> Vector {
    let result = vfmaq_n_f32(
        Vector::splat_f32(1.0),
        scaled_distance.mul(step_multiplier),
        6.6 * (std::f32::consts::FRAC_1_SQRT_2 - 1.0) / sigma,
    );
    vmaxq_f32(result, Vector::zero())
}

macro_rules! define_epf_neon {
    { $($v:vis unsafe fn $name:ident ($width:ident, $kernel_offsets:expr, $dist_offsets:expr $(,)?); )* } => {
        $(
            #[target_feature(enable = "neon")]
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
                assert!(width % 8 == 0);

                let input_buf = [input[0].buf(), input[1].buf(), input[2].buf()];
                let mut output_buf = {
                    let [a, b, c] = output;
                    [a.buf_mut(), b.buf_mut(), c.buf_mut()]
                };
                assert_eq!(input_buf[0].len(), width * height);
                assert_eq!(input_buf[1].len(), width * height);
                assert_eq!(input_buf[2].len(), width * height);
                assert_eq!(output_buf[0].len(), width * height);
                assert_eq!(output_buf[1].len(), width * height);
                assert_eq!(output_buf[2].len(), width * height);

                let height = height - 6;

                for y in 0..height {
                    let y8 = y / 8;
                    let is_y_border = (y % 8) == 0 || (y % 8) == 7;
                    let sm = if is_y_border {
                        let sm_y_edge = Vector::splat_f32(border_sad_mul * step_multiplier);
                        [sm_y_edge, sm_y_edge]
                    } else {
                        let sm = Vector::splat_f32(step_multiplier);
                        [
                            vsetq_lane_f32(border_sad_mul * step_multiplier, sm, 0),
                            vsetq_lane_f32(border_sad_mul * step_multiplier, sm, 3),
                        ]
                    };

                    for x8 in 1..width / 8 {
                        let Some(&sigma) = sigma_grid.get(x8 - 1, y8) else { break; };

                        for (dx, sm) in sm.into_iter().enumerate() {
                            let base_x = x8 * 8 + dx * Vector::SIZE;
                            let base_idx = (y + 3) * width + base_x;

                            // SAFETY: Indexing doesn't go out of bounds since we have padding after image region.
                            let mut sum_weights = Vector::splat_f32(1.0);
                            let mut sum_channels = input_buf.map(|buf| {
                                Vector::load(buf.as_ptr().add(base_idx))
                            });

                            if sigma < 0.3 {
                                for (buf, sum) in output_buf.iter_mut().zip(sum_channels) {
                                    sum.store(buf.as_mut_ptr().add(base_idx));
                                }
                                continue;
                            }

                            for offset in $kernel_offsets {
                                let kernel_idx = base_idx.wrapping_add_signed(offset);
                                let mut dist = Vector::zero();
                                for (buf, scale) in input_buf.into_iter().zip(channel_scale) {
                                    let mut acc = Vector::zero();
                                    for offset in $dist_offsets {
                                        let base_idx = base_idx.wrapping_add_signed(offset);
                                        let kernel_idx = kernel_idx.wrapping_add_signed(offset);
                                        acc = acc.add(vabdq_f32(
                                            Vector::load(buf.as_ptr().add(base_idx)),
                                            Vector::load(buf.as_ptr().add(kernel_idx)),
                                        ));
                                    }
                                    dist = vfmaq_n_f32(
                                        dist,
                                        acc,
                                        scale,
                                    );
                                }

                                let weight = weight_neon(
                                    dist,
                                    sigma,
                                    sm,
                                );
                                sum_weights = sum_weights.add(weight);

                                for (sum, buf) in sum_channels.iter_mut().zip(input_buf) {
                                    *sum = weight.muladd(Vector::load(buf.as_ptr().add(kernel_idx)), *sum);
                                }
                            }

                            for (buf, sum) in output_buf.iter_mut().zip(sum_channels) {
                                let val = sum.div(sum_weights);
                                val.store(buf.as_mut_ptr().add(base_idx));
                            }
                        }
                    }
                }
            }
        )*
    };
}

define_epf_neon! {
    pub unsafe fn epf_step0_neon(
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
    pub unsafe fn epf_step1_neon(
        width,
        [-width, -1, 1, width],
        [-width, -1, 0, 1, width],
    );
    pub unsafe fn epf_step2_neon(
        width,
        [-width, -1, 1, width],
        [0isize],
    );
}
