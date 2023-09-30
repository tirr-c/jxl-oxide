#![allow(dead_code)]

use jxl_grid::SimpleGrid;

#[inline]
fn weight(scaled_distance: f32, sigma: f32, step_multiplier: f32) -> f32 {
    let inv_sigma = step_multiplier * 6.6 * (1.0 - std::f32::consts::FRAC_1_SQRT_2) / sigma;
    (1.0 - scaled_distance * inv_sigma).max(0.0)
}

macro_rules! define_epf {
    { $($v:vis fn $name:ident ($width:ident, $kernel_offsets:expr, $dist_offsets:expr $(,)?); )* } => {
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
                let height = height - 6;
                for y in 0..height {
                    let y8 = y / 8;
                    let is_y_border = (y % 8) == 0 || (y % 8) == 7;
                    let sm = if is_y_border {
                        [border_sad_mul * step_multiplier; 8]
                    } else {
                        [
                            border_sad_mul * step_multiplier,
                            step_multiplier,
                            step_multiplier,
                            step_multiplier,
                            step_multiplier,
                            step_multiplier,
                            step_multiplier,
                            border_sad_mul * step_multiplier,
                        ]
                    };

                    for x8 in 1..width / 8 {
                        let base_x = x8 * 8;
                        let base_idx = (y + 3) * width + base_x;

                        let Some(&sigma_val) = sigma_grid.get(x8 - 1, y8) else { break; };
                        if sigma_val < 0.3 {
                            for (input, ch) in input.iter().zip(output.iter_mut()) {
                                let input_ch = input.buf();
                                let output_ch = ch.buf_mut();
                                output_ch[base_idx..][..8].copy_from_slice(&input_ch[base_idx..][..8]);
                            }
                            continue;
                        }

                        for (dx, sm) in sm.into_iter().enumerate() {
                            let base_idx = base_idx + dx;

                            let mut sum_weights = 1.0f32;
                            let mut sum_channels = [0.0f32; 3];
                            for (sum, ch) in sum_channels.iter_mut().zip(input) {
                                let ch = ch.buf();
                                *sum = ch[base_idx];
                            }

                            for offset in $kernel_offsets {
                                let kernel_idx = base_idx.wrapping_add_signed(offset);
                                let mut dist = 0.0f32;
                                for (ch, scale) in input.iter().zip(channel_scale) {
                                    let ch = ch.buf();
                                    for offset in $dist_offsets {
                                        let base_idx = base_idx.wrapping_add_signed(offset);
                                        let kernel_idx = kernel_idx.wrapping_add_signed(offset);
                                        dist = scale.mul_add((ch[base_idx] - ch[kernel_idx]).abs(), dist);
                                    }
                                }

                                let weight = weight(
                                    dist,
                                    sigma_val,
                                    sm,
                                );
                                sum_weights += weight;

                                for (sum, ch) in sum_channels.iter_mut().zip(input) {
                                    let ch = ch.buf();
                                    *sum = weight.mul_add(ch[kernel_idx], *sum);
                                }
                            }

                            for (sum, ch) in sum_channels.into_iter().zip(output.iter_mut()) {
                                let ch = ch.buf_mut();
                                ch[base_idx] = sum / sum_weights;
                            }
                        }
                    }
                }
            }
        )*
    };
}

define_epf! {
    pub fn epf_step0(
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
    pub fn epf_step1(
        width,
        [-width, -1, 1, width],
        [-width, -1, 0, 1, width],
    );
    pub fn epf_step2(
        width,
        [-width, -1, 1, width],
        [0isize],
    );
}

pub fn apply_gabor_like(fb: [&mut SimpleGrid<f32>; 3], weights_xyb: [[f32; 2]; 3]) {
    for (fb, [weight1, weight2]) in fb.into_iter().zip(weights_xyb) {
        run_gabor_inner(fb, weight1, weight2);
    }
}

#[inline(always)]
pub fn run_gabor_inner(fb: &mut jxl_grid::SimpleGrid<f32>, weight1: f32, weight2: f32) {
    let global_weight = (1.0 + weight1 * 4.0 + weight2 * 4.0).recip();

    let width = fb.width();
    let height = fb.height();
    if width * height <= 1 {
        return;
    }

    let mut input = vec![0f32; width * (height + 2)];
    input[width..][..width * height].copy_from_slice(fb.buf());
    input[..width].copy_from_slice(&fb.buf()[..width]);
    input[width * (height + 1)..][..width].copy_from_slice(&fb.buf()[width * (height - 1)..][..width]);

    let input = &*input;
    let output = fb.buf_mut();

    if width == 1 {
        for idx in 0..height {
            output[idx] = (
                input[idx + 1] +
                (input[idx] + input[idx + 1] + input[idx + 1] + input[idx + 2]) * weight1 +
                (input[idx] + input[idx + 2]) * weight2 * 2.0
            ) * global_weight;
        }
        return;
    }

    let len = width * height - 2;
    let center = &input[width + 1..][..len];
    let sides = [
        &input[1..][..len],
        &input[width..][..len],
        &input[width + 2..][..len],
        &input[width * 2 + 1..][..len],
    ];
    let diags = [
        &input[..len],
        &input[2..][..len],
        &input[width * 2..][..len],
        &input[width * 2 + 2..][..len],
    ];

    for (idx, out) in output[1..][..len].iter_mut().enumerate() {
        *out = (
            center[idx] +
            (sides[0][idx] + sides[1][idx] + sides[2][idx] + sides[3][idx]) * weight1 +
            (diags[0][idx] + diags[1][idx] + diags[2][idx] + diags[3][idx]) * weight2
        ) * global_weight;
    }

    // left side
    let center = &input[width..];
    let sides = [
        input,
        &input[width..],
        &input[width + 1..],
        &input[width * 2..],
    ];
    let diags = [
        input,
        &input[1..],
        &input[width * 2..],
        &input[width * 2 + 1..],
    ];
    for idx in 0..height {
        let offset = idx * width;
        output[offset] = (
            center[offset] +
            (sides[0][offset] + sides[1][offset] + sides[2][offset] + sides[3][offset]) * weight1 +
            (diags[0][offset] + diags[1][offset] + diags[2][offset] + diags[3][offset]) * weight2
        ) * global_weight;
    }

    // right side
    let center = &input[width * 2 - 1..];
    let sides = [
        &input[width - 1..],
        &input[width * 2 - 2..],
        &input[width * 2 - 1..],
        &input[width * 3 - 1..],
    ];
    let diags = [
        &input[width - 2..],
        &input[width - 1..],
        &input[width * 3 - 2..],
        &input[width * 3 - 1..],
    ];
    for idx in 0..height {
        let offset = idx * width;
        output[width - 1 + offset] = (
            center[offset] +
            (sides[0][offset] + sides[1][offset] + sides[2][offset] + sides[3][offset]) * weight1 +
            (diags[0][offset] + diags[1][offset] + diags[2][offset] + diags[3][offset]) * weight2
        ) * global_weight;
    }
}
