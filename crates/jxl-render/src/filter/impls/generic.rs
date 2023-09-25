use jxl_grid::SimpleGrid;

#[inline]
fn weight(scaled_distance: f32, sigma: f32, step_multiplier: f32) -> f32 {
    let inv_sigma = step_multiplier * 6.6 * (1.0 - std::f32::consts::FRAC_1_SQRT_2) / sigma;
    (1.0 - scaled_distance * inv_sigma).max(0.0)
}

#[allow(clippy::too_many_arguments)]
fn epf_step(
    input: &[SimpleGrid<f32>; 3],
    output: &mut [SimpleGrid<f32>; 3],
    sigma_grid: &SimpleGrid<f32>,
    channel_scale: [f32; 3],
    border_sad_mul: f32,
    step_multiplier: f32,
    kernel_coords: &'static [(isize, isize)],
    dist_coords: &'static [(isize, isize)],
) {
    let width = input[0].width();
    let height = input[0].height();
    for y in 0..height - 7 {
        let y8 = y / 8;
        let is_y_border = (y % 8) == 0 || (y % 8) == 7;
        let y = y + 3;

        for x in 0..width - 6 {
            let x8 = x / 8;
            let is_border = is_y_border || (x % 8) == 0 || (x % 8) == 7;
            let x = x + 3;

            let sigma_val = *sigma_grid.get(x8, y8).unwrap();
            if sigma_val < 0.3 {
                for (input, ch) in input.iter().zip(output.iter_mut()) {
                    let input_ch = input.buf();
                    let output_ch = ch.buf_mut();
                    output_ch[y * width + x] = input_ch[y * width + x];
                }
                continue;
            }

            let mut sum_weights = 1.0f32;
            let mut sum_channels = [0.0f32; 3];
            for (sum, ch) in sum_channels.iter_mut().zip(input) {
                let ch = ch.buf();
                *sum = ch[y * width + x];
            }

            for &(dx, dy) in kernel_coords {
                let tx = x as isize + dx;
                let ty = y as isize + dy;
                let mut dist = 0.0f32;
                for (ch, scale) in input.iter().zip(channel_scale) {
                    let ch = ch.buf();
                    for &(dx, dy) in dist_coords {
                        let x = x as isize + dx;
                        let y = y as isize + dy;
                        let tx = (tx + dx) as usize;
                        let ty = (ty + dy) as usize;
                        let x = x as usize;
                        let y = y as usize;
                        dist += (ch[y * width + x] - ch[ty * width + tx]).abs() * scale;
                    }
                }

                let weight = weight(
                    dist,
                    sigma_val,
                    step_multiplier * if is_border { border_sad_mul } else { 1.0 },
                );
                sum_weights += weight;

                let tx = tx as usize;
                let ty = ty as usize;
                for (sum, ch) in sum_channels.iter_mut().zip(input) {
                    let ch = ch.buf();
                    *sum += ch[ty * width + tx] * weight;
                }
            }

            for (sum, ch) in sum_channels.into_iter().zip(output.iter_mut()) {
                let ch = ch.buf_mut();
                ch[y * width + x] = sum / sum_weights;
            }
        }
    }
}

pub fn epf_step0(
    input: &[SimpleGrid<f32>; 3],
    output: &mut [SimpleGrid<f32>; 3],
    sigma_grid: &SimpleGrid<f32>,
    channel_scale: [f32; 3],
    border_sad_mul: f32,
    step_multiplier: f32,
) {
    epf_step(
        &fb_in,
        &mut fb_out,
        sigma_grid,
        channel_scale,
        sigma.border_sad_mul,
        sigma.pass0_sigma_scale,
        &[
            (0, -1), (-1, 0), (1, 0), (0, 1),
            (0, -2), (-1, -1), (1, -1), (-2, 0), (2, 0), (-1, 1), (1, 1), (0, 2),
        ],
        &[(0, 0), (0, -1), (-1, 0), (1, 0), (0, 1)],
    );
}

pub fn epf_step1(
    input: &[SimpleGrid<f32>; 3],
    output: &mut [SimpleGrid<f32>; 3],
    sigma_grid: &SimpleGrid<f32>,
    channel_scale: [f32; 3],
    border_sad_mul: f32,
    step_multiplier: f32,
) {
    epf_step(
        &fb_in,
        &mut fb_out,
        sigma_grid,
        channel_scale,
        sigma.border_sad_mul,
        sigma.pass0_sigma_scale,
        &[(0, -1), (-1, 0), (1, 0), (0, 1)],
        &[(0, 0), (0, -1), (-1, 0), (1, 0), (0, 1)],
    );
}

pub fn epf_step2(
    input: &[SimpleGrid<f32>; 3],
    output: &mut [SimpleGrid<f32>; 3],
    sigma_grid: &SimpleGrid<f32>,
    channel_scale: [f32; 3],
    border_sad_mul: f32,
    step_multiplier: f32,
) {
    epf_step(
        &fb_in,
        &mut fb_out,
        sigma_grid,
        channel_scale,
        sigma.border_sad_mul,
        sigma.pass0_sigma_scale,
        &[(0, -1), (-1, 0), (1, 0), (0, 1)],
        &[(0, 0)],
    );
}

pub fn apply_gabor_like(fb: [&mut SimpleGrid<f32>; 3], weights_xyb: [[f32; 2]; 3]) {
    for (fb, [weight1, weight2]) in fb.into_iter().zip(weights_xyb) {
        super::run_gabor_inner(fb, weight1, weight2);
    }
}
