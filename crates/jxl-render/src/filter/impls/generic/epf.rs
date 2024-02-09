use crate::{filter::impls::common, util};

use super::EpfRow;

pub(crate) fn epf_row<const STEP: usize>(epf_row: EpfRow<'_, '_>) {
    let EpfRow {
        input_rows,
        output_rows,
        width,
        x,
        y,
        sigma_row,
        epf_params,
        skip_inner,
        ..
    } = epf_row;
    let kernel_offsets = common::epf_kernel_offsets::<STEP>();
    let dist_offsets = common::epf_dist_offsets::<STEP>();

    let step_multiplier = if STEP == 0 {
        epf_params.sigma.pass0_sigma_scale
    } else if STEP == 2 {
        epf_params.sigma.pass2_sigma_scale
    } else {
        1.0
    };
    let border_sad_mul = epf_params.sigma.border_sad_mul;
    let channel_scale = epf_params.channel_scale;

    let is_y_border = (y + 1) & 0b110 == 0;
    let sm = if is_y_border {
        [step_multiplier * border_sad_mul; 8]
    } else {
        let neg_x = 8 - (x & 7);
        let mut sm = [step_multiplier; 8];
        sm[neg_x & 7] *= border_sad_mul;
        sm[(neg_x + 7) & 7] *= border_sad_mul;
        sm
    };

    let padding = 3 - STEP;
    let (left_edge_width, right_edge_width) = if width < padding * 2 {
        let left_edge_width = width.saturating_sub(padding);
        (left_edge_width, width - left_edge_width)
    } else {
        (padding, padding)
    };

    let simd_range = {
        let start = (x + left_edge_width + 7) & !7;
        let end = (x + width - right_edge_width) & !7;
        if start > end {
            let start = start - x;
            start..start
        } else {
            let start = start - x;
            let end = end - x;
            start..end
        }
    };

    let (left_padding_end, right_padding_start) = if skip_inner {
        (simd_range.start, simd_range.end)
    } else {
        (left_edge_width, width.saturating_sub(padding))
    };

    for dx in 0..left_padding_end {
        let sm_idx = dx & 7;
        let sigma_x = (x + dx) / 8 - x / 8;
        let sigma_val = sigma_row[sigma_x];
        if sigma_val < 0.3 {
            for c in 0..3 {
                output_rows[c][dx] = input_rows[c][3][dx];
            }
            continue;
        }

        let mut sum_weights = 1.0f32;
        let mut sum_channels: [f32; 3] = std::array::from_fn(|c| input_rows[c][3][dx]);

        for &(kx, ky) in kernel_offsets {
            let kernel_dy = 3 + ky;
            let kernel_dx = dx as isize + kx;
            let mut dist = 0f32;
            for c in 0..3 {
                let scale = channel_scale[c];
                let mut acc = 0f32;
                for &(ix, iy) in dist_offsets {
                    let kernel_dy = (kernel_dy + iy) as usize;
                    let kernel_dx = util::mirror(kernel_dx + ix, width);
                    let base_dy = (3 + iy) as usize;
                    let base_dx = util::mirror(dx as isize + ix, width);

                    acc += (input_rows[c][kernel_dy][kernel_dx] - input_rows[c][base_dy][base_dx])
                        .abs();
                }
                dist += scale * acc;
            }

            let weight = weight(dist, sigma_val, sm[sm_idx]);
            sum_weights += weight;

            let kernel_dy = kernel_dy as usize;
            let kernel_dx = util::mirror(kernel_dx, width);
            for (c, sum) in sum_channels.iter_mut().enumerate() {
                *sum += weight * input_rows[c][kernel_dy][kernel_dx];
            }
        }

        for (c, sum) in sum_channels.into_iter().enumerate() {
            output_rows[c][dx] = sum / sum_weights;
        }
    }

    if !skip_inner {
        for dx in left_padding_end..right_padding_start {
            let sm_idx = dx & 7;
            let sigma_x = (x + dx) / 8 - x / 8;
            let sigma_val = sigma_row[sigma_x];
            if sigma_val < 0.3 {
                for c in 0..3 {
                    output_rows[c][dx] = input_rows[c][3][dx];
                }
                continue;
            }

            let mut sum_weights = 1.0f32;
            let mut sum_channels: [f32; 3] = std::array::from_fn(|c| input_rows[c][3][dx]);

            for &(kx, ky) in kernel_offsets {
                let kernel_dy = 3 + ky;
                let kernel_dx = dx as isize + kx;
                let mut dist = 0f32;
                for c in 0..3 {
                    let scale = channel_scale[c];
                    let mut acc = 0f32;
                    for &(ix, iy) in dist_offsets {
                        let kernel_dy = (kernel_dy + iy) as usize;
                        let kernel_dx = (kernel_dx + ix) as usize;
                        let base_dy = (3 + iy) as usize;
                        let base_dx = (dx as isize + ix) as usize;

                        acc += (input_rows[c][kernel_dy][kernel_dx]
                            - input_rows[c][base_dy][base_dx])
                            .abs();
                    }
                    dist += scale * acc;
                }

                let weight = weight(dist, sigma_val, sm[sm_idx]);
                sum_weights += weight;

                let kernel_dy = kernel_dy as usize;
                let kernel_dx = kernel_dx as usize;
                for (c, sum) in sum_channels.iter_mut().enumerate() {
                    *sum += weight * input_rows[c][kernel_dy][kernel_dx];
                }
            }

            for (c, sum) in sum_channels.into_iter().enumerate() {
                output_rows[c][dx] = sum / sum_weights;
            }
        }
    }

    for dx in right_padding_start..width {
        let sm_idx = dx & 7;
        let sigma_x = (x + dx) / 8 - x / 8;
        let sigma_val = sigma_row[sigma_x];
        if sigma_val < 0.3 {
            for c in 0..3 {
                output_rows[c][dx] = input_rows[c][3][dx];
            }
            continue;
        }

        let mut sum_weights = 1.0f32;
        let mut sum_channels: [f32; 3] = std::array::from_fn(|c| input_rows[c][3][dx]);

        for &(kx, ky) in kernel_offsets {
            let kernel_dy = 3 + ky;
            let kernel_dx = dx as isize + kx;
            let mut dist = 0f32;
            for c in 0..3 {
                let scale = channel_scale[c];
                let mut acc = 0f32;
                for &(ix, iy) in dist_offsets {
                    let kernel_dy = (kernel_dy + iy) as usize;
                    let kernel_dx = util::mirror(kernel_dx + ix, width);
                    let base_dy = (3 + iy) as usize;
                    let base_dx = util::mirror(dx as isize + ix, width);

                    acc += (input_rows[c][kernel_dy][kernel_dx] - input_rows[c][base_dy][base_dx])
                        .abs();
                }
                dist += scale * acc;
            }

            let weight = weight(dist, sigma_val, sm[sm_idx]);
            sum_weights += weight;

            let kernel_dy = kernel_dy as usize;
            let kernel_dx = util::mirror(kernel_dx, width);
            for (c, sum) in sum_channels.iter_mut().enumerate() {
                *sum += weight * input_rows[c][kernel_dy][kernel_dx];
            }
        }

        for (c, sum) in sum_channels.into_iter().enumerate() {
            output_rows[c][dx] = sum / sum_weights;
        }
    }
}

#[inline]
fn weight(scaled_distance: f32, sigma: f32, step_multiplier: f32) -> f32 {
    let neg_inv_sigma = 6.6 * (std::f32::consts::FRAC_1_SQRT_2 - 1.0) / sigma * step_multiplier;
    (1.0 + scaled_distance * neg_inv_sigma).max(0.0)
}
