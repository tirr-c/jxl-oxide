use std::collections::HashMap;

use jxl_frame::{data::LfGroup, filter::EdgePreservingFilter, FrameHeader};
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
    for y in 3..height - 3 {
        let y8 = y / 8;
        let is_y_border = (y % 8) == 0 || (y % 8) == 7;

        for x in 3..width - 3 {
            let x8 = x / 8;
            let is_x_border = (x % 8) == 0 || (x % 8) == 7;
            let is_border = is_y_border || is_x_border;

            let sigma_val = *sigma_grid.get(x8, y8).unwrap();
            if sigma_val < 0.3 {
                input.iter().zip(output.iter_mut()).for_each(|(input_ch, output_ch)| {
                    let input_buf = input_ch.buf();
                    let output_buf = output_ch.buf_mut();
                    let index = y * width + x;
                    output_buf[index] = input_buf[index];
                });
                continue;
            }

            let mut sum_weights = 1.0f32;
            let mut sum_channels = [0.0f32; 3];
            sum_channels.iter_mut().zip(input.iter()).for_each(|(sum, ch)| {
                let ch = ch.buf();
                *sum = ch[y * width + x];
            });

            for &(dx, dy) in kernel_coords {
                let tx = x as isize + dx;
                let ty = y as isize + dy;
                let mut dist = 0.0f32;
                input.iter().zip(channel_scale.iter()).for_each(|(ch, scale)| {
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
                });

                let weight = weight(
                    dist,
                    sigma_val,
                    step_multiplier * if is_border { border_sad_mul } else { 1.0 },
                );
                sum_weights += weight;

                let tx = tx as usize;
                let ty = ty as usize;
                sum_channels.iter_mut().zip(input.iter()).for_each(|(sum, ch)| {
                    let ch = ch.buf();
                    *sum += ch[ty * width + tx] * weight;
                });
            }

            sum_channels.into_iter().zip(output.iter_mut()).for_each(|(sum, ch)| {
                let ch = ch.buf_mut();
                ch[y * width + x] = sum / sum_weights;
            });
        }
    }
}

pub fn apply_epf(
    fb: [&mut SimpleGrid<f32>; 3],
    lf_groups: &HashMap<u32, LfGroup>,
    frame_header: &FrameHeader,
) {
    let EdgePreservingFilter::Enabled {
        iters,
        channel_scale,
        ref sigma,
        sigma_for_modular,
        ..
    } = frame_header.restoration_filter.epf else { return; };

    let span = tracing::span!(tracing::Level::TRACE, "apply_epf");
    let _guard = span.enter();

    let width = fb[0].width();
    let height = fb[0].height();
    let mut fb_in = [
        SimpleGrid::new(width + 6, height + 6),
        SimpleGrid::new(width + 6, height + 6),
        SimpleGrid::new(width + 6, height + 6),
    ];
    let mut fb_out = [
        SimpleGrid::new(width + 6, height + 6),
        SimpleGrid::new(width + 6, height + 6),
        SimpleGrid::new(width + 6, height + 6),
    ];
    for (output, input) in fb_in.iter_mut().zip(&fb) {
        let output = output.buf_mut();
        let input = input.buf();
        for y in 0..height {
            output[(y + 3) * (width + 6) + 3..][..width].copy_from_slice(&input[y * width..][..width]);
        }
    }

    tracing::debug!("Preparing sigma grid");
    let w8 = (width + 7) / 8;
    let h8 = (height + 7) / 8;
    let mut sigma_grid = SimpleGrid::new(w8, h8);
    let mut need_sigma_init = true;

    let lf_groups_per_row = frame_header.lf_groups_per_row();
    let lf_group_dim8 = frame_header.group_dim();
    for (&lf_group_idx, lf_group) in lf_groups {
        let base_x = ((lf_group_idx % lf_groups_per_row) * lf_group_dim8) as usize;
        let base_y = ((lf_group_idx / lf_groups_per_row) * lf_group_dim8) as usize;
        if let Some(hf_meta) = &lf_group.hf_meta {
            need_sigma_init = false;
            let epf_sigma = &hf_meta.epf_sigma;
            for y8 in 0..epf_sigma.height() {
                for x8 in 0..epf_sigma.width() {
                    let sigma = *epf_sigma.get(x8, y8).unwrap();
                    *sigma_grid.get_mut(base_x + x8, base_y + y8).unwrap() = sigma;
                }
            }
        }
    }
    if need_sigma_init {
        for sigma in sigma_grid.buf_mut() {
            *sigma = sigma_for_modular;
        }
    }

    // Step 0
    if iters == 3 {
        tracing::debug!("Running step 0");
        for output in &mut fb_in {
            let output = output.buf_mut();

            for y in 3..height + 3 {
                output[y * (width + 6)] = output[y * (width + 6) + 5];
                output[y * (width + 6) + 1] = output[y * (width + 6) + 4];
                output[y * (width + 6) + 2] = output[y * (width + 6) + 3];
                output[(y + 1) * (width + 6) - 3] = output[(y + 1) * (width + 6) - 4];
                output[(y + 1) * (width + 6) - 2] = output[(y + 1) * (width + 6) - 5];
                output[(y + 1) * (width + 6) - 1] = output[(y + 1) * (width + 6) - 6];
            }

            let (out_chunk, in_chunk) = output.split_at_mut((width + 6) * 3);
            let in_chunk = &in_chunk[..(width + 6) * 3];
            for (out_row, in_row) in out_chunk.chunks_exact_mut(width + 6).zip(in_chunk.chunks_exact(width + 6).rev()) {
                out_row.copy_from_slice(in_row);
            }

            let (in_chunk, out_chunk) = output.split_at_mut((width + 6) * (height + 3));
            for (out_row, in_row) in out_chunk.chunks_exact_mut(width + 6).zip(in_chunk.chunks_exact(width + 6).rev()) {
                out_row.copy_from_slice(in_row);
            }
        }

        epf_step(
            &fb_in,
            &mut fb_out,
            &sigma_grid,
            channel_scale,
            sigma.border_sad_mul,
            sigma.pass0_sigma_scale,
            &[
                (0, -1), (-1, 0), (1, 0), (0, 1),
                (0, -2), (-1, -1), (1, -1), (-2, 0), (2, 0), (-1, 1), (1, 1), (0, 2),
            ],
            &[(0, 0), (0, -1), (-1, 0), (1, 0), (0, 1)],
        );
        std::mem::swap(&mut fb_in, &mut fb_out);
    }

    // Step 1
    {
        tracing::debug!("Running step 1");
        for output in &mut fb_in {
            let output = output.buf_mut();

            for y in 3..height + 3 {
                output[y * (width + 6)] = output[y * (width + 6) + 5];
                output[y * (width + 6) + 1] = output[y * (width + 6) + 4];
                output[y * (width + 6) + 2] = output[y * (width + 6) + 3];
                output[(y + 1) * (width + 6) - 3] = output[(y + 1) * (width + 6) - 4];
                output[(y + 1) * (width + 6) - 2] = output[(y + 1) * (width + 6) - 5];
                output[(y + 1) * (width + 6) - 1] = output[(y + 1) * (width + 6) - 6];
            }

            let (out_chunk, in_chunk) = output.split_at_mut((width + 6) * 3);
            let in_chunk = &in_chunk[..(width + 6) * 3];
            for (out_row, in_row) in out_chunk.chunks_exact_mut(width + 6).zip(in_chunk.chunks_exact(width + 6).rev()) {
                out_row.copy_from_slice(in_row);
            }

            let (in_chunk, out_chunk) = output.split_at_mut((width + 6) * (height + 3));
            for (out_row, in_row) in out_chunk.chunks_exact_mut(width + 6).zip(in_chunk.chunks_exact(width + 6).rev()) {
                out_row.copy_from_slice(in_row);
            }
        }

        epf_step(
            &fb_in,
            &mut fb_out,
            &sigma_grid,
            channel_scale,
            sigma.border_sad_mul,
            1.0,
            &[(0, -1), (-1, 0), (1, 0), (0, 1)],
            &[(0, 0), (0, -1), (-1, 0), (1, 0), (0, 1)],
        );
        std::mem::swap(&mut fb_in, &mut fb_out);
    }

    // Step 2
    if iters >= 2 {
        tracing::debug!("Running step 2");
        for output in &mut fb_in {
            let output = output.buf_mut();

            for y in 3..height + 3 {
                output[y * (width + 6)] = output[y * (width + 6) + 5];
                output[y * (width + 6) + 1] = output[y * (width + 6) + 4];
                output[y * (width + 6) + 2] = output[y * (width + 6) + 3];
                output[(y + 1) * (width + 6) - 3] = output[(y + 1) * (width + 6) - 4];
                output[(y + 1) * (width + 6) - 2] = output[(y + 1) * (width + 6) - 5];
                output[(y + 1) * (width + 6) - 1] = output[(y + 1) * (width + 6) - 6];
            }

            let (out_chunk, in_chunk) = output.split_at_mut((width + 6) * 3);
            let in_chunk = &in_chunk[..(width + 6) * 3];
            for (out_row, in_row) in out_chunk.chunks_exact_mut(width + 6).zip(in_chunk.chunks_exact(width + 6).rev()) {
                out_row.copy_from_slice(in_row);
            }

            let (in_chunk, out_chunk) = output.split_at_mut((width + 6) * (height + 3));
            for (out_row, in_row) in out_chunk.chunks_exact_mut(width + 6).zip(in_chunk.chunks_exact(width + 6).rev()) {
                out_row.copy_from_slice(in_row);
            }
        }

        epf_step(
            &fb_in,
            &mut fb_out,
            &sigma_grid,
            channel_scale,
            sigma.border_sad_mul,
            sigma.pass2_sigma_scale,
            &[(0, -1), (-1, 0), (1, 0), (0, 1)],
            &[(0, 0)],
        );
        std::mem::swap(&mut fb_in, &mut fb_out);
    }

    for (output, input) in fb.into_iter().zip(fb_in) {
        let output = output.buf_mut();
        let input = input.buf();
        for y in 0..height {
            output[y * width..][..width].copy_from_slice(&input[(y + 3) * (width + 6) + 3..][..width]);
        }
    }
}
