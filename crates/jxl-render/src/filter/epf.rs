use std::collections::BTreeMap;

use jxl_frame::{data::LfGroup, filter::EdgePreservingFilter, FrameHeader};
use jxl_grid::SimpleGrid;

fn mirror(x: isize, size: usize) -> usize {
    let mut x = if x < 0 { (x + 1).unsigned_abs() } else { x as usize };
    if x >= size {
        let tx = x % (size * 2);
        if tx >= size {
            x = size * 2 - tx - 1;
        }
    }
    x
}

fn weight(scaled_distance: f32, sigma: f32, step_multiplier: f32) -> f32 {
    let inv_sigma = step_multiplier * 6.6 * (std::f32::consts::FRAC_1_SQRT_2 - 1.0) / sigma;
    (1.0 - scaled_distance * inv_sigma).max(0.0)
}

#[allow(clippy::too_many_arguments)]
fn epf_step(
    input: [&SimpleGrid<f32>; 3],
    mut output: [&mut SimpleGrid<f32>; 3],
    sigma_grid: &SimpleGrid<f32>,
    channel_scale: [f32; 3],
    border_sad_mul: f32,
    step_multiplier: f32,
    kernel_coords: &'static [(isize, isize)],
    dist_coords: &'static [(isize, isize)],
) {
    let width = input[0].width();
    let height = input[0].height();
    for y in 0..height {
        let y8 = y / 8;
        let is_y_border = (y % 8) == 0 || (y % 8) == 7;

        for x in 0..width {
            let x8 = x / 8;
            let sigma_val = *sigma_grid.get(x8, y8).unwrap();
            if sigma_val < 0.3 {
                continue;
            }
            let is_border = is_y_border || (x % 8) == 0 || (x % 8) == 7;

            let mut sum_weights = weight(0.0f32, sigma_val, step_multiplier);
            let mut sum_channels = [0.0f32; 3];
            for (sum, ch) in sum_channels.iter_mut().zip(input) {
                let ch = ch.buf();
                *sum = ch[y * width + x] * sum_weights;
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
                        let tx = tx + dx;
                        let ty = ty + dy;
                        let x = mirror(x, width);
                        let y = mirror(y, height);
                        let tx = mirror(tx, width);
                        let ty = mirror(ty, height);
                        dist += (ch[y * width + x] - ch[ty * width + tx]).abs() * scale;
                    }
                }

                let weight = weight(
                    dist * if is_border { border_sad_mul } else { 1.0 },
                    sigma_val,
                    step_multiplier,
                );
                sum_weights += weight;
                for (sum, ch) in sum_channels.iter_mut().zip(input) {
                    let ch = ch.buf();
                    *sum += ch[y * width + x] * weight;
                }
            }

            for (sum, ch) in sum_channels.into_iter().zip(&mut output) {
                let ch = ch.buf_mut();
                ch[y * width + x] = sum / sum_weights;
            }
        }
    }
}

pub fn apply_epf(
    mut fb: [&mut SimpleGrid<f32>; 3],
    lf_groups: &BTreeMap<u32, LfGroup>,
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

    tracing::debug!("Preparing sigma grid");
    let mut out_fb = [
        fb[0].clone(),
        fb[1].clone(),
        fb[2].clone(),
    ];
    let width = fb[0].width();
    let height = fb[0].height();
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
        let old = [
            &*fb[0],
            &*fb[1],
            &*fb[2],
        ];
        let new = {
            let [a, b, c] = &mut out_fb;
            [a, b, c]
        };
        epf_step(
            old,
            new,
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

        for (old, new) in fb.iter_mut().zip(&mut out_fb) {
            std::mem::swap(*old, new);
        }
    }

    // Step 1
    {
        tracing::debug!("Running step 1");
        let old = [
            &*fb[0],
            &*fb[1],
            &*fb[2],
        ];
        let new = {
            let [a, b, c] = &mut out_fb;
            [a, b, c]
        };
        epf_step(
            old,
            new,
            &sigma_grid,
            channel_scale,
            sigma.border_sad_mul,
            1.0,
            &[(0, -1), (-1, 0), (1, 0), (0, 1)],
            &[(0, 0), (0, -1), (-1, 0), (1, 0), (0, 1)],
        );

        for (old, new) in fb.iter_mut().zip(&mut out_fb) {
            std::mem::swap(*old, new);
        }
    }

    // Step 2
    if iters >= 2 {
        tracing::debug!("Running step 2");
        let old = [
            &*fb[0],
            &*fb[1],
            &*fb[2],
        ];
        let new = {
            let [a, b, c] = &mut out_fb;
            [a, b, c]
        };
        epf_step(
            old,
            new,
            &sigma_grid,
            channel_scale,
            sigma.border_sad_mul,
            sigma.pass2_sigma_scale,
            &[(0, -1), (-1, 0), (1, 0), (0, 1)],
            &[(0, 0)],
        );

        for (old, new) in fb.iter_mut().zip(&mut out_fb) {
            std::mem::swap(*old, new);
        }
    }
}
