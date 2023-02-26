use std::collections::BTreeMap;

use jxl_frame::{data::LfGroup, filter::EdgePreservingFilter, FrameHeader};
use jxl_grid::SimpleGrid;

use crate::FrameBuffer;

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

pub fn apply_epf(
    fb: &mut FrameBuffer,
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

    tracing::debug!("Preparing bitmap");
    let mut out_fb = fb.clone();
    let width = fb.width() as usize;
    let height = fb.height() as usize;
    let w8 = (width + 7) / 8;
    let h8 = (height + 7) / 8;
    let mut bitmap = SimpleGrid::new(w8, h8);
    let mut sigma_grid = SimpleGrid::new(w8, h8);
    let mut need_bitmap_init = true;

    let lf_groups_per_row = frame_header.lf_groups_per_row();
    let lf_group_dim8 = frame_header.group_dim();
    for (&lf_group_idx, lf_group) in lf_groups {
        let base_x = ((lf_group_idx % lf_groups_per_row) * lf_group_dim8) as usize;
        let base_y = ((lf_group_idx / lf_groups_per_row) * lf_group_dim8) as usize;
        if let Some(hf_meta) = &lf_group.hf_meta {
            need_bitmap_init = false;
            let epf_sigma = &hf_meta.epf_sigma;
            for y8 in 0..epf_sigma.height() {
                for x8 in 0..epf_sigma.width() {
                    let sigma = *epf_sigma.get(x8, y8).unwrap();
                    *sigma_grid.get_mut(base_x + x8, base_y + y8).unwrap() = sigma;
                    *bitmap.get_mut(base_x + x8, base_y + y8).unwrap() = sigma >= 0.3;
                }
            }
        }
    }
    if need_bitmap_init {
        for bitmap in bitmap.buf_mut() {
            *bitmap = true;
        }
        for sigma in sigma_grid.buf_mut() {
            *sigma = sigma_for_modular;
        }
    }

    // Step 0
    if iters == 3 {
        tracing::debug!("Running step 0");
        todo!();
    }

    // Step 1
    {
        tracing::debug!("Running step 1");
        let step_multiplier = 1f32;

        let channels = &fb.channel_buffers()[..3];
        let mut out_channels = out_fb.channel_buffers_mut();
        for y in 0..height {
            let y8 = y / 8;
            let is_y_border = (y % 8) == 0 || (y % 8) == 7;

            for x in 0..width {
                let x8 = x / 8;
                if !*bitmap.get(x8, y8).unwrap() {
                    continue;
                }
                let is_border = is_y_border || (x % 8) == 0 || (x % 8) == 7;
                let sigma_val = *sigma_grid.get(x8, y8).unwrap();

                let mut sum_weights = weight(0.0f32, sigma_val, step_multiplier);
                let mut sum_channels = [0.0f32; 3];
                for (sum, &ch) in sum_channels.iter_mut().zip(channels) {
                    *sum = ch[y * width + x] * sum_weights;
                }

                for (dx, dy) in [(0isize, -1isize), (-1, 0), (1, 0), (0, 1)] {
                    let tx = x as isize + dx;
                    let ty = y as isize + dy;
                    let mut dist = 0.0f32;
                    for (&ch, scale) in channels.iter().zip(channel_scale) {
                        for (dx, dy) in [(0isize, 0isize), (0, -1), (-1, 0), (1, 0), (0, 1)] {
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
                        dist * if is_border { sigma.border_sad_mul } else { 1.0 },
                        sigma_val,
                        step_multiplier,
                    );
                    sum_weights += weight;
                    for (sum, &ch) in sum_channels.iter_mut().zip(channels) {
                        *sum += ch[y * width + x] * weight;
                    }
                }

                for (sum, ch) in sum_channels.into_iter().zip(&mut out_channels) {
                    ch[y * width + x] = sum / sum_weights;
                }
            }
        }

        std::mem::swap(fb, &mut out_fb);
    }

    // Step 2
    if iters >= 2 {
        tracing::debug!("Running step 2");
        todo!();
    }
}
