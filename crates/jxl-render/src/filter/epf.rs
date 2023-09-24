use std::collections::HashMap;

use jxl_frame::{data::LfGroup, filter::EdgePreservingFilter, FrameHeader};
use jxl_grid::SimpleGrid;

use crate::{region::ImageWithRegion, Region};

pub fn apply_epf(
    fb: &mut ImageWithRegion,
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

    let region = fb.region();
    let fb = fb.buffer_mut();
    assert!(region.left % 8 == 0);
    assert!(region.top % 8 == 0);

    let width = region.width as usize;
    let height = region.height as usize;
    // Extra padding for SIMD
    let mut fb_in = [
        SimpleGrid::new(width + 6, height + 7),
        SimpleGrid::new(width + 6, height + 7),
        SimpleGrid::new(width + 6, height + 7),
    ];
    let mut fb_out = [
        SimpleGrid::new(width + 6, height + 7),
        SimpleGrid::new(width + 6, height + 7),
        SimpleGrid::new(width + 6, height + 7),
    ];
    for (output, input) in fb_in.iter_mut().zip(&*fb) {
        let output = output.buf_mut();
        let input = input.buf();
        for y in 0..height {
            output[(y + 3) * (width + 6) + 3..][..width].copy_from_slice(&input[y * width..][..width]);
        }
    }

    tracing::debug!("Preparing sigma grid");
    let sigma_region = region.downsample(3);
    let mut sigma_image = ImageWithRegion::from_region(1, sigma_region);
    let sigma_grid = &mut sigma_image.buffer_mut()[0];
    let mut need_sigma_init = true;

    let lf_groups_per_row = frame_header.lf_groups_per_row();
    let lf_group_dim8 = frame_header.group_dim();
    for (&lf_group_idx, lf_group) in lf_groups {
        let base_x = ((lf_group_idx % lf_groups_per_row) * lf_group_dim8) as usize;
        let base_y = ((lf_group_idx / lf_groups_per_row) * lf_group_dim8) as usize;
        let lf_region = Region {
            left: base_x as i32,
            top: base_y as i32,
            width: lf_group_dim8,
            height: lf_group_dim8,
        };
        let intersection = sigma_region.intersection(lf_region);
        if intersection.is_empty() {
            continue;
        }

        if let Some(hf_meta) = &lf_group.hf_meta {
            need_sigma_init = false;
            let epf_sigma = &hf_meta.epf_sigma;

            let lf_region = intersection.translate(-lf_region.left, -lf_region.top);
            let sigma_region = intersection.translate(-sigma_region.left, -sigma_region.top);
            for y8 in 0..lf_region.height as usize {
                let lf_y = lf_region.top as usize + y8;
                let sigma_y = sigma_region.top as usize + y8;
                for x8 in 0..lf_region.width as usize {
                    let lf_x = lf_region.left as usize + x8;
                    let sigma_x = sigma_region.left as usize + x8;
                    *sigma_grid.get_mut(sigma_x, sigma_y).unwrap() = *epf_sigma.get(lf_x, lf_y).unwrap();
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

        #[cfg(target_arch = "x86_64")]
        {
            super::x86_64::epf_step0_sse2(
                &fb_in,
                &mut fb_out,
                sigma_grid,
                channel_scale,
                sigma.border_sad_mul,
                sigma.pass0_sigma_scale,
            );
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            super::generic::epf_step(
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

        #[cfg(target_arch = "x86_64")]
        {
            super::x86_64::epf_step1_sse2(
                &fb_in,
                &mut fb_out,
                sigma_grid,
                channel_scale,
                sigma.border_sad_mul,
                1.0,
            );
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            super::generic::epf_step(
                &fb_in,
                &mut fb_out,
                sigma_grid,
                channel_scale,
                sigma.border_sad_mul,
                1.0,
                &[(0, -1), (-1, 0), (1, 0), (0, 1)],
                &[(0, 0), (0, -1), (-1, 0), (1, 0), (0, 1)],
            );
        }
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

        #[cfg(target_arch = "x86_64")]
        {
            super::x86_64::epf_step2_sse2(
                &fb_in,
                &mut fb_out,
                sigma_grid,
                channel_scale,
                sigma.border_sad_mul,
                sigma.pass2_sigma_scale,
            );
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            super::generic::epf_step(
                &fb_in,
                &mut fb_out,
                sigma_grid,
                channel_scale,
                sigma.border_sad_mul,
                sigma.pass2_sigma_scale,
                &[(0, -1), (-1, 0), (1, 0), (0, 1)],
                &[(0, 0)],
            );
        }
        std::mem::swap(&mut fb_in, &mut fb_out);
    }

    for (output, input) in fb.iter_mut().zip(fb_in) {
        let output = output.buf_mut();
        let input = input.buf();
        for y in 0..height {
            output[y * width..][..width].copy_from_slice(&input[(y + 3) * (width + 6) + 3..][..width]);
        }
    }
}
