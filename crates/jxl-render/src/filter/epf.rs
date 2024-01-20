use std::collections::HashMap;

use jxl_frame::{data::LfGroup, filter::EdgePreservingFilter, FrameHeader};
use jxl_grid::SimpleGrid;
use jxl_modular::Sample;

use crate::{region::ImageWithRegion, Region};

pub fn apply_epf<S: Sample>(
    fb: &mut ImageWithRegion,
    lf_groups: &HashMap<u32, LfGroup<S>>,
    frame_header: &FrameHeader,
    pool: &jxl_threadpool::JxlThreadPool,
) -> crate::Result<()> {
    let EdgePreservingFilter::Enabled {
        iters,
        channel_scale,
        ref sigma,
        sigma_for_modular,
        ..
    } = frame_header.restoration_filter.epf
    else {
        return Ok(());
    };

    let span = tracing::span!(tracing::Level::TRACE, "Edge-preserving filter");
    let _guard = span.enter();

    let tracker = fb.alloc_tracker().cloned();
    let region = fb.region();
    let fb = fb.buffer_mut();
    assert!(region.left % 8 == 0);
    assert!(region.top % 8 == 0);

    let width = region.width as usize;
    let height = region.height as usize;
    // Mirror padding, extra padding for SIMD
    let padded_width = (width + 8 + 3 + 7) & !7;
    let padded_height = height + 6;
    let mut fb_in = [
        SimpleGrid::with_alloc_tracker(padded_width, padded_height, tracker.as_ref())?,
        SimpleGrid::with_alloc_tracker(padded_width, padded_height, tracker.as_ref())?,
        SimpleGrid::with_alloc_tracker(padded_width, padded_height, tracker.as_ref())?,
    ];
    let fb_out = <&mut [_; 3]>::try_from(fb).unwrap();

    tracing::debug!("Preparing sigma grid");
    let sigma_region = region.downsample(3);
    let mut sigma_image =
        ImageWithRegion::from_region_and_tracker(1, sigma_region, false, tracker.as_ref())?;
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
                    *sigma_grid.get_mut(sigma_x, sigma_y).unwrap() =
                        *epf_sigma.get(lf_x, lf_y).unwrap();
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
        for (output, input) in fb_in.iter_mut().zip(&*fb_out) {
            let output = output.buf_mut();
            let input = input.buf();
            for y in 0..height {
                output[(y + 3) * padded_width + 8..][..width]
                    .copy_from_slice(&input[y * width..][..width]);
            }
        }
        for output in &mut fb_in {
            let output = output.buf_mut();

            for row in output.chunks_exact_mut(padded_width).skip(3).take(height) {
                row[7] = row[8];
                row[8 + width] = row[8 + width - 1];
                row[6] = row[9];
                row[8 + width + 1] = row[8 + width - 2];
                row[5] = row[10];
                row[8 + width + 2] = row[8 + width - 3];
            }

            let (out_chunk, in_chunk) = output.split_at_mut(padded_width * 3);
            let in_chunk = &in_chunk[..padded_width * 3];
            for (out_row, in_row) in out_chunk
                .chunks_exact_mut(padded_width)
                .zip(in_chunk.chunks_exact(padded_width).rev())
            {
                out_row.copy_from_slice(in_row);
            }

            let (in_chunk, out_chunk) = output.split_at_mut(padded_width * (height + 3));
            for (out_row, in_row) in out_chunk
                .chunks_exact_mut(padded_width)
                .zip(in_chunk.chunks_exact(padded_width).rev())
            {
                out_row.copy_from_slice(in_row);
            }
        }

        super::impls::epf_step0(
            &fb_in,
            fb_out,
            sigma_grid,
            channel_scale,
            sigma.border_sad_mul,
            sigma.pass0_sigma_scale,
            pool,
        );
    }

    // Step 1
    {
        tracing::debug!("Running step 1");
        for (output, input) in fb_in.iter_mut().zip(&*fb_out) {
            let output = output.buf_mut();
            let input = input.buf();
            for y in 0..height {
                output[(y + 3) * padded_width + 8..][..width]
                    .copy_from_slice(&input[y * width..][..width]);
            }
        }
        for output in &mut fb_in {
            let output = output.buf_mut();

            for row in output.chunks_exact_mut(padded_width).skip(3).take(height) {
                row[7] = row[8];
                row[8 + width] = row[8 + width - 1];
                row[6] = row[9];
                row[8 + width + 1] = row[8 + width - 2];
                row[5] = row[10];
                row[8 + width + 2] = row[8 + width - 3];
            }

            let (out_chunk, in_chunk) = output.split_at_mut(padded_width * 3);
            let in_chunk = &in_chunk[..padded_width * 3];
            for (out_row, in_row) in out_chunk
                .chunks_exact_mut(padded_width)
                .zip(in_chunk.chunks_exact(padded_width).rev())
            {
                out_row.copy_from_slice(in_row);
            }

            let (in_chunk, out_chunk) = output.split_at_mut(padded_width * (height + 3));
            for (out_row, in_row) in out_chunk
                .chunks_exact_mut(padded_width)
                .zip(in_chunk.chunks_exact(padded_width).rev())
            {
                out_row.copy_from_slice(in_row);
            }
        }

        super::impls::epf_step1(
            &fb_in,
            fb_out,
            sigma_grid,
            channel_scale,
            sigma.border_sad_mul,
            1.0,
            pool,
        );
    }

    // Step 2
    if iters >= 2 {
        tracing::debug!("Running step 2");
        for (output, input) in fb_in.iter_mut().zip(&*fb_out) {
            let output = output.buf_mut();
            let input = input.buf();
            for y in 0..height {
                output[(y + 3) * padded_width + 8..][..width]
                    .copy_from_slice(&input[y * width..][..width]);
            }
        }
        for output in &mut fb_in {
            let output = output.buf_mut();

            for row in output.chunks_exact_mut(padded_width).skip(3).take(height) {
                row[7] = row[8];
                row[8 + width] = row[8 + width - 1];
                row[6] = row[9];
                row[8 + width + 1] = row[8 + width - 2];
                row[5] = row[10];
                row[8 + width + 2] = row[8 + width - 3];
            }

            let (out_chunk, in_chunk) = output.split_at_mut(padded_width * 3);
            let in_chunk = &in_chunk[..padded_width * 3];
            for (out_row, in_row) in out_chunk
                .chunks_exact_mut(padded_width)
                .zip(in_chunk.chunks_exact(padded_width).rev())
            {
                out_row.copy_from_slice(in_row);
            }

            let (in_chunk, out_chunk) = output.split_at_mut(padded_width * (height + 3));
            for (out_row, in_row) in out_chunk
                .chunks_exact_mut(padded_width)
                .zip(in_chunk.chunks_exact(padded_width).rev())
            {
                out_row.copy_from_slice(in_row);
            }
        }

        super::impls::epf_step2(
            &fb_in,
            fb_out,
            sigma_grid,
            channel_scale,
            sigma.border_sad_mul,
            sigma.pass2_sigma_scale,
            pool,
        );
    }

    Ok(())
}
