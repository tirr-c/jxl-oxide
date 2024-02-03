use std::collections::HashMap;

use jxl_frame::{data::LfGroup, filter::EdgePreservingFilter, FrameHeader};
use jxl_grid::SimpleGrid;
use jxl_modular::Sample;

use crate::{ImageWithRegion, Region};

pub fn apply_epf<S: Sample>(
    fb: &mut ImageWithRegion,
    lf_groups: &HashMap<u32, LfGroup<S>>,
    frame_header: &FrameHeader,
    pool: &jxl_threadpool::JxlThreadPool,
) -> crate::Result<()> {
    let EdgePreservingFilter::Enabled(epf_params) = &frame_header.restoration_filter.epf
    else {
        return Ok(());
    };
    let iters = epf_params.iters;

    let span = tracing::span!(tracing::Level::TRACE, "Edge-preserving filter");
    let _guard = span.enter();

    let tracker = fb.alloc_tracker().cloned();
    let region = fb.region();
    let fb = fb.buffer_mut();

    let width = region.width as usize;
    let height = region.height as usize;
    let fb_in = <&mut [_; 3]>::try_from(fb).unwrap();
    let mut fb_out = [
        SimpleGrid::with_alloc_tracker(width, height, tracker.as_ref())?,
        SimpleGrid::with_alloc_tracker(width, height, tracker.as_ref())?,
        SimpleGrid::with_alloc_tracker(width, height, tracker.as_ref())?,
    ];

    tracing::debug!("Preparing sigma grid");
    let sigma_region = region.downsample_separate(0, 3);
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
            left: base_x as i32 * 8,
            top: base_y as i32,
            width: lf_group_dim8 * 8,
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
                for x in 0..lf_region.width as usize {
                    let lf_x = (lf_region.left as usize + x) / 8;
                    let sigma_x = sigma_region.left as usize + x;
                    *sigma_grid.get_mut(sigma_x, sigma_y).unwrap() =
                        *epf_sigma.get(lf_x, lf_y).unwrap();
                }
            }
        }
    }
    if need_sigma_init {
        for sigma in sigma_grid.buf_mut() {
            *sigma = epf_params.sigma_for_modular;
        }
    }

    // Step 0
    if iters == 3 {
        tracing::debug!("Running step 0");
        super::impls::epf_step0(
            fb_in,
            &mut fb_out,
            sigma_grid,
            region,
            epf_params,
            pool,
        );
        std::mem::swap(&mut fb_in[0], &mut fb_out[0]);
        std::mem::swap(&mut fb_in[1], &mut fb_out[1]);
        std::mem::swap(&mut fb_in[2], &mut fb_out[2]);
    }

    // Step 1
    {
        tracing::debug!("Running step 1");
        super::impls::epf_step1(
            fb_in,
            &mut fb_out,
            sigma_grid,
            region,
            epf_params,
            pool,
        );
        std::mem::swap(&mut fb_in[0], &mut fb_out[0]);
        std::mem::swap(&mut fb_in[1], &mut fb_out[1]);
        std::mem::swap(&mut fb_in[2], &mut fb_out[2]);
    }

    // Step 2
    if iters >= 2 {
        tracing::debug!("Running step 2");
        super::impls::epf_step2(
            fb_in,
            &mut fb_out,
            sigma_grid,
            region,
            epf_params,
            pool,
        );
        std::mem::swap(&mut fb_in[0], &mut fb_out[0]);
        std::mem::swap(&mut fb_in[1], &mut fb_out[1]);
        std::mem::swap(&mut fb_in[2], &mut fb_out[2]);
    }

    Ok(())
}
