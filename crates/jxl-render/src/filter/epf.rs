use std::collections::HashMap;

use jxl_frame::{data::LfGroup, filter::EdgePreservingFilter, FrameHeader};
use jxl_grid::SimpleGrid;
use jxl_modular::Sample;

use crate::ImageWithRegion;

pub fn apply_epf<S: Sample>(
    fb: &mut ImageWithRegion,
    lf_groups: &HashMap<u32, LfGroup<S>>,
    frame_header: &FrameHeader,
    pool: &jxl_threadpool::JxlThreadPool,
) -> crate::Result<()> {
    let EdgePreservingFilter::Enabled(epf_params) = &frame_header.restoration_filter.epf else {
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

    let num_lf_groups = frame_header.num_lf_groups() as usize;
    let mut sigma_grid_map = vec![None::<&SimpleGrid<f32>>; num_lf_groups];

    for (&lf_group_idx, lf_group) in lf_groups {
        if let Some(hf_meta) = &lf_group.hf_meta {
            sigma_grid_map[lf_group_idx as usize] = Some(&hf_meta.epf_sigma);
        }
    }

    // Step 0
    if iters == 3 {
        tracing::debug!("Running step 0");
        super::impls::epf::<0>(
            fb_in,
            &mut fb_out,
            frame_header,
            &sigma_grid_map,
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
        super::impls::epf::<1>(
            fb_in,
            &mut fb_out,
            frame_header,
            &sigma_grid_map,
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
        super::impls::epf::<2>(
            fb_in,
            &mut fb_out,
            frame_header,
            &sigma_grid_map,
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
