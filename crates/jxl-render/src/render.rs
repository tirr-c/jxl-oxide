use jxl_frame::{
    filter::{EdgePreservingFilter, Gabor},
    header::Encoding,
};
use jxl_grid::AlignedGrid;
use jxl_modular::Sample;
use jxl_threadpool::JxlThreadPool;

use crate::{
    blend, features, filter, modular, state::RenderCache, util, vardct, Error, ImageWithRegion,
    IndexedFrame, Reference, ReferenceFrames, Region, Result,
};

pub(crate) fn render_frame<S: Sample>(
    frame: &IndexedFrame,
    reference_frames: ReferenceFrames<S>,
    cache: &mut RenderCache<S>,
    image_region: Region,
    pool: JxlThreadPool,
    frame_visibility: (usize, usize),
) -> Result<ImageWithRegion> {
    let frame_region = util::image_region_to_frame(frame, image_region, false);
    tracing::debug!(
        index = frame.idx,
        ?image_region,
        ?frame_region,
        "Rendering frame"
    );

    let image_header = frame.image_header();
    let frame_header = frame.header();
    let frame_region = util::pad_lf_region(frame_header, frame_region);

    let upsampled_full_frame_region =
        Region::with_size(frame_header.sample_width(1), frame_header.sample_height(1));
    let upsampling_valid_region = util::pad_upsampling(image_header, frame_header, frame_region)
        .intersection(upsampled_full_frame_region);

    let full_frame_region = Region::with_size(
        frame_header.color_sample_width(),
        frame_header.color_sample_height(),
    );
    let color_padded_region = util::pad_color_region(image_header, frame_header, frame_region)
        .intersection(full_frame_region);

    let mut fb = match frame_header.encoding {
        Encoding::Modular => modular::render_modular(frame, cache, color_padded_region, &pool)?,
        Encoding::VarDct => {
            let result = vardct::render_vardct(
                frame,
                reference_frames.lf.as_ref(),
                cache,
                color_padded_region,
                &pool,
            );
            match (result, reference_frames.lf) {
                (Ok(grid), _) => grid,
                (Err(e), Some(lf)) if e.unexpected_eof() => {
                    let render = lf.image.run_with_image()?;
                    let render = render.blend(None, &pool)?;
                    render.upsample_lf(lf.frame.header().lf_level)?
                }
                (Err(e), _) => return Err(e),
            }
        }
    };

    if frame.header().do_ycbcr {
        fb.upsample_jpeg(color_padded_region, image_header.metadata.bit_depth)?;
    }

    let mut scratch_buffer = None;
    if let Gabor::Enabled(weights) = frame_header.restoration_filter.gab {
        fb.convert_modular_color(image_header.metadata.bit_depth)?;
        if scratch_buffer.is_none() {
            let tracker = fb.alloc_tracker();
            let width = color_padded_region.width as usize;
            let height = color_padded_region.height as usize;
            scratch_buffer = Some([
                AlignedGrid::with_alloc_tracker(width, height, tracker)?,
                AlignedGrid::with_alloc_tracker(width, height, tracker)?,
                AlignedGrid::with_alloc_tracker(width, height, tracker)?,
            ]);
        }
        let fb_scratch = scratch_buffer.as_mut().unwrap();
        filter::apply_gabor_like(&mut fb, color_padded_region, fb_scratch, weights, &pool);
    }
    if let EdgePreservingFilter::Enabled(epf_params) = &frame_header.restoration_filter.epf {
        fb.convert_modular_color(image_header.metadata.bit_depth)?;
        let fb_scratch = if let Some(buffer) = scratch_buffer {
            buffer
        } else {
            let tracker = fb.alloc_tracker();
            let width = color_padded_region.width as usize;
            let height = color_padded_region.height as usize;
            [
                AlignedGrid::with_alloc_tracker(width, height, tracker)?,
                AlignedGrid::with_alloc_tracker(width, height, tracker)?,
                AlignedGrid::with_alloc_tracker(width, height, tracker)?,
            ]
        };
        filter::apply_epf(
            &mut fb,
            fb_scratch,
            color_padded_region,
            &cache.lf_groups,
            frame_header,
            epf_params,
            &pool,
        );
    }

    tracing::trace!(?upsampling_valid_region);
    fb.upsample_nonseparable(image_header, frame_header, upsampling_valid_region)?;

    render_features(
        frame,
        &mut fb,
        reference_frames.refs.clone(),
        cache,
        frame_visibility.0,
        frame_visibility.1,
        &pool,
    )?;

    if !frame_header.save_before_ct && !frame_header.is_last {
        fb.convert_modular_color(image_header.metadata.bit_depth)?;
        let ct_done = util::convert_color_for_record(
            image_header,
            frame_header.do_ycbcr,
            fb.as_color_floats_mut(),
            &pool,
        );
        fb.set_ct_done(ct_done);
    }

    Ok(fb)
}

fn render_features<S: Sample>(
    frame: &IndexedFrame,
    grid: &mut ImageWithRegion,
    reference_grids: [Option<Reference<S>>; 4],
    cache: &mut RenderCache<S>,
    visible_frames_num: usize,
    invisible_frames_num: usize,
    pool: &JxlThreadPool,
) -> Result<()> {
    let image_header = frame.image_header();
    let frame_header = frame.header();
    let Some(lf_global) = cache.lf_global.as_ref() else {
        tracing::trace!("LfGlobal not available, skipping feature rendering");
        return Ok(());
    };
    let base_correlations_xb = lf_global.vardct.as_ref().map(|x| {
        (
            x.lf_chan_corr.base_correlation_x,
            x.lf_chan_corr.base_correlation_b,
        )
    });

    if let Some(patches) = &lf_global.patches {
        for patch in &patches.patches {
            let Some(ref_grid) = &reference_grids[patch.ref_idx as usize] else {
                return Err(Error::InvalidReference(patch.ref_idx));
            };
            let ref_header = ref_grid.frame.f.header();
            let oriented_image_region = Region::with_size(ref_header.width, ref_header.height)
                .translate(ref_header.x0, ref_header.y0);
            let ref_grid_image = std::sync::Arc::clone(&ref_grid.image).run_with_image()?;
            let mut ref_grid_image = ref_grid_image.blend(Some(oriented_image_region), pool)?;
            blend::patch(image_header, grid, &mut ref_grid_image, patch)?;
        }
    }

    if let Some(splines) = &lf_global.splines {
        features::render_spline(frame_header, grid, splines, base_correlations_xb)?;
    }
    if let Some(noise) = &lf_global.noise {
        features::render_noise(
            frame.header(),
            visible_frames_num,
            invisible_frames_num,
            base_correlations_xb,
            grid,
            noise,
        )?;
    }

    Ok(())
}
