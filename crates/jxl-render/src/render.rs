use std::collections::HashMap;

use jxl_frame::{data::*, filter::Gabor, header::Encoding, FrameHeader};
use jxl_grid::SimpleGrid;
use jxl_image::ImageHeader;
use jxl_modular::Sample;
use jxl_threadpool::JxlThreadPool;

use crate::{
    blend, features, filter, modular,
    region::{ImageWithRegion, Region},
    state::RenderCache,
    util, vardct, Error, IndexedFrame, Reference, ReferenceFrames, Result,
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
    let full_frame_region = Region::with_size(
        frame_header.color_sample_width(),
        frame_header.color_sample_height(),
    );
    let frame_region = util::pad_lf_region(frame_header, frame_region);
    let color_padded_region = util::pad_color_region(image_header, frame_header, frame_region);
    let color_padded_region = color_padded_region.intersection(full_frame_region);

    let (mut fb, gmodular) = match frame_header.encoding {
        Encoding::Modular => {
            let (grid, gmodular) =
                modular::render_modular(frame, cache, color_padded_region, &pool)?;
            (grid, Some(gmodular))
        }
        Encoding::VarDct => {
            let result = vardct::render_vardct(
                frame,
                reference_frames.lf.as_ref(),
                cache,
                color_padded_region,
                &pool,
            );
            match (result, reference_frames.lf) {
                (Ok((grid, gmodular)), _) => (grid, Some(gmodular)),
                (Err(e), Some(lf)) if e.unexpected_eof() => {
                    let mut cache = HashMap::new();
                    let render = lf.image.run_with_image()?;
                    let render = render.blend(&mut cache, None, &pool)?;
                    (util::upsample_lf(&render, &lf.frame, frame_region)?, None)
                }
                (Err(e), _) => return Err(e),
            }
        }
    };
    if fb.region().intersection(full_frame_region) != fb.region() {
        let mut new_fb = fb.clone_intersection(full_frame_region)?;
        std::mem::swap(&mut fb, &mut new_fb);
    }

    let [a, b, c] = fb.buffer_mut() else { panic!() };
    if frame.header().do_ycbcr {
        filter::apply_jpeg_upsampling([a, b, c], frame_header.jpeg_upsampling);
    }
    if let Gabor::Enabled(weights) = frame_header.restoration_filter.gab {
        filter::apply_gabor_like([a, b, c], weights)?;
    }
    filter::apply_epf(&mut fb, &cache.lf_groups, frame_header, &pool)?;

    upsample_color_channels(&mut fb, image_header, frame_header, frame_region)?;
    if let Some(gmodular) = gmodular {
        append_extra_channels(frame, &mut fb, gmodular, frame_region)?;
    }

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
        let ct_done = util::convert_color_for_record(
            image_header,
            frame_header.do_ycbcr,
            fb.buffer_mut(),
            &pool,
        );
        fb.set_ct_done(ct_done);
    }

    Ok(fb)
}

fn upsample_color_channels(
    fb: &mut ImageWithRegion,
    image_header: &ImageHeader,
    frame_header: &FrameHeader,
    original_region: Region,
) -> Result<()> {
    let ct_done = fb.ct_done();
    let upsample_factor = frame_header.upsampling.ilog2();
    let upsampled_region = fb.region().upsample(upsample_factor);
    let upsampled_buffer = if upsample_factor == 0 {
        if fb.region() == original_region {
            return Ok(());
        }
        fb.take_buffer()
    } else {
        let mut buffer = fb.take_buffer();
        tracing::trace_span!("Upsample color channels").in_scope(|| -> Result<_> {
            for (idx, g) in buffer.iter_mut().enumerate() {
                features::upsample(g, image_header, frame_header, idx)?;
            }
            Ok(())
        })?;
        buffer
    };

    *fb = ImageWithRegion::from_buffer(
        upsampled_buffer,
        upsampled_region.left,
        upsampled_region.top,
        ct_done,
    );
    Ok(())
}

fn append_extra_channels<S: Sample>(
    frame: &IndexedFrame,
    fb: &mut ImageWithRegion,
    gmodular: GlobalModular<S>,
    original_region: Region,
) -> Result<()> {
    let _guard = tracing::trace_span!("Append extra channels").entered();

    let fb_region = fb.region();
    let image_header = frame.image_header();
    let frame_header = frame.header();
    let tracker = frame.alloc_tracker();

    let extra_channel_from = gmodular.extra_channel_from();
    let Some(gmodular) = gmodular.modular.into_image() else {
        return Ok(());
    };
    let mut channel_data = gmodular.into_image_channels();
    let channel_data = channel_data.drain(extra_channel_from..);

    for (idx, g) in channel_data.enumerate() {
        tracing::debug!(ec_idx = idx, "Attaching extra channels");

        let upsampling = frame_header.ec_upsampling[idx];
        let ec_info = &image_header.metadata.ec_info[idx];

        let upsample_factor = upsampling.ilog2() + ec_info.dim_shift;
        let region = if upsample_factor > 0 {
            original_region
                .downsample(upsample_factor)
                .pad(2 + (upsample_factor - 1) / 3)
        } else {
            original_region
        };
        let bit_depth = ec_info.bit_depth;

        let width = region.width as usize;
        let height = region.height as usize;
        let mut out = SimpleGrid::with_alloc_tracker(width, height, tracker)?;
        modular::copy_modular_groups(&g, &mut out, region, bit_depth, false);
        features::upsample(&mut out, image_header, frame_header, idx + 3)?;

        let upsampled_region = region.upsample(upsample_factor);
        if upsampled_region == fb_region {
            fb.push_channel(out);
        } else {
            let out = ImageWithRegion::from_buffer(
                vec![out],
                upsampled_region.left,
                upsampled_region.top,
                false,
            );
            let fb_out = fb.add_channel()?;
            out.clone_region_channel(fb_region, 0, fb_out);
        }
    }

    Ok(())
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
        let mut cache = HashMap::new();
        for patch in &patches.patches {
            let Some(ref_grid) = &reference_grids[patch.ref_idx as usize] else {
                return Err(Error::InvalidReference(patch.ref_idx));
            };
            let ref_header = ref_grid.frame.f.header();
            let oriented_image_region = Region::with_size(ref_header.width, ref_header.height)
                .translate(ref_header.x0, ref_header.y0);
            let ref_grid_image = std::sync::Arc::clone(&ref_grid.image).run_with_image()?;
            let ref_grid_image =
                ref_grid_image.blend(&mut cache, Some(oriented_image_region), pool)?;
            blend::patch(image_header, grid, &ref_grid_image, patch);
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
