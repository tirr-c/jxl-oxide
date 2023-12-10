use jxl_color::{ColourEncoding, EnumColourEncoding};
use jxl_frame::{
    data::*,
    filter::{EdgePreservingFilter, Gabor},
    header::Encoding,
    FrameHeader,
};
use jxl_grid::SimpleGrid;
use jxl_image::ImageHeader;
use jxl_threadpool::JxlThreadPool;

use crate::{
    blend, features, filter, modular,
    region::{ImageWithRegion, Region},
    state::RenderCache,
    vardct, Error, IndexedFrame, Reference, ReferenceFrames, Result,
};

pub(crate) fn render_frame(
    frame: &IndexedFrame,
    reference_frames: ReferenceFrames,
    cache: &mut RenderCache,
    image_region: Option<Region>,
    pool: JxlThreadPool,
    frame_visibility: (usize, usize),
) -> Result<ImageWithRegion> {
    let frame_region = crate::image_region_to_frame(frame, image_region, false);

    let image_header = frame.image_header();
    let frame_header = frame.header();
    let full_frame_region = Region::with_size(
        frame_header.color_sample_width(),
        frame_header.color_sample_height(),
    );
    let frame_region = if frame_header.lf_level != 0 {
        // Lower level frames might be padded, so apply padding to LF frames
        frame_region.pad(4 * frame_header.lf_level + 32)
    } else {
        frame_region
    };

    let color_upsample_factor = frame_header.upsampling.ilog2();
    let max_upsample_factor = frame_header
        .ec_upsampling
        .iter()
        .zip(image_header.metadata.ec_info.iter())
        .map(|(upsampling, ec_info)| upsampling.ilog2() + ec_info.dim_shift)
        .max()
        .unwrap_or(color_upsample_factor);

    let mut color_padded_region = if max_upsample_factor > 0 {
        // Additional upsampling pass is needed for every 3 levels of upsampling factor.
        let padded_region = frame_region
            .downsample(max_upsample_factor)
            .pad(2 + (max_upsample_factor - 1) / 3);
        let upsample_diff = max_upsample_factor - color_upsample_factor;
        padded_region.upsample(upsample_diff)
    } else {
        frame_region
    };

    // TODO: actual region could be smaller.
    if let EdgePreservingFilter::Enabled { iters, .. } = frame_header.restoration_filter.epf {
        // EPF references adjacent samples.
        color_padded_region = if iters == 1 {
            color_padded_region.pad(2)
        } else if iters == 2 {
            color_padded_region.pad(5)
        } else {
            color_padded_region.pad(6)
        };
    }
    if frame_header.restoration_filter.gab.enabled() {
        // Gabor-like filter references adjacent samples.
        color_padded_region = color_padded_region.pad(1);
    }
    if frame_header.do_ycbcr {
        // Chroma upsampling references adjacent samples.
        color_padded_region = color_padded_region.pad(1).downsample(2).upsample(2);
    }
    if frame_header.restoration_filter.epf.enabled() {
        // EPF performs filtering in 8x8 blocks.
        color_padded_region = color_padded_region.container_aligned(8);
    }
    color_padded_region = color_padded_region.intersection(full_frame_region);

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
                image_region,
                &pool,
            );
            match (result, reference_frames.lf) {
                (Ok((grid, gmodular)), _) => (grid, Some(gmodular)),
                (Err(e), Some(lf)) if e.unexpected_eof() => {
                    let render = lf.image.run_with_image(image_region)?;
                    (super::upsample_lf(&render, &lf.frame, frame_region)?, None)
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
        image_region,
        &mut fb,
        reference_frames.refs.clone(),
        cache,
        frame_visibility.0,
        frame_visibility.1,
    )?;

    // save_before_ct is always false if is_last = true
    if !frame_header.save_before_ct && !frame_header.is_last {
        convert_color_for_record(image_header, frame_header.do_ycbcr, fb.buffer_mut());
    }

    Ok(
        if !frame_header.frame_type.is_normal_frame() || frame_header.resets_canvas {
            fb
        } else {
            blend::blend(
                frame.image_header(),
                image_region,
                reference_frames.refs,
                frame,
                &fb,
            )?
        },
    )
}

fn upsample_color_channels(
    fb: &mut ImageWithRegion,
    image_header: &ImageHeader,
    frame_header: &FrameHeader,
    original_region: Region,
) -> Result<()> {
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

    let upsampled_fb = ImageWithRegion::from_buffer(
        upsampled_buffer,
        upsampled_region.left,
        upsampled_region.top,
    );
    tracing::trace_span!("Copy upsampled color channels").in_scope(|| -> Result<_> {
        *fb = ImageWithRegion::from_region_and_tracker(
            upsampled_fb.channels(),
            original_region,
            fb.alloc_tracker(),
        )?;
        for (channel_idx, output) in fb.buffer_mut().iter_mut().enumerate() {
            upsampled_fb.clone_region_channel(original_region, channel_idx, output);
        }
        Ok(())
    })
}

fn append_extra_channels(
    frame: &IndexedFrame,
    fb: &mut ImageWithRegion,
    gmodular: GlobalModular,
    original_region: Region,
) -> Result<()> {
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

        let upsampled_region = region.upsample(upsample_factor);
        features::upsample(&mut out, image_header, frame_header, idx + 3)?;
        let out =
            ImageWithRegion::from_buffer(vec![out], upsampled_region.left, upsampled_region.top);
        let cropped = fb.add_channel()?;
        out.clone_region_channel(fb_region, 0, cropped);
    }

    Ok(())
}

fn render_features(
    frame: &IndexedFrame,
    image_region: Option<Region>,
    grid: &mut ImageWithRegion,
    reference_grids: [Option<Reference>; 4],
    cache: &mut RenderCache,
    visible_frames_num: usize,
    invisible_frames_num: usize,
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
            let ref_grid_image = ref_grid.image.run_with_image(image_region)?;
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

fn convert_color_for_record(
    image_header: &ImageHeader,
    do_ycbcr: bool,
    grid: &mut [SimpleGrid<f32>],
) {
    // save_before_ct = false

    let metadata = &image_header.metadata;
    if do_ycbcr {
        // xyb_encoded = false
        let [cb, y, cr, ..] = grid else { panic!() };
        jxl_color::ycbcr_to_rgb([cb, y, cr]);
    } else if metadata.xyb_encoded {
        // want_icc = false
        let [x, y, b, ..] = grid else { panic!() };
        tracing::trace_span!("XYB to target colorspace").in_scope(|| {
            tracing::trace!(colour_encoding = ?metadata.colour_encoding);
            let transform = jxl_color::ColorTransform::new(
                &jxl_color::ColorEncodingWithProfile::new(ColourEncoding::Enum(EnumColourEncoding::xyb())),
                &jxl_color::ColorEncodingWithProfile::new(metadata.colour_encoding.clone()),
                &metadata.opsin_inverse_matrix,
                metadata.tone_mapping.intensity_target,
            );
            transform.run(
                &mut [x.buf_mut(), y.buf_mut(), b.buf_mut()],
                &jxl_color::NullCms,
            ).unwrap();
        });
    }
}
