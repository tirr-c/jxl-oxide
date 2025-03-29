use std::time::Duration;

use jxl_oxide::{AllocTracker, CropInfo, EnumColourEncoding, JxlImage, Render};

use crate::commands::decode::*;
use crate::{output, Error, Result};

pub fn handle_decode(args: DecodeArgs) -> Result<()> {
    let _guard = tracing::trace_span!("Handle decode subcommand").entered();

    let pool = crate::create_thread_pool(args.num_threads);

    let mut image_builder = JxlImage::builder().pool(pool.clone());
    if args.approx_memory_limit != 0 {
        let tracker = AllocTracker::with_limit(args.approx_memory_limit);
        image_builder = image_builder.alloc_tracker(tracker);
    }
    let mut image = image_builder.open(&args.input).map_err(Error::ReadJxl)?;
    if !image.is_loading_done() {
        tracing::warn!("Partial image");
    }

    let explicit_jpeg = args.output_format == Some(OutputFormat::JpegReconstruct);
    let mut output_format = args.output_format.unwrap_or_else(|| {
        if let Some(output_path) = &args.output {
            if output_path
                .extension()
                .map(|x| x == "jpg" || x == "jpeg")
                .unwrap_or(false)
            {
                OutputFormat::JpegReconstruct
            } else {
                OutputFormat::Png
            }
        } else {
            OutputFormat::Png
        }
    });

    if output_format == OutputFormat::JpegReconstruct {
        tracing::info!("Reconstructing to JPEG");

        let mut buf = Vec::new();
        let decode_start = std::time::Instant::now();
        let result = image.reconstruct_jpeg(&mut buf);
        let elapsed = decode_start.elapsed();

        if let Err(e) = result {
            tracing::error!(%e, "Failed to reconstruct");
            if explicit_jpeg {
                return Err(Error::Reconstruct(e));
            }

            tracing::warn!("Falling back to decode-to-pixels");
            output_format = OutputFormat::Png
        } else {
            let total_pixels = image.width() * image.height();
            let mps = total_pixels as f64 / 1e6;
            let elapsed_seconds = elapsed.as_secs_f64();
            tracing::info!(
                "Took {:.2} ms ({:.2} MP/s)",
                elapsed_seconds * 1000.0,
                mps / elapsed_seconds
            );

            if let Some(output_path) = args.output {
                std::fs::write(output_path, buf).map_err(Error::WriteImage)?;
            } else {
                tracing::info!("No output path specified, skipping output encoding");
            }

            return Ok(());
        }
    }

    let output_png = args.output.is_some()
        && matches!(
            output_format,
            OutputFormat::Png | OutputFormat::Png8 | OutputFormat::Png16
        );
    if let Some(icc_path) = &args.target_icc {
        tracing::debug!("Setting target ICC profile");
        let icc_profile = std::fs::read(icc_path).map_err(Error::ReadIcc)?;
        match image.request_icc(&icc_profile) {
            Ok(_) => {}
            Err(e) => {
                tracing::error!(%e, "Target ICC profile is malformed");
            }
        }
    } else if let Some(encoding) = &args.target_colorspace {
        tracing::debug!(?encoding, "Setting target color space");
        image.request_color_encoding(encoding.clone());
    } else if output_png && image.pixel_format().has_black() {
        tracing::debug!("Input is CMYK; setting target color encoding to sRGB");
        image.request_color_encoding(EnumColourEncoding::srgb(
            jxl_oxide::color::RenderingIntent::Relative,
        ));
    }

    let image_meta = &image.image_header().metadata;
    tracing::info!("Image dimension: {}x{}", image.width(), image.height());
    tracing::debug!(colour_encoding = format_args!("{:?}", image_meta.colour_encoding));

    if let Some(icc_path) = &args.icc_output {
        if let Some(icc) = image.original_icc() {
            tracing::debug!("Writing ICC profile");
            std::fs::write(icc_path, icc).map_err(Error::WriteIcc)?;
        } else {
            tracing::info!("No embedded ICC profile, skipping icc_output");
        }
    }

    if let Some(cms) = args.cms {
        match cms {
            Cms::Lcms2 => {
                tracing::debug!("Using Little CMS 2");
                image.set_cms(jxl_oxide::Lcms2);
            }
            Cms::Moxcms => {
                tracing::debug!("Using moxcms");
                image.set_cms(jxl_oxide::Moxcms);
            }
        }
    }

    let crop = args.crop.and_then(|crop| {
        if crop.width == 0 && crop.height == 0 {
            None
        } else if crop.width == 0 {
            Some(CropInfo {
                width: image.width(),
                ..crop
            })
        } else if crop.height == 0 {
            Some(CropInfo {
                height: image.height(),
                ..crop
            })
        } else {
            Some(crop)
        }
    });

    if let Some(crop) = crop {
        tracing::debug!(crop = format_args!("{:?}", crop), "Cropped decoding");
    }

    let crop_region = crop.unwrap_or(CropInfo {
        width: image.width(),
        height: image.height(),
        left: 0,
        top: 0,
    });
    let CropInfo { width, height, .. } = crop_region;
    let total_pixels = width * height;
    let mps = total_pixels as f64 / 1e6;

    if output_format == OutputFormat::Npy {
        image.set_render_spot_color(false);
    }

    let keyframes = if let Some(num_reps @ 2..) = args.num_reps {
        tracing::info!("Running {num_reps} repetitions");

        let mut durations = Vec::with_capacity(num_reps as usize);
        for _ in 0..num_reps - 1 {
            // Resets internal cache
            image.set_image_region(crop_region);
            let (_, elapsed) = run_once(&mut image)?;
            durations.push(elapsed);
        }
        image.set_image_region(crop_region);
        let (keyframes, elapsed) = run_once(&mut image)?;
        durations.push(elapsed);

        let min = durations.iter().min().unwrap().as_secs_f64();
        let max = durations.iter().max().unwrap().as_secs_f64();
        let geomean = durations
            .iter()
            .fold(1f64, |acc, elapsed| acc * elapsed.as_secs_f64());
        let geomean = geomean.powf(1.0 / num_reps as f64);

        tracing::info!(
            "Geomean: {:.3} ms ({:.3} MP/s)",
            geomean * 1000.0,
            mps / geomean
        );
        tracing::info!(
            "Range: [{:.3} ms, {:.3} ms] ([{:.3} MP/s, {:.3} MP/s])",
            min * 1000.0,
            max * 1000.0,
            mps / max,
            mps / min,
        );

        keyframes
    } else {
        image.set_image_region(crop_region);
        let (keyframes, elapsed) = run_once(&mut image)?;
        let elapsed_seconds = elapsed.as_secs_f64();
        tracing::info!(
            "Took {:.2} ms ({:.2} MP/s)",
            elapsed_seconds * 1000.0,
            mps / elapsed_seconds
        );
        keyframes
    };

    if let Some(output) = &args.output {
        if keyframes.is_empty() {
            tracing::warn!("No keyframes are decoded");
            return Ok(());
        }

        tracing::debug!(?output_format);
        let pixel_format = image.pixel_format();
        let output = std::fs::File::create(output).map_err(Error::WriteImage)?;
        match output_format {
            OutputFormat::Png => {
                let force_bit_depth = if let Some(encoding) = &args.target_colorspace {
                    if encoding.is_srgb_gamut() {
                        Some(png::BitDepth::Eight)
                    } else {
                        None
                    }
                } else {
                    None
                };

                output::write_png(
                    output,
                    &image,
                    &keyframes,
                    pixel_format,
                    force_bit_depth,
                    width,
                    height,
                )
                .map_err(Error::WriteImage)?;
            }
            OutputFormat::Png8 => {
                output::write_png(
                    output,
                    &image,
                    &keyframes,
                    pixel_format,
                    Some(png::BitDepth::Eight),
                    width,
                    height,
                )
                .map_err(Error::WriteImage)?;
            }
            OutputFormat::Png16 => {
                output::write_png(
                    output,
                    &image,
                    &keyframes,
                    pixel_format,
                    Some(png::BitDepth::Sixteen),
                    width,
                    height,
                )
                .map_err(Error::WriteImage)?;
            }
            OutputFormat::Npy => {
                if args.icc_output.is_none() {
                    tracing::warn!("--icc-output is not set. Numpy buffer alone cannot be used to display image as its colorspace is unknown.");
                }

                output::write_npy(output, &keyframes, width, height).map_err(Error::WriteImage)?;
            }
            OutputFormat::JpegReconstruct => unreachable!("should have been processed before"),
        }
    } else {
        tracing::info!("No output path specified, skipping output encoding");
    };

    Ok(())
}

fn run_once(image: &mut JxlImage) -> Result<(Vec<Render>, Duration)> {
    let mut keyframes = Vec::new();
    #[allow(unused_mut)]
    let mut rendered = false;

    let decode_start = std::time::Instant::now();
    #[cfg(feature = "rayon")]
    if let Some(rayon_pool) = image.pool().as_rayon_pool() {
        keyframes = rayon_pool
            .install(|| {
                use rayon::prelude::*;

                (0..image.num_loaded_keyframes())
                    .into_par_iter()
                    .map(|idx| image.render_frame_cropped(idx))
                    .collect::<std::result::Result<Vec<_>, _>>()
            })
            .map_err(Error::Render)?;
        rendered = true;
    }

    if !rendered {
        for idx in 0..image.num_loaded_keyframes() {
            let frame = image.render_frame_cropped(idx).map_err(Error::Render)?;
            keyframes.push(frame);
        }
    }

    if let Ok(frame) = image.render_loading_frame_cropped() {
        tracing::warn!("Rendered partially loaded frame");
        keyframes.push(frame);
    }

    let elapsed = decode_start.elapsed();
    Ok((keyframes, elapsed))
}
