use std::io::prelude::*;
use std::time::Duration;

use jxl_oxide::{
    AllocTracker, CropInfo, EnumColourEncoding, FrameBuffer, JxlImage, JxlThreadPool, PixelFormat,
    Render,
};

use crate::commands::decode::*;
use crate::{Error, Result};

pub fn handle_decode(args: DecodeArgs) -> Result<()> {
    let _guard = tracing::trace_span!("Handle decode subcommand").entered();

    #[cfg(feature = "rayon")]
    let pool = JxlThreadPool::rayon(args.num_threads);
    #[cfg(not(feature = "rayon"))]
    let pool = JxlThreadPool::none();

    let mut image_builder = JxlImage::builder()
        .pool(pool.clone())
        .lz77_mode(args.lz77_mode.into());
    if args.approx_memory_limit != 0 {
        let tracker = AllocTracker::with_limit(args.approx_memory_limit);
        image_builder = image_builder.alloc_tracker(tracker);
    }
    let mut image = image_builder.open(&args.input).map_err(Error::ReadJxl)?;
    if !image.is_loading_done() {
        tracing::warn!("Partial image");
    }

    let output_png = args.output.is_some()
        && matches!(
            args.output_format,
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

    if args.output_format == OutputFormat::Npy {
        image.set_render_spot_colour(false);
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

        tracing::debug!(output_format = format_args!("{:?}", args.output_format));
        let pixel_format = image.pixel_format();
        let output = std::fs::File::create(output).map_err(Error::WriteImage)?;
        match args.output_format {
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

                write_png(
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
                write_png(
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
                write_png(
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

                write_npy(output, &keyframes, width, height).map_err(Error::WriteImage)?;
            }
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
            let frame = image
                .render_frame_cropped(idx)
                .expect("rendering frames failed");
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

fn write_png<W: Write>(
    output: W,
    image: &JxlImage,
    keyframes: &[Render],
    pixfmt: PixelFormat,
    force_bit_depth: Option<png::BitDepth>,
    width: u32,
    height: u32,
) -> std::io::Result<()> {
    // Color encoding information
    let source_icc = image.rendered_icc();
    let cicp = image.rendered_cicp();
    let metadata = &image.image_header().metadata;

    let mut encoder = png::Encoder::new(output, width, height);

    let color_type = match pixfmt {
        PixelFormat::Gray => png::ColorType::Grayscale,
        PixelFormat::Graya => png::ColorType::GrayscaleAlpha,
        PixelFormat::Rgb => png::ColorType::Rgb,
        PixelFormat::Rgba => png::ColorType::Rgba,
        _ => {
            tracing::error!("Cannot output CMYK PNG");
            panic!();
        }
    };
    encoder.set_color(color_type);

    let bit_depth = force_bit_depth.unwrap_or(if metadata.bit_depth.bits_per_sample() > 8 {
        png::BitDepth::Sixteen
    } else {
        png::BitDepth::Eight
    });
    let sixteen_bits = bit_depth == png::BitDepth::Sixteen;
    if sixteen_bits {
        encoder.set_depth(png::BitDepth::Sixteen);
    } else {
        encoder.set_depth(png::BitDepth::Eight);
    }

    if let Some(animation) = &metadata.animation {
        let num_plays = animation.num_loops;
        encoder
            .set_animated(keyframes.len() as u32, num_plays)
            .unwrap();
    }

    encoder.validate_sequence(true);

    let mut writer = encoder.write_header()?;

    tracing::debug!("Embedding ICC profile");
    let compressed_icc = miniz_oxide::deflate::compress_to_vec_zlib(&source_icc, 7);
    let mut iccp_chunk_data = vec![b'0', 0, 0];
    iccp_chunk_data.extend(compressed_icc);
    writer.write_chunk(png::chunk::iCCP, &iccp_chunk_data)?;

    if let Some(cicp) = cicp {
        tracing::debug!(cicp = format_args!("{:?}", cicp), "Writing cICP chunk");
        writer.write_chunk(png::chunk::ChunkType([b'c', b'I', b'C', b'P']), &cicp)?;
    }

    tracing::debug!("Writing image data");
    for keyframe in keyframes {
        if let Some(animation) = &metadata.animation {
            let duration = keyframe.duration();
            let numer = animation.tps_denominator * duration;
            let denom = animation.tps_numerator;
            let (numer, denom) = if numer >= 0x10000 || denom >= 0x10000 {
                if duration == 0xffffffff {
                    tracing::warn!(numer, denom, "Writing multi-page image in APNG");
                } else {
                    tracing::warn!(numer, denom, "Frame duration is not representable in APNG");
                }
                let duration = (numer as f32 / denom as f32) * 65535.0;
                (duration as u16, 0xffffu16)
            } else {
                (numer as u16, denom as u16)
            };
            writer.set_frame_delay(numer, denom).unwrap();
        }

        let mut stream = keyframe.stream();
        let mut fb = FrameBuffer::new(
            stream.width() as usize,
            stream.height() as usize,
            stream.channels() as usize,
        );
        stream.write_to_buffer(fb.buf_mut());

        if sixteen_bits {
            let mut buf = vec![0u8; fb.width() * fb.height() * fb.channels() * 2];
            for (b, s) in buf.chunks_exact_mut(2).zip(fb.buf()) {
                let w = (*s * 65535.0 + 0.5).clamp(0.0, 65535.0) as u16;
                let [b0, b1] = w.to_be_bytes();
                b[0] = b0;
                b[1] = b1;
            }
            writer.write_image_data(&buf)?;
        } else {
            let mut buf = vec![0u8; fb.width() * fb.height() * fb.channels()];
            for (b, s) in buf.iter_mut().zip(fb.buf()) {
                *b = (*s * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            }
            writer.write_image_data(&buf)?;
        }
    }

    writer.finish()?;
    Ok(())
}

fn write_npy<W: Write>(
    output: W,
    keyframes: &[Render],
    width: u32,
    height: u32,
) -> std::io::Result<()> {
    let channels = {
        let first_frame = keyframes.first().unwrap();
        first_frame.color_channels().len() + first_frame.extra_channels().0.len()
    };

    let mut output = std::io::BufWriter::new(output);
    output.write_all(b"\x93NUMPY\x01\x00").unwrap();
    let header = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, {}, {}, {}), }}\n",
        keyframes.len(),
        height,
        width,
        channels,
    );
    output.write_all(&(header.len() as u16).to_le_bytes())?;
    output.write_all(header.as_bytes())?;

    tracing::debug!("Writing image data");
    for keyframe in keyframes {
        let fb = keyframe.image_all_channels();
        for sample in fb.buf() {
            output.write_all(&sample.to_bits().to_le_bytes())?;
        }
    }

    output.flush()?;
    Ok(())
}
