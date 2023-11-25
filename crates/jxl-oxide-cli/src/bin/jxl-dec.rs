use std::{io::prelude::*, path::PathBuf};

use clap::Parser;
use jxl_oxide::{
    AllocTracker, CropInfo, FrameBuffer, JxlImage, JxlThreadPool, PixelFormat, Render,
};
use lcms2::Profile;

enum LcmsTransform {
    Grayscale(lcms2::Transform<f32, f32, lcms2::GlobalContext, lcms2::AllowCache>),
    GrayscaleAlpha(lcms2::Transform<[f32; 2], [f32; 2], lcms2::GlobalContext, lcms2::AllowCache>),
    Rgb(lcms2::Transform<[f32; 3], [f32; 3], lcms2::GlobalContext, lcms2::AllowCache>),
    Rgba(lcms2::Transform<[f32; 4], [f32; 4], lcms2::GlobalContext, lcms2::AllowCache>),
}

impl LcmsTransform {
    fn transform_in_place(&self, fb: &mut FrameBuffer) {
        use LcmsTransform::*;

        match self {
            Grayscale(t) => t.transform_in_place(fb.buf_mut()),
            GrayscaleAlpha(t) => t.transform_in_place(fb.buf_grouped_mut()),
            Rgb(t) => t.transform_in_place(fb.buf_grouped_mut()),
            Rgba(t) => t.transform_in_place(fb.buf_grouped_mut()),
        }
    }
}

/// Decodes JPEG XL image.
#[derive(Debug, Parser)]
#[command(version)]
struct Args {
    /// Output file
    #[arg(short, long)]
    output: Option<PathBuf>,
    /// Output ICC file
    #[arg(long)]
    icc_output: Option<PathBuf>,
    /// Input file
    input: PathBuf,
    /// (unstable) Region to render, in format of 'width height left top'
    #[arg(long, value_parser = parse_crop_info)]
    crop: Option<CropInfo>,
    /// (unstable) Approximate memory limit, in bytes
    #[arg(long, default_value_t = 0)]
    approx_memory_limit: usize,
    /// Format to output
    #[arg(value_enum, short = 'f', long, default_value_t = OutputFormat::Png)]
    output_format: OutputFormat,
    /// Print debug information
    #[arg(short, long)]
    verbose: bool,
    /// Number of parallelism to use
    #[cfg(feature = "rayon")]
    #[arg(short = 'j', long)]
    num_threads: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum OutputFormat {
    /// PNG, respects bit depth information.
    Png,
    /// PNG, always 8-bit.
    Png8,
    /// PNG, always 16-bit.
    Png16,
    /// Numpy, used for conformance test.
    Npy,
}

fn parse_crop_info(s: &str) -> Result<CropInfo, std::num::ParseIntError> {
    let s = s.trim();
    let mut it = s.split_whitespace().map(|s| s.parse::<u32>());
    let Some(w) = it.next().transpose()? else {
        return Ok(CropInfo {
            width: 0,
            height: 0,
            left: 0,
            top: 0,
        });
    };
    let Some(h) = it.next().transpose()? else {
        return Ok(CropInfo {
            width: w,
            height: w,
            left: 0,
            top: 0,
        });
    };
    let Some(x) = it.next().transpose()? else {
        return Ok(CropInfo {
            width: w,
            height: h,
            left: 0,
            top: 0,
        });
    };
    let Some(y) = it.next().transpose()? else {
        return Ok(CropInfo {
            width: w,
            height: w,
            left: h,
            top: x,
        });
    };
    Ok(CropInfo {
        width: w,
        height: h,
        left: x,
        top: y,
    })
}

fn main() {
    let args = Args::parse();

    let filter = if args.verbose {
        tracing::level_filters::LevelFilter::DEBUG
    } else {
        tracing::level_filters::LevelFilter::INFO
    };
    let env_filter = tracing_subscriber::EnvFilter::builder()
        .with_default_directive(filter.into())
        .from_env_lossy();
    tracing_subscriber::fmt()
        .with_span_events(tracing_subscriber::fmt::format::FmtSpan::ACTIVE)
        .with_env_filter(env_filter)
        .init();

    let span = tracing::span!(tracing::Level::TRACE, "jxl-dec");
    let _guard = span.enter();

    #[cfg(feature = "rayon")]
    let pool = JxlThreadPool::rayon(args.num_threads);
    #[cfg(not(feature = "rayon"))]
    let pool = JxlThreadPool::none();

    let mut image_builder = JxlImage::builder().pool(pool.clone());
    if args.approx_memory_limit != 0 {
        let tracker = AllocTracker::with_limit(args.approx_memory_limit);
        image_builder = image_builder.alloc_tracker(tracker);
    }
    let mut image = image_builder
        .open(&args.input)
        .expect("Failed to open file");
    if !image.is_loading_done() {
        tracing::warn!("Partial image");
    }

    let image_size = &image.image_header().size;
    let image_meta = &image.image_header().metadata;
    tracing::info!("Image dimension: {}x{}", image.width(), image.height());
    tracing::debug!(colour_encoding = format_args!("{:?}", image_meta.colour_encoding));

    if let Some(icc_path) = &args.icc_output {
        tracing::debug!("Writing ICC profile");
        std::fs::write(icc_path, image.rendered_icc()).expect("Failed to write ICC profile");
    }

    let crop = args.crop.and_then(|crop| {
        if crop.width == 0 && crop.height == 0 {
            None
        } else if crop.width == 0 {
            Some(CropInfo {
                width: image_size.width,
                ..crop
            })
        } else if crop.height == 0 {
            Some(CropInfo {
                height: image_size.height,
                ..crop
            })
        } else {
            Some(crop)
        }
    });

    let (width, height) = if let Some(crop) = crop {
        tracing::debug!(crop = format_args!("{:?}", crop), "Cropped decoding");
        (crop.width, crop.height)
    } else {
        (image_size.width, image_size.height)
    };

    let decode_start = std::time::Instant::now();

    let mut keyframes = Vec::new();
    if args.output_format == OutputFormat::Npy {
        image.set_render_spot_colour(false);
    }

    #[allow(unused_mut)]
    let mut rendered = false;
    #[cfg(feature = "rayon")]
    if let Some(rayon_pool) = &pool.as_rayon_pool() {
        keyframes = rayon_pool
            .install(|| {
                use rayon::prelude::*;

                (0..image.num_loaded_keyframes())
                    .into_par_iter()
                    .map(|idx| image.render_frame_cropped(idx, crop))
                    .collect::<Result<Vec<_>, _>>()
            })
            .expect("rendering frames failed");
        rendered = true;
    }

    if !rendered {
        for idx in 0..image.num_loaded_keyframes() {
            let frame = image
                .render_frame_cropped(idx, crop)
                .expect("rendering frames failed");
            keyframes.push(frame);
        }
    }

    if let Ok(frame) = image.render_loading_frame_cropped(crop) {
        tracing::warn!("Rendered partially loaded frame");
        keyframes.push(frame);
    }

    let elapsed = decode_start.elapsed();
    let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
    tracing::info!("Took {:.2} ms", elapsed_ms);

    if let Some(output) = &args.output {
        if keyframes.is_empty() {
            tracing::warn!("No keyframes are decoded");
            return;
        }

        tracing::debug!(output_format = format_args!("{:?}", args.output_format));
        let pixel_format = image.pixel_format();
        let output = std::fs::File::create(output).expect("failed to open output file");
        match args.output_format {
            OutputFormat::Png => {
                write_png(
                    output,
                    &image,
                    &keyframes,
                    pixel_format,
                    None,
                    width,
                    height,
                );
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
                );
            }
            OutputFormat::Png16 => {
                write_png(
                    output,
                    &image,
                    &keyframes,
                    pixel_format,
                    Some(png::BitDepth::Eight),
                    width,
                    height,
                );
            }
            OutputFormat::Npy => {
                if args.icc_output.is_none() {
                    tracing::warn!("--icc-output is not set. Numpy buffer alone cannot be used to display image as its colorspace is unknown.");
                }

                write_npy(output, &image, &keyframes, width, height);
            }
        }
    } else {
        tracing::info!("No output path specified, skipping output encoding");
    };
}

fn write_png<W: Write>(
    output: W,
    image: &JxlImage,
    keyframes: &[Render],
    pixfmt: PixelFormat,
    force_bit_depth: Option<png::BitDepth>,
    width: u32,
    height: u32,
) {
    // Color encoding information
    let source_icc = image.rendered_icc();
    let embedded_icc = image.original_icc();
    let metadata = &image.image_header().metadata;
    let colour_encoding = &metadata.colour_encoding;
    let cicp = colour_encoding.cicp();

    let (width, height, _, _) = metadata.apply_orientation(width, height, 0, 0, false);
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

    let mut transform = None;
    let icc_cicp = if let Some(icc) = embedded_icc {
        if metadata.xyb_encoded {
            let source_profile = Profile::new_icc(&source_icc)
                .expect("Failed to create profile from jxl-oxide ICC profile");

            let target_profile = Profile::new_icc(icc);
            match target_profile {
                Err(err) => {
                    tracing::warn!("Embedded ICC has error: {}", err);
                    None
                }
                Ok(target_profile) => {
                    transform = Some(match color_type {
                        png::ColorType::Grayscale => LcmsTransform::Grayscale(
                            lcms2::Transform::new(
                                &source_profile,
                                lcms2::PixelFormat::GRAY_FLT,
                                &target_profile,
                                lcms2::PixelFormat::GRAY_FLT,
                                lcms2::Intent::RelativeColorimetric,
                            )
                            .expect("Failed to create transform"),
                        ),
                        png::ColorType::GrayscaleAlpha => LcmsTransform::GrayscaleAlpha(
                            lcms2::Transform::new(
                                &source_profile,
                                lcms2::PixelFormat(4390924 + 128), // GRAYA_FLT
                                &target_profile,
                                lcms2::PixelFormat(4390924 + 128), // GRAYA_FLT
                                lcms2::Intent::RelativeColorimetric,
                            )
                            .expect("Failed to create transform"),
                        ),
                        png::ColorType::Rgb => LcmsTransform::Rgb(
                            lcms2::Transform::new(
                                &source_profile,
                                lcms2::PixelFormat::RGB_FLT,
                                &target_profile,
                                lcms2::PixelFormat::RGB_FLT,
                                lcms2::Intent::RelativeColorimetric,
                            )
                            .expect("Failed to create transform"),
                        ),
                        png::ColorType::Rgba => LcmsTransform::Rgba(
                            lcms2::Transform::new(
                                &source_profile,
                                lcms2::PixelFormat::RGBA_FLT,
                                &target_profile,
                                lcms2::PixelFormat::RGBA_FLT,
                                lcms2::Intent::RelativeColorimetric,
                            )
                            .expect("Failed to create transform"),
                        ),
                        _ => unreachable!(),
                    });

                    Some((icc, None))
                }
            }
        } else {
            Some((icc, None))
        }
    } else {
        // TODO: emit gAMA and cHRM
        Some((&*source_icc, cicp))
    };
    encoder.validate_sequence(true);

    let mut writer = encoder.write_header().expect("failed to write header");

    if let Some((icc, cicp)) = &icc_cicp {
        tracing::debug!("Embedding ICC profile");
        let compressed_icc = miniz_oxide::deflate::compress_to_vec_zlib(icc, 7);
        let mut iccp_chunk_data = vec![b'0', 0, 0];
        iccp_chunk_data.extend(compressed_icc);
        writer
            .write_chunk(png::chunk::iCCP, &iccp_chunk_data)
            .expect("failed to write iCCP");

        if let Some(cicp) = *cicp {
            tracing::debug!(cicp = format_args!("{:?}", cicp), "Writing cICP chunk");
            writer
                .write_chunk(png::chunk::ChunkType([b'c', b'I', b'C', b'P']), &cicp)
                .expect("failed to write cICP");
        }
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

        let mut fb = keyframe.image();
        if let Some(transform) = &transform {
            transform.transform_in_place(&mut fb);
        }

        if sixteen_bits {
            let mut buf = vec![0u8; fb.width() * fb.height() * fb.channels() * 2];
            for (b, s) in buf.chunks_exact_mut(2).zip(fb.buf()) {
                let w = (*s * 65535.0 + 0.5).clamp(0.0, 65535.0) as u16;
                let [b0, b1] = w.to_be_bytes();
                b[0] = b0;
                b[1] = b1;
            }
            writer
                .write_image_data(&buf)
                .expect("failed to write frame");
        } else {
            let mut buf = vec![0u8; fb.width() * fb.height() * fb.channels()];
            for (b, s) in buf.iter_mut().zip(fb.buf()) {
                *b = (*s * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            }
            writer
                .write_image_data(&buf)
                .expect("failed to write frame");
        }
    }

    writer.finish().expect("failed to finish writing png");
}

fn write_npy<W: Write>(output: W, image: &JxlImage, keyframes: &[Render], width: u32, height: u32) {
    let metadata = &image.image_header().metadata;
    let (width, height, _, _) = metadata.apply_orientation(width, height, 0, 0, false);
    let channels = {
        let first_frame = keyframes.first().unwrap();
        first_frame.color_channels().len() + first_frame.extra_channels().len()
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
    output
        .write_all(&(header.len() as u16).to_le_bytes())
        .unwrap();
    output.write_all(header.as_bytes()).unwrap();

    tracing::debug!("Writing image data");
    for keyframe in keyframes {
        let fb = keyframe.image_all_channels();
        for sample in fb.buf() {
            output.write_all(&sample.to_bits().to_le_bytes()).unwrap();
        }
    }

    output.flush().unwrap();
}
