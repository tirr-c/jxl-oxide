use std::path::PathBuf;

use clap::Parser;
use jxl_bitstream::read_bits;
use jxl_color::RenderingIntent;
use jxl_frame::ProgressiveResult;
use jxl_grid::Grid;
use jxl_image::Headers;
use jxl_render::{FrameBuffer, RenderContext};
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

#[derive(Debug, Parser)]
#[command(version, about)]
struct Args {
    /// Output file
    #[arg(short, long)]
    output: Option<PathBuf>,
    /// Output ICC file
    #[arg(long)]
    icc_output: Option<PathBuf>,
    /// Input file
    input: PathBuf,
    #[arg(long, value_parser = str::parse::<CropInfo>)]
    crop: Option<CropInfo>,
    #[arg(long)]
    experimental_progressive: bool,
    #[arg(short, long)]
    verbose: bool,
}

#[derive(Debug, Default, Clone, Copy)]
struct CropInfo {
    width: u32,
    height: u32,
    left: u32,
    top: u32,
}

impl std::str::FromStr for CropInfo {
    type Err = std::num::ParseIntError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        let mut it = s.split_whitespace().map(|s| s.parse::<u32>());
        let Some(w) = it.next().transpose()? else {
            return Ok(Self {
                width: 0,
                height: 0,
                left: 0,
                top: 0,
            });
        };
        let Some(h) = it.next().transpose()? else {
            return Ok(Self {
                width: w,
                height: w,
                left: 0,
                top: 0,
            });
        };
        let Some(x) = it.next().transpose()? else {
            return Ok(Self {
                width: w,
                height: h,
                left: 0,
                top: 0,
            });
        };
        let Some(y) = it.next().transpose()? else {
            return Ok(Self {
                width: w,
                height: w,
                left: h,
                top: x,
            });
        };
        Ok(Self {
            width: w,
            height: h,
            left: x,
            top: y,
        })
    }
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

    let span = tracing::span!(tracing::Level::TRACE, "jxl_dec (main)");
    let _guard = span.enter();

    let file = std::fs::File::open(&args.input).expect("Failed to open file");
    let mut bitstream = jxl_bitstream::Bitstream::new_detect(file);
    let headers = read_bits!(bitstream, Bundle(Headers)).expect("Failed to read headers");
    tracing::info!("Image dimension: {}x{}", headers.size.width, headers.size.height);

    let mut render = RenderContext::new(&headers);
    tracing::debug!(colour_encoding = format_args!("{:?}", headers.metadata.colour_encoding));
    let icc = render.read_icc_if_exists(&mut bitstream).expect("failed to decode ICC");
    let icc = if icc.is_empty() {
        None
    } else {
        Some(icc.to_vec())
    };
    if let Some(icc_path) = &args.icc_output {
        if let Some(icc) = &icc {
            tracing::info!("Writing ICC profile");
            std::fs::write(icc_path, icc).expect("Failed to write ICC profile");
        } else {
            tracing::warn!("Input does not have embedded ICC profile, ignoring --icc-output");
        }
    }

    if headers.metadata.preview.is_some() {
        bitstream.zero_pad_to_byte().expect("Zero-padding failed");

        let frame = read_bits!(bitstream, Bundle(jxl_frame::Frame), &headers).expect("Failed to read frame header");

        let toc = frame.toc();
        let bookmark = toc.bookmark() + (toc.total_byte_size() * 8);
        bitstream.skip_to_bookmark(bookmark).expect("Failed to skip");
    }

    let mut crop = args.crop.and_then(|crop| {
        if crop.width == 0 && crop.height == 0 {
            None
        } else if crop.width == 0 {
            Some(CropInfo {
                width: headers.size.width,
                ..crop
            })
        } else if crop.height == 0 {
            Some(CropInfo {
                height: headers.size.height,
                ..crop
            })
        } else {
            Some(crop)
        }
    });

    if let Some(crop) = &mut crop {
        tracing::debug!(crop = format_args!("{:?}", crop), "Cropped decoding");
        let (w, h, x, y) = headers.metadata.apply_orientation(
            crop.width,
            crop.height,
            crop.left,
            crop.top,
            true,
        );
        crop.width = w;
        crop.height = h;
        crop.left = x;
        crop.top = y
    }

    if args.experimental_progressive {
        if let Some(path) = &args.output {
            std::fs::create_dir_all(path).expect("cannot create directory");
        }
    }

    let (width, height) = if let Some(crop) = crop {
        (crop.width, crop.height)
    } else {
        (headers.size.width, headers.size.height)
    };

    let decode_start = std::time::Instant::now();

    loop {
        let result = run(&mut bitstream, &mut render, crop, args.experimental_progressive);

        let keyframe_idx = render.loaded_keyframes() - 1;
        let frame = render.keyframe(keyframe_idx).unwrap();
        if result == ProgressiveResult::FrameComplete && frame.header().is_last {
            break;
        }
    }

    let elapsed = decode_start.elapsed();
    let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
    tracing::info!("Took {:.2} ms", elapsed_ms);

    if let Some(output) = &args.output {
        let output = std::fs::File::create(output).expect("failed to open output file");
        let (width, height, _, _) = headers.metadata.apply_orientation(width, height, 0, 0, false);
        let mut encoder = png::Encoder::new(output, width, height);

        let has_alpha_channel = headers.metadata.alpha().is_some();
        if has_alpha_channel {
            tracing::debug!("Image has alpha channel");
        }
        let color_type = match (headers.metadata.grayscale(), has_alpha_channel) {
            (false, false) => png::ColorType::Rgb,
            (false, true) => png::ColorType::Rgba,
            (true, false) => png::ColorType::Grayscale,
            (true, true) => png::ColorType::GrayscaleAlpha,
        };
        encoder.set_color(color_type);

        let sixteen_bits = headers.metadata.bit_depth.bits_per_sample() > 8;
        if sixteen_bits {
            encoder.set_depth(png::BitDepth::Sixteen);
        } else {
            encoder.set_depth(png::BitDepth::Eight);
        }

        let keyframes = render.loaded_keyframes();
        if let Some(animation) = &headers.metadata.animation {
            let num_plays = animation.num_loops;
            encoder.set_animated(keyframes as u32, num_plays).unwrap();
        }

        let mut transform = None;
        let colour_encoding = &headers.metadata.colour_encoding;
        let icc_cicp = if let Some(icc) = icc {
            if headers.metadata.xyb_encoded {
                let source_icc = jxl_color::icc::colour_encoding_to_icc(colour_encoding).expect("failed to build ICC profile");
                let source_profile = Profile::new_icc(&source_icc).expect("Failed to create profile from jxl-oxide ICC profile");

                let target_profile = Profile::new_icc(&icc);
                match target_profile {
                    Err(err) => {
                        tracing::warn!("Embedded ICC has error: {}", err);
                        None
                    },
                    Ok(target_profile) => {
                        transform = Some(match color_type {
                            png::ColorType::Grayscale => LcmsTransform::Grayscale(lcms2::Transform::new(
                                &source_profile,
                                lcms2::PixelFormat::GRAY_FLT,
                                &target_profile,
                                lcms2::PixelFormat::GRAY_FLT,
                                lcms2::Intent::RelativeColorimetric,
                            ).expect("Failed to create transform")),
                            png::ColorType::GrayscaleAlpha => LcmsTransform::GrayscaleAlpha(lcms2::Transform::new(
                                &source_profile,
                                lcms2::PixelFormat(4390924 + 128), // GRAYA_FLT
                                &target_profile,
                                lcms2::PixelFormat(4390924 + 128), // GRAYA_FLT
                                lcms2::Intent::RelativeColorimetric,
                            ).expect("Failed to create transform")),
                            png::ColorType::Rgb => LcmsTransform::Rgb(lcms2::Transform::new(
                                &source_profile,
                                lcms2::PixelFormat::RGB_FLT,
                                &target_profile,
                                lcms2::PixelFormat::RGB_FLT,
                                lcms2::Intent::RelativeColorimetric,
                            ).expect("Failed to create transform")),
                            png::ColorType::Rgba => LcmsTransform::Rgba(lcms2::Transform::new(
                                &source_profile,
                                lcms2::PixelFormat::RGBA_FLT,
                                &target_profile,
                                lcms2::PixelFormat::RGBA_FLT,
                                lcms2::Intent::RelativeColorimetric,
                            ).expect("Failed to create transform")),
                            _ => unreachable!(),
                        });

                        Some((icc, None))
                    },
                }
            } else {
                Some((icc, None))
            }
        } else if colour_encoding.is_srgb() {
            encoder.set_srgb(match colour_encoding.rendering_intent {
                RenderingIntent::Perceptual => png::SrgbRenderingIntent::Perceptual,
                RenderingIntent::Relative => png::SrgbRenderingIntent::RelativeColorimetric,
                RenderingIntent::Saturation => png::SrgbRenderingIntent::Saturation,
                RenderingIntent::Absolute => png::SrgbRenderingIntent::AbsoluteColorimetric,
            });

            None
        } else {
            let icc = jxl_color::icc::colour_encoding_to_icc(colour_encoding).expect("failed to build ICC profile");
            let cicp = colour_encoding.cicp();
            // TODO: emit gAMA and cHRM
            Some((icc, cicp))
        };
        encoder.validate_sequence(true);

        let mut writer = encoder
            .write_header()
            .expect("failed to write header");

        if let Some((icc, cicp)) = &icc_cicp {
            tracing::debug!("Embedding ICC profile");
            let compressed_icc = miniz_oxide::deflate::compress_to_vec_zlib(icc, 7);
            let mut iccp_chunk_data = vec![b'0', 0, 0];
            iccp_chunk_data.extend(compressed_icc);
            writer.write_chunk(png::chunk::iCCP, &iccp_chunk_data).expect("failed to write iCCP");

            if let Some(cicp) = *cicp {
                tracing::debug!(cicp = format_args!("{:?}", cicp), "Writing cICP chunk");
                writer.write_chunk(png::chunk::ChunkType([b'c', b'I', b'C', b'P']), &cicp).expect("failed to write cICP");
            }
        }

        let region = crop.as_ref().map(|crop| (crop.left, crop.top, crop.width, crop.height));

        tracing::debug!("Writing image data");
        for keyframe_idx in 0..keyframes {
            if let Some(animation) = &headers.metadata.animation {
                let frame = render.keyframe(keyframe_idx).unwrap();
                let duration = frame.header().duration;
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

            let mut grids = render.render_keyframe_cropped(keyframe_idx, region).unwrap();
            filter_alpha_channel(&mut grids, &headers);
            let grids = grids.into_iter().map(Grid::from).collect::<Vec<_>>();
            let mut fb = FrameBuffer::from_grids(&grids, headers.metadata.orientation).unwrap();
            if let Some(transform) = &transform {
                transform.transform_in_place(&mut fb);
            }

            if sixteen_bits {
                let mut buf = vec![0u8; fb.width() * fb.height() * fb.channels() * 2];
                for (b, s) in buf.chunks_exact_mut(2).zip(fb.buf()) {
                    let w = (*s * 65535.0).clamp(0.0, 65535.0) as u16;
                    let [b0, b1] = w.to_be_bytes();
                    b[0] = b0;
                    b[1] = b1;
                }
                writer.write_image_data(&buf).expect("failed to write frame");
            } else {
                let mut buf = vec![0u8; fb.width() * fb.height() * fb.channels()];
                for (b, s) in buf.iter_mut().zip(fb.buf()) {
                    *b = (*s * 255.0).clamp(0.0, 255.0) as u8;
                }
                writer.write_image_data(&buf).expect("failed to write frame");
            }
        }

        writer.finish().expect("failed to finish writing png");
    } else {
        tracing::info!("No output path specified, skipping PNG encoding");
    };
}

fn run<R: std::io::Read>(
    bitstream: &mut jxl_bitstream::Bitstream<R>,
    render: &mut RenderContext,
    crop: Option<CropInfo>,
    progressive: bool,
) -> ProgressiveResult {
    let region = crop.as_ref().map(|crop| (crop.left, crop.top, crop.width, crop.height));

    let result = render
        .load_until_keyframe(bitstream, progressive, region)
        .expect("failed to load frames");
    let keyframe_idx = render.loaded_keyframes() - 1;
    render.render_keyframe_cropped(keyframe_idx, region).expect("failed to render");

    result
}

fn filter_alpha_channel(grids: &mut Vec<jxl_grid::SimpleGrid<f32>>, headers: &Headers) {
    let color_channels = if headers.metadata.grayscale() { 1 } else { 3 };
    let alpha_channel = headers.metadata.alpha();
    if let Some(idx) = alpha_channel {
        let alpha_premultiplied = headers.metadata.ec_info[idx].alpha_associated().unwrap();
        if alpha_premultiplied {
            tracing::warn!("Premultiplied alpha is not supported for output");
        }

        let alpha = grids.drain(color_channels..).nth(idx).unwrap();
        grids.push(alpha);
    } else {
        grids.drain(color_channels..);
    }
}
