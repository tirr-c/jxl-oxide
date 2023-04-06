use std::path::PathBuf;

use clap::Parser;
use jxl_bitstream::read_bits;
use jxl_frame::ProgressiveResult;
use jxl_image::{FrameBuffer, Headers, ExtraChannelType};
use jxl_render::RenderContext;

#[derive(Debug, Parser)]
#[command(version, about)]
struct Args {
    /// Output file
    #[arg(short, long)]
    output: Option<PathBuf>,
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
    render.read_icc_if_exists(&mut bitstream).expect("failed to decode ICC");

    if headers.metadata.have_preview {
        bitstream.zero_pad_to_byte().expect("Zero-padding failed");

        let frame = read_bits!(bitstream, Bundle(jxl_frame::Frame), &headers).expect("Failed to read frame header");

        let toc = frame.toc();
        let bookmark = toc.bookmark() + (toc.total_byte_size() * 8);
        bitstream.skip_to_bookmark(bookmark).expect("Failed to skip");
    }

    let crop = args.crop.and_then(|crop| {
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

    if let Some(crop) = &crop {
        tracing::debug!(crop = format_args!("{:?}", crop), "Cropped decoding");
    }

    if args.experimental_progressive {
        if let Some(path) = &args.output {
            std::fs::create_dir_all(path).expect("cannot create directory");
        }
    }

    let mut idx = 0usize;
    loop {
        let (result, fb) = run(&mut bitstream, &mut render, &headers, crop, args.experimental_progressive);

        if let Some(output) = &args.output {
            let mut output = output.clone();
            if args.experimental_progressive {
                output.push(format!("{}.png", idx));
            }

            tracing::info!("Encoding samples to PNG");
            let width = fb.width();
            let height = fb.height();
            let output = std::fs::File::create(output).expect("failed to open output file");
            let mut encoder = png::Encoder::new(output, width as u32, height as u32);
            encoder.set_color(if fb.channels() == 4 { png::ColorType::Rgba } else { png::ColorType::Rgb });

            let sixteen_bits = headers.metadata.bit_depth.bits_per_sample() > 8;
            if sixteen_bits {
                encoder.set_depth(png::BitDepth::Sixteen);
            } else {
                encoder.set_depth(png::BitDepth::Eight);
            }

            let colour_encoding = &headers.metadata.colour_encoding;
            let icc_cicp = if colour_encoding.is_srgb() {
                encoder.set_srgb(match colour_encoding.rendering_intent {
                    jxl_image::RenderingIntent::Perceptual => png::SrgbRenderingIntent::Perceptual,
                    jxl_image::RenderingIntent::Relative => png::SrgbRenderingIntent::RelativeColorimetric,
                    jxl_image::RenderingIntent::Saturation => png::SrgbRenderingIntent::Saturation,
                    jxl_image::RenderingIntent::Absolute => png::SrgbRenderingIntent::AbsoluteColorimetric,
                });

                None
            } else {
                let icc = jxl_color::icc::colour_encoding_to_icc(colour_encoding).expect("failed to build ICC profile");
                let cicp = colour_encoding.png_cicp();
                // TODO: emit gAMA and cHRM
                Some((icc, cicp))
            };

            let mut writer = encoder
                .write_header()
                .expect("failed to write header");

            if let Some((icc, cicp)) = icc_cicp {
                tracing::debug!("Embedding ICC profile");
                let compressed_icc = miniz_oxide::deflate::compress_to_vec_zlib(&icc, 7);
                let mut iccp_chunk_data = vec![b' ', 0, 0];
                iccp_chunk_data.extend(compressed_icc);
                writer.write_chunk(png::chunk::iCCP, &iccp_chunk_data).expect("failed to write iCCP");

                if let Some(cicp) = cicp {
                    tracing::debug!(cicp = format_args!("{:?}", cicp), "Writing cICP chunk");
                    writer.write_chunk(png::chunk::ChunkType([b'c', b'I', b'C', b'P']), &cicp).expect("failed to write cICP");
                }
            }

            let mut writer = writer.into_stream_writer().unwrap();

            tracing::debug!("Writing image data");
            if sixteen_bits {
                let mut buf = vec![0u8; fb.width() * fb.channels() * 2];
                for row in fb.buf().chunks(fb.width() * fb.channels()) {
                    for (s, b) in row.iter().zip(buf.chunks_exact_mut(2)) {
                        let w = (*s * 65535.0).clamp(0.0, 65535.0) as u16;
                        let [b0, b1] = w.to_be_bytes();
                        b[0] = b0;
                        b[1] = b1;
                    }
                    std::io::Write::write_all(&mut writer, &buf[..row.len() * 2]).expect("failed to write image data");
                }
            } else {
                let mut buf = vec![0u8; fb.width() * fb.channels()];
                for row in fb.buf().chunks(fb.width() * fb.channels()) {
                    for (s, b) in row.iter().zip(&mut buf) {
                        *b = (*s * 255.0).clamp(0.0, 255.0) as u8;
                    }
                    std::io::Write::write_all(&mut writer, &buf[..row.len()]).expect("failed to write image data");
                }
            }
            writer.finish().expect("failed to finish writing png");

            idx += 1;
        } else {
            tracing::info!("No output path specified, skipping PNG encoding");
        }

        if result == ProgressiveResult::FrameComplete {
            break;
        }
    }
}

fn run<R: std::io::Read>(
    bitstream: &mut jxl_bitstream::Bitstream<R>,
    render: &mut RenderContext,
    headers: &Headers,
    crop: Option<CropInfo>,
    progressive: bool,
) -> (ProgressiveResult, FrameBuffer) {
    let region = crop.as_ref().map(|crop| (crop.left, crop.top, crop.width, crop.height));

    let decode_start = std::time::Instant::now();
    let result = render
        .load_cropped(bitstream, progressive, region)
        .expect("failed to load frames");
    let fb = render.render_cropped(region).expect("failed to render");

    let mut grids = fb.into_iter().map(From::from).collect::<Vec<_>>();
    let alpha_channel = headers.metadata.ec_info.iter().position(|ec| ec.ty == ExtraChannelType::Alpha);
    if let Some(idx) = alpha_channel {
        tracing::debug!("Image has alpha channel");

        let alpha_premultiplied = headers.metadata.ec_info[idx].alpha_associated;
        if alpha_premultiplied {
            tracing::warn!("Premultiplied alpha is not supported for output");
        }

        let alpha = grids.drain(3..).nth(idx).unwrap();
        grids.push(alpha);
    } else {
        grids.drain(3..);
    }

    let image = FrameBuffer::from_grids(&grids).unwrap();
    let elapsed = decode_start.elapsed();

    let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
    tracing::info!("Took {:.2} ms", elapsed_ms);

    (result, image)
}
