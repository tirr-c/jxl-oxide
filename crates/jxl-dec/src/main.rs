use std::{path::PathBuf, fs::File};

use clap::Parser;
use jxl_bitstream::read_bits;
use jxl_frame::ProgressiveResult;
use jxl_image::{FrameBuffer, Headers, TransferFunction};
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
    tracing_subscriber::fmt()
        .with_span_events(tracing_subscriber::fmt::format::FmtSpan::ACTIVE)
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let span = tracing::span!(tracing::Level::TRACE, "jxl_dec (main)");
    let _guard = span.enter();

    let args = Args::parse();

    let file = std::fs::File::open(&args.input).expect("Failed to open file");
    let mut bitstream = jxl_bitstream::Bitstream::new(file);
    let headers = read_bits!(bitstream, Bundle(Headers)).expect("Failed to read headers");

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
        let (result, mut fb) = run(&mut bitstream, &mut render, &headers, crop, args.experimental_progressive);

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
            encoder.set_depth(png::BitDepth::Eight);
            // TODO: set colorspace
            encoder.set_srgb(match headers.metadata.colour_encoding.rendering_intent {
                jxl_image::RenderingIntent::Perceptual => png::SrgbRenderingIntent::Perceptual,
                jxl_image::RenderingIntent::Relative => png::SrgbRenderingIntent::RelativeColorimetric,
                jxl_image::RenderingIntent::Saturation => png::SrgbRenderingIntent::Saturation,
                jxl_image::RenderingIntent::Absolute => png::SrgbRenderingIntent::AbsoluteColorimetric,
            });
            let mut writer = encoder
                .write_header()
                .expect("failed to write header")
                .into_stream_writer()
                .unwrap();

            if headers.metadata.xyb_encoded || headers.metadata.colour_encoding.tf == TransferFunction::Linear {
                jxl_color::tf::linear_to_srgb(fb.buf_mut());
            }

            let mut buf = vec![0u8; fb.width() * fb.channels()];
            for row in fb.buf().chunks(fb.width() * fb.channels()) {
                for (s, b) in row.iter().zip(&mut buf) {
                    *b = (*s * 255.0).clamp(0.0, 255.0) as u8;
                }
                std::io::Write::write_all(&mut writer, &buf[..row.len()]).expect("failed to write image data");
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

fn run(
    bitstream: &mut jxl_bitstream::Bitstream<File>,
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
    let mut fb = render.render_cropped(region).expect("failed to render");

    if headers.metadata.xyb_encoded {
        let fb_yxb = {
            let mut it = fb.iter_mut();
            [
                it.next().unwrap(),
                it.next().unwrap(),
                it.next().unwrap(),
            ]
        };
        jxl_color::xyb::perform_inverse_xyb(fb_yxb, &headers.metadata);
    }

    let grids = fb.into_iter().map(From::from).collect::<Vec<_>>();
    let image = FrameBuffer::from_grids(&grids).unwrap();
    let elapsed = decode_start.elapsed();

    let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
    tracing::info!(elapsed_ms, "Took {:.2} ms", elapsed_ms);

    (result, image)
}
