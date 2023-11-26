use std::path::PathBuf;

use clap::Parser;
use jxl_oxide::JxlImage;

/// Prints information about JPEG XL image.
#[derive(Debug, Parser)]
#[command(version)]
struct Args {
    /// Input file
    input: PathBuf,
    /// Output information of all frames in addition to keyframes
    #[arg(long)]
    all_frames: bool,
    /// Output group sizes and offsets
    #[arg(long)]
    with_offset: bool,
    /// Print debug information
    #[arg(short, long)]
    verbose: bool,
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

    let span = tracing::span!(tracing::Level::TRACE, "jxl-info");
    let _guard = span.enter();

    let image = JxlImage::builder()
        .open(&args.input)
        .expect("Failed to open file");
    let image_size = &image.image_header().size;
    let image_meta = &image.image_header().metadata;
    let image_reader = image.reader();

    println!("JPEG XL image ({:?})", image_reader.kind());

    println!("  Image dimension: {}x{}", image.width(), image.height());
    if image_meta.orientation != 1 {
        println!(
            "    Encoded image dimension: {}x{}",
            image_size.width, image_size.height
        );
        print!("    Orientation of encoded image: ");
        match image_meta.orientation {
            2 => println!("flipped horizontally"),
            3 => println!("rotated 180 degrees"),
            4 => println!("flipped vertically"),
            5 => println!("transposed"),
            6 => println!("rotated 90 degrees CCW"),
            7 => println!("rotated 90 degrees CCW, and then flipped horizontally"),
            8 => println!("rotated 90 degrees CW"),
            _ => {}
        }
    }

    println!(
        "  Bit depth: {} bits",
        image_meta.bit_depth.bits_per_sample()
    );
    if image_meta.xyb_encoded {
        println!("  XYB encoded, suggested display color encoding:");
    } else {
        println!("  Color encoding:");
    }
    if let Some(icc) = image.original_icc() {
        print!("    ");
        if image_meta.grayscale() {
            print!("Grayscale, embedded ICC profile ({} bytes)", icc.len());
        } else {
            println!("Embedded ICC profile ({} bytes)", icc.len());
        }
    } else {
        let colour_encoding = &image_meta.colour_encoding;
        print!("    Colorspace: ");
        match colour_encoding.colour_space {
            jxl_oxide::color::ColourSpace::Rgb => println!("RGB"),
            jxl_oxide::color::ColourSpace::Grey => println!("Grayscale"),
            jxl_oxide::color::ColourSpace::Xyb => println!("XYB"),
            jxl_oxide::color::ColourSpace::Unknown => println!("Unknown"),
        }

        print!("    White point: ");
        match colour_encoding.white_point {
            jxl_oxide::color::WhitePoint::D65 => println!("D65"),
            jxl_oxide::color::WhitePoint::Custom(xy) => {
                println!("{}, {}", xy.x as f64 / 10e6, xy.y as f64 / 10e6)
            }
            jxl_oxide::color::WhitePoint::E => println!("E"),
            jxl_oxide::color::WhitePoint::Dci => println!("DCI"),
        }

        print!("    Primaries: ");
        match colour_encoding.primaries {
            jxl_oxide::color::Primaries::Srgb => println!("sRGB"),
            jxl_oxide::color::Primaries::Custom { red, green, blue } => {
                println!(
                    "{}, {}; {}, {}; {}, {}",
                    red.x as f64 / 10e6,
                    red.y as f64 / 10e6,
                    green.x as f64 / 10e6,
                    green.y as f64 / 10e6,
                    blue.x as f64 / 10e6,
                    blue.y as f64 / 10e6,
                );
            }
            jxl_oxide::color::Primaries::Bt2100 => println!("BT.2100"),
            jxl_oxide::color::Primaries::P3 => println!("P3"),
        }

        print!("    Transfer function: ");
        match colour_encoding.tf {
            jxl_oxide::color::TransferFunction::Gamma(g) => println!("Gamma {}", g as f64 / 10e7),
            jxl_oxide::color::TransferFunction::Bt709 => println!("BT.709"),
            jxl_oxide::color::TransferFunction::Unknown => println!("Unknown"),
            jxl_oxide::color::TransferFunction::Linear => println!("Linear"),
            jxl_oxide::color::TransferFunction::Srgb => println!("sRGB"),
            jxl_oxide::color::TransferFunction::Pq => println!("PQ (HDR)"),
            jxl_oxide::color::TransferFunction::Dci => println!("DCI"),
            jxl_oxide::color::TransferFunction::Hlg => println!("Hybrid log-gamma (HDR)"),
        }
    }

    if let Some(animation) = &image_meta.animation {
        println!(
            "  Animated ({}/{} ticks per second)",
            animation.tps_numerator, animation.tps_denominator
        );
    }

    let animated = image_meta.animation.is_some();
    for idx in 0..image.num_loaded_frames() + 1 {
        let Some(frame) = image.frame(idx) else {
            break;
        };
        let frame_header = frame.header();
        let is_keyframe = frame_header.is_keyframe();
        if !args.all_frames && !is_keyframe {
            continue;
        }

        print!("Frame #{idx}");
        if is_keyframe {
            print!(" (keyframe)");
        }
        if !frame.is_loading_done() {
            print!(" (partial)");
        }
        println!();

        if !frame_header.name.is_empty() {
            println!("  Name: {}", &*frame_header.name);
        }
        match frame_header.encoding {
            jxl_oxide::frame::Encoding::VarDct => println!("  VarDCT (lossy)"),
            jxl_oxide::frame::Encoding::Modular => println!("  Modular (maybe lossless)"),
        }
        print!("  Frame type: ");
        match frame_header.frame_type {
            jxl_oxide::frame::FrameType::RegularFrame => print!("Regular"),
            jxl_oxide::frame::FrameType::LfFrame => print!(
                "LF, level {} ({}x downsampled)",
                frame_header.lf_level,
                1 << (frame_header.lf_level * 3)
            ),
            jxl_oxide::frame::FrameType::ReferenceOnly => {
                print!("Reference only, slot {}", frame_header.save_as_reference)
            }
            jxl_oxide::frame::FrameType::SkipProgressive => {
                print!("Regular (skip progressive rendering)")
            }
        }
        println!();

        print!(
            "  {}x{}",
            frame_header.color_sample_width(),
            frame_header.color_sample_height(),
        );
        if frame_header.frame_type.is_normal_frame() {
            print!("; ({}, {})", frame_header.x0, frame_header.y0);
        }
        println!();

        if frame_header.do_ycbcr {
            println!(
                "  YCbCr, upsampling factor: {:?}",
                frame_header.jpeg_upsampling
            );
        }
        if animated && is_keyframe {
            if frame_header.duration == 0xffffffff {
                println!("  End of a page");
            } else {
                println!(
                    "  Duration: {} tick{}",
                    frame_header.duration,
                    if frame_header.duration == 1 { "" } else { "s" }
                );
            }
        }

        if args.with_offset {
            let offset = image.frame_offset(idx).unwrap();
            println!(
                "  Offset (in codestream): {offset} (0x{offset:x})",
                offset = offset
            );

            let toc = frame.toc();
            println!(
                "  Frame header size: {size} (0x{size:x}) byte{plural}",
                size = toc.bookmark(),
                plural = if toc.bookmark() == 1 { "" } else { "s" },
            );
            println!("  Group sizes, in bitstream order:");
            for group in toc.iter_bitstream_order() {
                println!(
                    "    {:?}: {size} (0x{size:x}) byte{plural}",
                    group.kind,
                    size = group.size,
                    plural = if group.size == 1 { "" } else { "s" },
                );
            }
        }
    }

    if !image.is_loading_done() {
        println!("Partial file");
    }
}
