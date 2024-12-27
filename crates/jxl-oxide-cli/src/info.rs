use jxl_oxide::color::*;
use jxl_oxide::frame::*;
use jxl_oxide::image::BitDepth;
use jxl_oxide::{
    AuxBoxData, ColorEncodingWithProfile, ExtraChannelType, JpegReconstructionStatus, JxlImage,
};

use crate::{commands::info::*, Error, Result};

pub fn handle_info(args: InfoArgs) -> Result<()> {
    let _guard = tracing::trace_span!("Handle info subcommand").entered();

    let image = JxlImage::builder()
        .open(&args.input)
        .map_err(Error::ReadJxl)?;
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

    print_bit_depth(image_meta.bit_depth, "  ");

    if image_meta.xyb_encoded {
        println!("  XYB encoded, suggested display color encoding:");
    } else {
        println!("  Color encoding:");
    }
    match &image_meta.colour_encoding {
        ColourEncoding::Enum(colour_encoding) => {
            print_colour_encoding(colour_encoding, "    ");
        }
        ColourEncoding::IccProfile(colour_space) => {
            let icc = image.original_icc().unwrap();
            print!("    ");
            if *colour_space == ColourSpace::Grey {
                println!("Grayscale, embedded ICC profile ({} bytes)", icc.len());
            } else {
                println!("Embedded ICC profile ({} bytes)", icc.len());
            }

            if let Ok(encoding) = ColorEncodingWithProfile::with_icc(icc) {
                if let ColourEncoding::Enum(encoding) = encoding.encoding() {
                    print_colour_encoding(encoding, "      ");
                }
            }
        }
    }

    if let Some(animation) = &image_meta.animation {
        println!(
            "  Animated ({}/{} ticks per second)",
            animation.tps_numerator, animation.tps_denominator
        );
    }

    if !image_meta.ec_info.is_empty() {
        println!("Extra channel info:");
        for (ec_idx, ec) in image_meta.ec_info.iter().enumerate() {
            print!("  #{ec_idx} ");
            if !ec.name.is_empty() {
                print!("{} ", &*ec.name);
            }

            match &ec.ty {
                ExtraChannelType::Alpha { alpha_associated } => {
                    print!("Alpha");
                    if *alpha_associated {
                        println!(" (premultiplied)");
                    } else {
                        println!();
                    }
                }
                ExtraChannelType::Depth => println!("Depth"),
                ExtraChannelType::SpotColour {
                    red,
                    green,
                    blue,
                    solidity,
                } => {
                    println!("Spot Color ({red}, {green}, {blue})/{solidity}");
                }
                ExtraChannelType::SelectionMask => println!("Selection Mask"),
                ExtraChannelType::Black => println!("Black"),
                ExtraChannelType::Cfa { cfa_channel } => {
                    println!("Color Filter Array of channel {cfa_channel}");
                }
                ExtraChannelType::Thermal => println!("Thermal"),
                ExtraChannelType::NonOptional => {
                    println!("Unknown non-optional channel");
                }
                ExtraChannelType::Optional => println!("Unknown optional channel"),
            }

            if ec.bit_depth != image_meta.bit_depth {
                print_bit_depth(ec.bit_depth, "    ");
            }

            if ec.dim_shift > 0 {
                println!("    {}x downsampled", 1 << ec.dim_shift);
            }
        }
    }

    let jbrd_status = match image.jpeg_reconstruction_status() {
        JpegReconstructionStatus::Available => Some("available"),
        JpegReconstructionStatus::Invalid => Some("invalid"),
        JpegReconstructionStatus::Unavailable => None,
        JpegReconstructionStatus::NeedMoreData => Some("partial"),
    };
    if let Some(status) = jbrd_status {
        println!("JPEG bitstream reconstruction data: {status}");
    }

    let aux_boxes = image.aux_boxes();
    match aux_boxes.first_exif() {
        Ok(AuxBoxData::Data(exif)) => {
            let data = exif.payload();
            let size = data.len();
            println!("Exif metadata: {size} byte(s)");
        }
        Ok(_) => {}
        Err(e) => {
            println!("Invalid Exif metadata: {e}");
        }
    }
    if let AuxBoxData::Data(data) = aux_boxes.first_xml() {
        let size = data.len();
        println!("XML metadata: {size} byte(s)");
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
            Encoding::VarDct => println!("  VarDCT (lossy)"),
            Encoding::Modular => println!("  Modular (maybe lossless)"),
        }
        print!("  Frame type: ");
        match frame_header.frame_type {
            FrameType::RegularFrame => print!("Regular"),
            FrameType::LfFrame => print!(
                "LF, level {} ({}x downsampled)",
                frame_header.lf_level,
                1 << (frame_header.lf_level * 3)
            ),
            FrameType::ReferenceOnly => {
                print!("Reference only, slot {}", frame_header.save_as_reference)
            }
            FrameType::SkipProgressive => {
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

    Ok(())
}

fn print_colour_encoding(encoding: &EnumColourEncoding, indent: &str) {
    print!("{indent}Colorspace: ");
    match encoding.colour_space {
        ColourSpace::Rgb => println!("RGB"),
        ColourSpace::Grey => println!("Grayscale"),
        ColourSpace::Xyb => println!("XYB"),
        ColourSpace::Unknown => println!("Unknown"),
    }

    print!("{indent}White point: ");
    match encoding.white_point {
        WhitePoint::D65 => println!("D65"),
        WhitePoint::Custom(xy) => {
            println!("{}, {}", xy.x as f64 / 1e6, xy.y as f64 / 1e6)
        }
        WhitePoint::E => println!("E"),
        WhitePoint::Dci => println!("DCI"),
    }

    print!("{indent}Primaries: ");
    match encoding.primaries {
        Primaries::Srgb => println!("sRGB"),
        Primaries::Custom { red, green, blue } => {
            println!(
                "{}, {}; {}, {}; {}, {}",
                red.x as f64 / 1e6,
                red.y as f64 / 1e6,
                green.x as f64 / 1e6,
                green.y as f64 / 1e6,
                blue.x as f64 / 1e6,
                blue.y as f64 / 1e6,
            );
        }
        Primaries::Bt2100 => println!("BT.2100"),
        Primaries::P3 => println!("P3"),
    }

    print!("{indent}Transfer function: ");
    match encoding.tf {
        TransferFunction::Gamma { g, inverted: false } => {
            println!("Gamma {}", g as f64 / 1e7)
        }
        TransferFunction::Gamma { g, inverted: true } => {
            println!("Gamma {}", 1e7 / g as f64)
        }
        TransferFunction::Bt709 => println!("BT.709"),
        TransferFunction::Unknown => println!("Unknown"),
        TransferFunction::Linear => println!("Linear"),
        TransferFunction::Srgb => println!("sRGB"),
        TransferFunction::Pq => println!("PQ (HDR)"),
        TransferFunction::Dci => println!("DCI"),
        TransferFunction::Hlg => println!("Hybrid log-gamma (HDR)"),
    }
}

fn print_bit_depth(bit_depth: BitDepth, indent: &str) {
    print!("{indent}Bit depth: {} bits", bit_depth.bits_per_sample());
    if let BitDepth::FloatSample {
        bits_per_sample,
        exp_bits,
    } = bit_depth
    {
        println!(
            " (floating-point, {} mantissa bits)",
            bits_per_sample - exp_bits - 1
        );
    } else {
        println!();
    }
}
