use std::{io::prelude::*, path::PathBuf};

use clap::Parser;
use jxl_oxide::{
    color::{
        ColourSpace, Customxy, EnumColourEncoding, Primaries, RenderingIntent, TransferFunction,
        WhitePoint,
    },
    AllocTracker, CropInfo, FrameBuffer, JxlImage, JxlThreadPool, PixelFormat, Render,
};

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
    /// (unstable) Target colorspace specification
    ///
    /// Specification string consists of (optional) preset and a sequence of parameters delimited by commas.
    ///
    /// Parameters have a syntax of `name=value`. Possible parameter names:
    /// - type:   Color space type. Possible values:
    ///           - rgb
    ///           - gray
    ///           - xyb
    /// - gamut:  Color gamut. Invalid if type is Gray or XYB. Possible values:
    ///           - srgb
    ///           - p3
    ///           - bt2100
    /// - wp:     White point. Invalid if type is XYB. Possible values:
    ///           - d65
    ///           - d50
    ///           - dci
    ///           - e
    /// - tf:     Transfer function. Invalid if type is XYB. Possible values:
    ///           - srgb
    ///           - bt709
    ///           - dci
    ///           - pq
    ///           - hlg
    ///           - linear
    ///           - (gamma value)
    /// - intent: Rendering intent. Possible values:
    ///           - relative
    ///           - perceptual
    ///           - saturation
    ///           - absolute
    ///
    /// Presets define a set of parameters commonly used together. Possible presets:
    /// - srgb:       type=rgb,gamut=srgb,wp=d65,tf=srgb,intent=relative
    /// - display_p3: type=rgb,gamut=p3,wp=d65,tf=srgb,intent=relative
    /// - rec2020:    type=rgb,gamut=bt2100,wp=d65,tf=bt709,intent=relative
    /// - rec2100:    type=rgb,gamut=bt2100,wp=d65,intent=relative
    ///               Transfer function is not set for this preset; one should be provided, e.g. rec2100,tf=pq
    #[arg(long, value_parser = parse_color_encoding, verbatim_doc_comment)]
    target_colorspace: Option<EnumColourEncoding>,
    /// (unstable) Path to target ICC profile
    #[arg(long)]
    target_icc: Option<PathBuf>,
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

#[derive(Debug, Clone, Default)]
struct ColorspaceSpec {
    ty: Option<ColourSpace>,
    white_point: Option<WhitePoint>,
    gamut: Option<Primaries>,
    tf: Option<TransferFunction>,
    intent: Option<RenderingIntent>,
}

#[derive(Debug)]
struct ColorspaceSpecParseError(std::borrow::Cow<'static, str>);

impl From<&'static str> for ColorspaceSpecParseError {
    fn from(value: &'static str) -> Self {
        Self(value.into())
    }
}

impl From<String> for ColorspaceSpecParseError {
    fn from(value: String) -> Self {
        Self(value.into())
    }
}

impl std::fmt::Display for ColorspaceSpecParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for ColorspaceSpecParseError {}

impl ColorspaceSpec {
    fn new() -> Self {
        Self::default()
    }

    fn from_preset(preset: &str) -> Result<Self, ColorspaceSpecParseError> {
        let preset_lowercase = preset.to_lowercase();
        Ok(match &*preset_lowercase {
            "srgb" => ColorspaceSpec {
                ty: Some(ColourSpace::Rgb),
                white_point: Some(WhitePoint::D65),
                gamut: Some(Primaries::Srgb),
                tf: Some(TransferFunction::Srgb),
                intent: Some(RenderingIntent::Relative),
            },
            "display_p3" => ColorspaceSpec {
                ty: Some(ColourSpace::Rgb),
                white_point: Some(WhitePoint::D65),
                gamut: Some(Primaries::P3),
                tf: Some(TransferFunction::Srgb),
                intent: Some(RenderingIntent::Relative),
            },
            "rec2020" | "rec.2020" => ColorspaceSpec {
                ty: Some(ColourSpace::Rgb),
                white_point: Some(WhitePoint::D65),
                gamut: Some(Primaries::Bt2100),
                tf: Some(TransferFunction::Bt709),
                intent: Some(RenderingIntent::Relative),
            },
            "rec2100" | "rec.2100" => ColorspaceSpec {
                ty: Some(ColourSpace::Rgb),
                white_point: Some(WhitePoint::D65),
                gamut: Some(Primaries::Bt2100),
                tf: None,
                intent: Some(RenderingIntent::Relative),
            },
            _ => {
                return Err(format!("unknown preset `{preset}`").into());
            }
        })
    }

    fn add_param(&mut self, param: &str) -> Result<(), ColorspaceSpecParseError> {
        let (name, value) = param
            .split_once('=')
            .ok_or_else(|| format!("`{param}` is not a parameter spec"))?;
        let name_lowercase = name.to_ascii_lowercase();
        let value_lowercase = value.to_ascii_lowercase();
        match &*name_lowercase {
            "type" | "color_space" => match &*value_lowercase {
                "rgb" => self.ty = Some(ColourSpace::Rgb),
                "xyb" => {
                    let mut invalid_option = None;
                    if self.white_point.is_some() {
                        invalid_option = Some("white point");
                    } else if self.gamut.is_some() {
                        invalid_option = Some("color gamut");
                    } else if self.tf.is_some() {
                        invalid_option = Some("transfer function");
                    }
                    if let Some(invalid_option) = invalid_option {
                        return Err(format!(
                            "cannot set {invalid_option} when color space type is XYB"
                        )
                        .into());
                    }

                    self.ty = Some(ColourSpace::Xyb);
                }
                "gray" | "grey" | "grayscale" | "greyscale" => {
                    if self.gamut.is_some() {
                        return Err(
                            "cannot set color gamut when color space type is Grayscale".into()
                        );
                    }

                    self.ty = Some(ColourSpace::Grey);
                }
                _ => return Err(format!("unknown color space type `{value}`").into()),
            },
            "white_point" | "wp" => {
                if let Some(ColourSpace::Xyb) = self.ty {
                    return Err("cannot set white point if color space type is XYB".into());
                }

                let wp = match &*value_lowercase {
                    "d65" => WhitePoint::D65,
                    "dci" => WhitePoint::Dci,
                    "e" => WhitePoint::E,
                    "d50" => WhitePoint::Custom(Customxy {
                        x: 345669,
                        y: 358496,
                    }),
                    _ => return Err(format!("invalid white point `{value}`").into()),
                };
                self.white_point = Some(wp);
            }
            "gamut" | "primaries" => {
                if let Some(ColourSpace::Xyb) = self.ty {
                    return Err("cannot set white point if color space type is XYB".into());
                }
                if let Some(ColourSpace::Grey) = self.ty {
                    return Err("cannot set white point if color space type is Grayscale".into());
                }

                let gamut = match &*value_lowercase {
                    "srgb" | "bt709" => Primaries::Srgb,
                    "p3" | "dci" => Primaries::P3,
                    "2020" | "bt2020" | "bt.2020" | "rec2020" | "rec.2020" | "2100" | "bt2100"
                    | "bt.2100" | "rec2100" | "rec.2100" => Primaries::Bt2100,
                    _ => return Err(format!("invalid gamut `{value}`").into()),
                };
                self.gamut = Some(gamut);
            }
            "tf" | "transfer_function" | "curve" | "tone_curve" => {
                if let Some(ColourSpace::Xyb) = self.ty {
                    return Err("cannot set transfer function if color space type is XYB".into());
                }

                let tf = match &*value_lowercase {
                    "srgb" => TransferFunction::Srgb,
                    "bt709" | "bt.709" | "709" => TransferFunction::Bt709,
                    "dci" => TransferFunction::Dci,
                    "pq" | "perceptual_quantizer" => TransferFunction::Pq,
                    "hlg" | "hybrid_log_gamma" => TransferFunction::Hlg,
                    "linear" => TransferFunction::Linear,
                    gamma => {
                        let gamma = gamma
                            .parse::<f32>()
                            .map_err(|_| format!("invalid transfer function `{value}`"))?;
                        if !gamma.is_finite() || gamma < 1f32 {
                            return Err(format!("gamma of {gamma} is invalid").into());
                        }

                        TransferFunction::Gamma {
                            g: (gamma * 1e7 + 0.5) as u32,
                            inverted: false,
                        }
                    }
                };
                self.tf = Some(tf);
            }
            "intent" | "rendering_intent" => {
                let intent = match &*value_lowercase {
                    "relative" | "rel" | "relative_colorimetric" => RenderingIntent::Relative,
                    "perceptual" | "per" => RenderingIntent::Perceptual,
                    "saturation" | "sat" => RenderingIntent::Saturation,
                    "absolute" | "abs" | "absolute_colorimetric" => RenderingIntent::Absolute,
                    _ => return Err(format!("invalid rendering intent `{value}`").into()),
                };
                self.intent = Some(intent);
            }
            _ => return Err(format!("invalid parameter `{name}`").into()),
        }

        Ok(())
    }
}

fn parse_color_encoding(val: &str) -> Result<EnumColourEncoding, ColorspaceSpecParseError> {
    let mut params = val.split(',');

    let first = params.next().ok_or("parameters are required")?;
    let mut spec = ColorspaceSpec::from_preset(first).or_else(|preset_err| {
        let mut spec = ColorspaceSpec::new();
        spec.add_param(first).map(|_| spec).map_err(|param_err| {
            if first.contains('=') {
                param_err
            } else {
                preset_err
            }
        })
    })?;

    for param in params {
        spec.add_param(param)?;
    }

    let ty = if let Some(ty) = spec.ty {
        ty
    } else if spec.white_point.is_some() && spec.gamut.is_some() && spec.tf.is_some() {
        ColourSpace::Rgb
    } else {
        return Err("color space type is required".into());
    };

    Ok(match ty {
        ColourSpace::Rgb => {
            let white_point = spec.white_point.ok_or("white point is required")?;
            let primaries = spec.gamut.ok_or("color gamut is required")?;
            let tf = spec.tf.ok_or("transfer function is required")?;
            let rendering_intent = spec.intent.ok_or("rendering intent is required")?;
            EnumColourEncoding {
                colour_space: ColourSpace::Rgb,
                white_point,
                primaries,
                tf,
                rendering_intent,
            }
        }
        ColourSpace::Grey => {
            let white_point = spec.white_point.ok_or("white point is required")?;
            let tf = spec.tf.ok_or("transfer function is required")?;
            let rendering_intent = spec.intent.ok_or("rendering intent is required")?;
            EnumColourEncoding {
                colour_space: ColourSpace::Grey,
                white_point,
                primaries: Primaries::Srgb,
                tf,
                rendering_intent,
            }
        }
        ColourSpace::Xyb => {
            let rendering_intent = spec.intent.ok_or("rendering intent is required")?;
            EnumColourEncoding {
                colour_space: ColourSpace::Xyb,
                white_point: WhitePoint::D65,
                primaries: Primaries::Srgb,
                tf: TransferFunction::Srgb,
                rendering_intent,
            }
        }
        _ => unreachable!(),
    })
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

    let output_png = args.output.is_some()
        && matches!(
            args.output_format,
            OutputFormat::Png | OutputFormat::Png8 | OutputFormat::Png16
        );
    if let Some(icc_path) = &args.target_icc {
        tracing::debug!("Setting target ICC profile");
        let icc_profile = std::fs::read(icc_path).expect("Failed to read ICC profile");
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

    let image_size = &image.image_header().size;
    let image_meta = &image.image_header().metadata;
    tracing::info!("Image dimension: {}x{}", image.width(), image.height());
    tracing::debug!(colour_encoding = format_args!("{:?}", image_meta.colour_encoding));

    if let Some(icc_path) = &args.icc_output {
        if let Some(icc) = image.original_icc() {
            tracing::debug!("Writing ICC profile");
            std::fs::write(icc_path, icc).expect("Failed to write ICC profile");
        } else {
            tracing::info!("No embedded ICC profile, skipping icc_output");
        }
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
                    Some(png::BitDepth::Sixteen),
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
    let cicp = image.rendered_cicp();
    let metadata = &image.image_header().metadata;

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

    encoder.validate_sequence(true);

    let mut writer = encoder.write_header().expect("failed to write header");

    tracing::debug!("Embedding ICC profile");
    let compressed_icc = miniz_oxide::deflate::compress_to_vec_zlib(&source_icc, 7);
    let mut iccp_chunk_data = vec![b'0', 0, 0];
    iccp_chunk_data.extend(compressed_icc);
    writer
        .write_chunk(png::chunk::iCCP, &iccp_chunk_data)
        .expect("failed to write iCCP");

    if let Some(cicp) = cicp {
        tracing::debug!(cicp = format_args!("{:?}", cicp), "Writing cICP chunk");
        writer
            .write_chunk(png::chunk::ChunkType([b'c', b'I', b'C', b'P']), &cicp)
            .expect("failed to write cICP");
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
