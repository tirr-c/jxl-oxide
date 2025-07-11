use std::path::PathBuf;

use clap::Parser;
use jxl_oxide::{CropInfo, EnumColourEncoding};

#[derive(Debug, Parser)]
#[non_exhaustive]
pub struct DecodeArgs {
    /// Output file
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    /// Output ICC file
    #[arg(long)]
    pub icc_output: Option<PathBuf>,
    /// Input file
    pub input: PathBuf,
    /// (unstable) Region to render, in format of 'width height left top'
    #[arg(long, value_parser = parse_crop_info)]
    pub crop: Option<CropInfo>,
    /// (unstable) Approximate memory limit, in bytes
    #[arg(long, default_value_t = 0)]
    pub approx_memory_limit: usize,
    /// Format to output
    #[arg(value_enum, short = 'f', long)]
    pub output_format: Option<OutputFormat>,
    #[allow(clippy::doc_overindented_list_items, reason = "verbatim doc comment")]
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
    #[arg(long, value_parser = super::parse_color_encoding, verbatim_doc_comment)]
    pub target_colorspace: Option<EnumColourEncoding>,
    /// (unstable) Path to target ICC profile
    #[arg(long)]
    pub target_icc: Option<PathBuf>,
    /// Number of parallelism to use
    #[cfg_attr(feature = "rayon", arg(short = 'j', long))]
    #[cfg_attr(not(feature = "rayon"), arg(skip))]
    pub num_threads: Option<usize>,
    /// Number of repeated decoding, used for benchmarking
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..))]
    pub num_reps: Option<u32>,
    /// External color management system to use for ICC profiles.
    ///
    /// External CMS will handle ICC profiles jxl-oxide cannot handle.
    #[arg(long)]
    pub cms: Option<Cms>,
    /// (unstable) Force 32-bit Modular buffers when decoding.
    #[arg(long)]
    pub force_wide_buffers: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum OutputFormat {
    /// PNG, respects bit depth information.
    Png,
    /// PNG, always 8-bit.
    Png8,
    /// PNG, always 16-bit.
    Png16,
    /// JPEG bitstream reconstruction.
    #[value(name = "jpeg", alias("jpg"))]
    JpegReconstruct,
    /// Numpy, used for conformance test.
    Npy,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Cms {
    /// Little CMS 2.
    #[default]
    Lcms2,
    /// moxcms crate.
    Moxcms,
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
