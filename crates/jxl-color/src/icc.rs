//! Functions related to ICC profiles.
//!
//! - [`read_icc`] and [`decode_icc`] can be used to read embedded ICC profile from the bitstream.
//! - [`colour_encoding_to_icc`] can be used to create an ICC profile to embed into the decoded
//!   image file, or to be used by the color management system for various purposes.

mod decode;
mod parse;
mod synthesize;

pub use decode::{decode_icc, read_icc};
pub use parse::icc_tf;
pub(crate) use parse::parse_icc;
pub(crate) use parse::parse_icc_raw;
pub use synthesize::colour_encoding_to_icc;

#[derive(Debug)]
#[non_exhaustive]
pub struct IccHeader {
    pub color_space: [u8; 4],
    pub rendering_intent: crate::RenderingIntent,
}

#[derive(Debug)]
struct IccTag {
    tag: [u8; 4],
    data_offset: u32,
    len: u32,
}
