//! This crate provides a JPEG XL bitstream reader and helper macros. The bitstream reader supports both
//! bare codestream and container format, and it can detect which format to read.

mod bitstream;
pub mod container;
mod error;

pub use bitstream::{Bitstream, U32Specifier, U};
pub use container::{BitstreamKind, ContainerDetectingReader};
pub use error::{Error, Result};

/// Perform `UnpackSigned` for `u32`, as specified in the JPEG XL specification.
#[inline]
pub fn unpack_signed(x: u32) -> i32 {
    let bit = x & 1;
    let base = x >> 1;
    let flip = 0u32.wrapping_sub(bit);
    (base ^ flip) as i32
}

/// Perform `UnpackSigned` for `u64`, as specified in the JPEG XL specification.
#[inline]
pub fn unpack_signed_u64(x: u64) -> i64 {
    let bit = x & 1;
    let base = x >> 1;
    let flip = 0u64.wrapping_sub(bit);
    (base ^ flip) as i64
}
