//! This crate provides a set of functions related to color encodings defined by the JPEG XL
//! specification. Especially, these functions can perform conversion from the XYB colorspace to
//! all of the "enum colorspaces" which can be signalled in the JPEG XL image header.
//!
//! # Functions
//! - [`xyb_to_linear_srgb`] performs inverse XYB transform, which converts samples in the XYB
//!   colorspace to the linear sRGB signals.
//! - [`ycbcr_to_rgb`] converts YCbCr samples to RGB samples.
//! - [`from_linear_srgb`] converts linear sRGB samples, mainly produced by the previous call to
//!   [`xyb_to_linear_srgb`], to the given enum colorspace.
//!
//! # Modules
//! - [`consts`] defines constants used by the various colorspaces.
//! - [`icc`] provides functions related to ICC profiles.

mod ciexyz;
pub mod consts;
mod convert;
mod error;
mod fastmath;
pub mod header;
pub mod icc;
mod tf;
mod xyb;
mod ycbcr;

pub use error::*;
pub use convert::from_linear_srgb;
pub use header::*;
pub use xyb::xyb_to_linear_srgb;
pub use ycbcr::ycbcr_to_rgb;
