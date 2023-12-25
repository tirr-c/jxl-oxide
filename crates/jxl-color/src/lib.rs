//! This crate provides a set of functions related to color encodings defined by the JPEG XL
//! specification. Especially, these functions can perform conversion from the XYB colorspace to
//! all of the "enum colorspaces" which can be signalled in the JPEG XL image header.
//!
//! # Modules
//! - [`consts`] defines constants used by the various colorspaces.
//! - [`icc`] provides functions related to ICC profiles.

mod ciexyz;
mod cms;
pub mod consts;
mod convert;
mod error;
mod fastmath;
mod gamut;
pub mod header;
pub mod icc;
mod tf;
mod xyb;
mod ycbcr;

pub use cms::*;
pub use convert::*;
pub use error::*;
pub use header::*;
pub use ycbcr::ycbcr_to_rgb;
