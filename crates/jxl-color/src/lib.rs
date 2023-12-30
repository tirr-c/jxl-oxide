//! This crate provides a set of functions related to color encodings defined by the JPEG XL
//! specification.
//!
//! # Color transformation
//! jxl-color can handle the transformation between "well-known" color encodings without any
//! external color management system, since it is required by the specification. Such
//! transformations can be done by creating a [`ColorTransform`].
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
