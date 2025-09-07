//! Various integrations to other library crates.
//!
//! # Available integrations
//!
//! Integrations are enabled with feature flags.
//! - `JxlDecoder`, which implements `image::ImageDecoder` (`image` feature)
//! - `register_image_decoding_hook`, which registers `.jxl` format to `image` crate (`image` feature)

#[cfg(feature = "image")]
mod image;

#[cfg(feature = "image")]
pub use image::*;
