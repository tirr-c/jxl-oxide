//! Various integrations to other library crates.
//!
//! # Available integrations
//!
//! Integrations are enabled with feature flags.
//! - `JxlDecoder`, which implements `image::ImageDecoder` (`image` feature)

#[cfg(feature = "image")]
mod image;

#[cfg(feature = "image")]
pub use image::*;
