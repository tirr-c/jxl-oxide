#![allow(unsafe_op_in_unsafe_fn)]

#[cfg(target_arch = "aarch64")]
mod aarch64;
pub(super) mod generic;
#[cfg(all(target_family = "wasm", target_feature = "simd128"))]
mod wasm32;
#[cfg(target_arch = "x86_64")]
mod x86_64;

#[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_family = "wasm", target_feature = "simd128")
)))]
pub use generic::*;

#[cfg(target_arch = "x86_64")]
pub use x86_64::*;

#[cfg(target_arch = "aarch64")]
pub use aarch64::*;

#[cfg(all(target_family = "wasm", target_feature = "simd128"))]
pub use wasm32::*;
