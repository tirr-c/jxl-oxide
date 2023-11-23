//! This crate provides [`SimpleGrid`], [`CutGrid`] and [`PaddedGrid`], used in various
//! places involving images.
mod alloc_tracker;
mod simd;
mod simple_grid;
mod subgrid;
pub use alloc_tracker::*;
pub use simd::SimdVector;
pub use simple_grid::*;
pub use subgrid::*;

#[derive(Debug)]
pub enum Error {
    OutOfMemory(usize),
}

impl std::error::Error for Error {}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OutOfMemory(bytes) => write!(f, "failed to allocate {bytes} byte(s)"),
        }
    }
}
