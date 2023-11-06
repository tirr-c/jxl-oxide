//! This crate provides [`SimpleGrid`], [`CutGrid`] and [`PaddedGrid`], used in various
//! places involving images.
mod simd;
mod simple_grid;
mod subgrid;
pub use simd::SimdVector;
pub use simple_grid::*;
pub use subgrid::*;
