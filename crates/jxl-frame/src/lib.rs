pub mod data;
mod error;
pub mod encoding;
pub mod filter;
pub mod frame;
pub mod header;
mod noise;
mod patch;
mod spline;
mod toc;

pub use error::{Error, Result};
pub use header::FrameHeader;
pub use toc::Toc;
