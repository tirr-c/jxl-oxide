use std::path::PathBuf;

use clap::Parser;

/// Prints information about JPEG XL image.
#[derive(Debug, Parser)]
#[non_exhaustive]
pub struct InfoArgs {
    /// Input file
    pub input: PathBuf,
    /// Output information of all frames in addition to keyframes
    #[arg(long)]
    pub all_frames: bool,
    /// Output group sizes and offsets
    #[arg(long)]
    pub with_offset: bool,
}
