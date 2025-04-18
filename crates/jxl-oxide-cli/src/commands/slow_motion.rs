use std::path::PathBuf;

use clap::{Parser, value_parser};

#[derive(Debug, Parser)]
#[non_exhaustive]
pub struct SlowMotionArgs {
    /// Input file
    pub input: PathBuf,
    /// Output path
    #[arg(short, long)]
    pub output: PathBuf,
    /// Bytes per frame
    #[arg(short = 'b', long, value_parser = value_parser!(u32).range(1..=1024), default_value_t = 1)]
    pub bytes_per_frame: u32,
    /// Number of parallelism to use
    #[cfg_attr(feature = "rayon", arg(short = 'j', long))]
    #[cfg_attr(not(feature = "rayon"), arg(skip))]
    pub num_threads: Option<usize>,
    /// Font to use when displaying frame info
    #[arg(long, default_value_t = String::from("monospace"))]
    pub font: String,
}
