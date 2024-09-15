use std::path::PathBuf;

use clap::Parser;

#[derive(Debug, Parser)]
#[non_exhaustive]
pub struct ProgressiveArgs {
    /// Input file
    pub input: PathBuf,
    /// Bytes to feed per frame, in bytes
    #[arg(short, long)]
    pub step: u64,
    /// Output directory path
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    /// Number of parallelism to use
    #[cfg(feature = "rayon")]
    #[arg(short = 'j', long)]
    pub num_threads: Option<usize>,
    /// Font to use when displaying frame info
    #[cfg(feature = "__ffmpeg")]
    #[arg(long, default_value_t = String::from("monospace"))]
    pub font: String,
}
