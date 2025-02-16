use std::path::PathBuf;

use clap::Parser;

#[derive(Debug, Parser)]
#[non_exhaustive]
pub struct ProgressiveArgs {
    /// Input file
    pub input: PathBuf,
    /// Unit size of bytes to feed per frame
    ///
    /// Bytes per frame increases to 2x at 10% of the image, 4x at 25% and 8x at 50%.
    /// Default value is computed so that the first 10% of the image is loaded in about 5 seconds.
    #[arg(short = 's', long, default_value_t = 0)]
    pub unit_step: u64,
    /// Output directory path
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    /// Number of parallelism to use
    #[cfg_attr(feature = "rayon", arg(short = 'j', long))]
    #[cfg_attr(not(feature = "rayon"), arg(skip))]
    pub num_threads: Option<usize>,
    /// Font to use when displaying frame info
    #[cfg(feature = "__ffmpeg")]
    #[arg(long, default_value_t = String::from("monospace"))]
    pub font: String,
}
