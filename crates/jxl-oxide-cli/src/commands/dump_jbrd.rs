use std::path::PathBuf;

use clap::Parser;

#[derive(Debug, Parser)]
#[non_exhaustive]
pub struct DumpJbrd {
    /// Input file
    pub input: PathBuf,
}
