use std::path::PathBuf;

use clap::Parser;

#[derive(Debug, Parser)]
#[non_exhaustive]
pub struct GenerateFixtureArgs {
    /// Input file
    pub input: PathBuf,
    /// Output file
    pub output: PathBuf,
}
