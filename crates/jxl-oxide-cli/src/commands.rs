use clap::{Parser, Subcommand};

pub mod color_encoding;
pub mod decode;
#[cfg(feature = "__devtools")]
pub mod generate_fixture;
pub mod info;

pub use color_encoding::parse_color_encoding;
pub use decode::DecodeArgs;
#[cfg(feature = "__devtools")]
pub use generate_fixture::GenerateFixtureArgs;
pub use info::InfoArgs;

#[derive(Debug, Parser)]
#[command(version)]
pub struct Args {
    #[command(subcommand)]
    pub subcommand: Subcommands,
    #[command(flatten)]
    pub globals: GlobalArgs,
}

#[derive(Debug, Parser)]
#[non_exhaustive]
pub struct GlobalArgs {
    /// Print debug information
    #[arg(short, long)]
    pub verbose: bool,
}

#[derive(Debug, Subcommand)]
pub enum Subcommands {
    /// Decodes JPEG XL image.
    Decode(DecodeArgs),
    /// Prints information about JPEG XL image.
    Info(InfoArgs),
    /// (devtools) Generate fixture to use for testing.
    #[cfg(feature = "__devtools")]
    GenerateFixture(GenerateFixtureArgs),
}
