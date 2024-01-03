pub mod color_encoding;
pub mod decode;
#[cfg(feature = "__devtools")]
pub mod generate_fixture;
pub mod info;
#[cfg(test)]
pub mod tests;

pub use color_encoding::parse_color_encoding;
pub use decode::DecodeArgs;
#[cfg(feature = "__devtools")]
pub use generate_fixture::GenerateFixtureArgs;
pub use info::InfoArgs;

#[derive(Debug, clap::Parser)]
#[command(version)]
#[command(args_conflicts_with_subcommands = true)]
pub struct Args {
    #[command(subcommand)]
    pub subcommand: Option<Subcommands>,
    #[command(flatten)]
    pub decode: Option<DecodeArgs>,
    #[command(flatten)]
    pub globals: GlobalArgs,
}

#[derive(Debug, clap::Args)]
#[non_exhaustive]
pub struct GlobalArgs {
    /// Print debug information
    #[arg(short, long, global(true))]
    pub verbose: bool,
}

#[derive(Debug, clap::Subcommand)]
pub enum Subcommands {
    /// Decode JPEG XL image (assumed if no subcommand is specified).
    #[command(short_flag = 'd')]
    Decode(DecodeArgs),
    /// Print information about JPEG XL image.
    #[command(short_flag = 'I')]
    Info(InfoArgs),
    /// (devtools) Generate fixture to use for testing.
    #[cfg(feature = "__devtools")]
    GenerateFixture(GenerateFixtureArgs),
}
