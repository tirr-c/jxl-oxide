pub mod color_encoding;
pub mod decode;
#[cfg(feature = "__devtools")]
pub mod dump_jbrd;
#[cfg(feature = "__devtools")]
pub mod generate_fixture;
pub mod info;
#[cfg(feature = "__devtools")]
pub mod progressive;
#[cfg(feature = "__ffmpeg")]
pub mod slow_motion;
#[cfg(test)]
pub mod tests;

pub use color_encoding::parse_color_encoding;
pub use decode::DecodeArgs;
#[cfg(feature = "__devtools")]
pub use dump_jbrd::DumpJbrd;
#[cfg(feature = "__devtools")]
pub use generate_fixture::GenerateFixtureArgs;
pub use info::InfoArgs;
#[cfg(feature = "__devtools")]
pub use progressive::ProgressiveArgs;
#[cfg(feature = "__ffmpeg")]
pub use slow_motion::SlowMotionArgs;

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
    /// Print debug information; can be repeated.
    #[arg(short, long, global = true, action = clap::ArgAction::Count)]
    pub verbose: u8,
    /// Do not print logs to console.
    #[arg(short, long, global = true, conflicts_with = "verbose")]
    pub quiet: bool,
}

#[derive(Debug, clap::Subcommand)]
pub enum Subcommands {
    /// Decode JPEG XL image (assumed if no subcommand is specified).
    #[command(short_flag = 'd')]
    Decode(DecodeArgs),
    /// Print information about JPEG XL image.
    #[command(short_flag = 'I')]
    Info(InfoArgs),
    /// (devtools) Generate frames for progressive decoding animation.
    #[cfg(feature = "__devtools")]
    Progressive(ProgressiveArgs),
    /// (devtools) Generate fixture to use for testing.
    #[cfg(feature = "__devtools")]
    GenerateFixture(GenerateFixtureArgs),
    /// (devtools) Dump JPEG bitstream reconstruction data.
    #[cfg(feature = "__devtools")]
    DumpJbrd(DumpJbrd),
    /// (devtools, ffmpeg) Load an image byte-by-byte.
    #[cfg(feature = "__ffmpeg")]
    SlowMotion(SlowMotionArgs),
}
