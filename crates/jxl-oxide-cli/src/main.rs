use clap::Parser;
use jxl_oxide_cli::{Args, Subcommands};

fn main() {
    let Args {
        subcommand,
        globals,
    } = Args::parse();

    let filter = if globals.verbose {
        tracing::level_filters::LevelFilter::DEBUG
    } else {
        tracing::level_filters::LevelFilter::INFO
    };
    let env_filter = tracing_subscriber::EnvFilter::builder()
        .with_default_directive(filter.into())
        .from_env_lossy();
    tracing_subscriber::fmt()
        .with_span_events(tracing_subscriber::fmt::format::FmtSpan::ACTIVE)
        .with_env_filter(env_filter)
        .init();

    match subcommand {
        Subcommands::Decode(args) => {
            jxl_oxide_cli::decode::handle_decode(args);
        }
        Subcommands::Info(args) => {
            jxl_oxide_cli::info::handle_info(args);
        }
        #[cfg(feature = "__devtools")]
        Subcommands::GenerateFixture(args) => {
            jxl_oxide_cli::generate_fixture::handle_generate_fixture(args);
        }
    }
}
