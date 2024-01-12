use clap::Parser;
use jxl_oxide_cli::{Args, Subcommands};

fn main() -> std::process::ExitCode {
    let Args {
        subcommand,
        globals,
        decode,
    } = Args::parse();

    if !globals.quiet {
        let filter = match globals.verbose {
            0 => tracing::level_filters::LevelFilter::INFO,
            1 => tracing::level_filters::LevelFilter::DEBUG,
            2.. => tracing::level_filters::LevelFilter::TRACE,
        };
        let env_filter = tracing_subscriber::EnvFilter::builder()
            .with_default_directive(filter.into())
            .from_env_lossy();
        tracing_subscriber::fmt()
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::ACTIVE)
            .with_env_filter(env_filter)
            .init();
    }

    let result = match subcommand {
        Some(Subcommands::Decode(args)) => jxl_oxide_cli::decode::handle_decode(args),
        None => jxl_oxide_cli::decode::handle_decode(decode.unwrap()),
        Some(Subcommands::Info(args)) => jxl_oxide_cli::info::handle_info(args),
        #[cfg(feature = "__devtools")]
        Some(Subcommands::GenerateFixture(args)) => {
            jxl_oxide_cli::generate_fixture::handle_generate_fixture(args);
            Ok(())
        }
    };

    if let Err(e) = result {
        tracing::error!("{e}");
        std::process::ExitCode::FAILURE
    } else {
        std::process::ExitCode::SUCCESS
    }
}
