use std::io::IsTerminal;

use clap::Parser;
use crossterm::ExecutableCommand;
use jxl_oxide_cli::commands::TermColorKind;
use jxl_oxide_cli::{Args, Subcommands};

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() -> std::process::ExitCode {
    let Args {
        subcommand,
        globals,
        decode,
    } = Args::parse();

    // Make sure the terminal handle ANSI escape sequences.
    let mut stderr = std::io::stderr();
    let is_tty = stderr.is_terminal();
    let use_colors = match globals.color {
        TermColorKind::Auto => is_tty,
        TermColorKind::Never => false,
        TermColorKind::Always => true,
    };
    if use_colors {
        stderr.execute(crossterm::style::ResetColor).ok();
    }

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
            .with_writer(std::io::stderr)
            .with_ansi(use_colors)
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
        #[cfg(feature = "__devtools")]
        Some(Subcommands::Progressive(args)) => {
            jxl_oxide_cli::progressive::handle_progressive(args)
        }
        #[cfg(feature = "__devtools")]
        Some(Subcommands::DumpJbrd(args)) => jxl_oxide_cli::dump_jbrd::handle_dump_jbrd(args),
        #[cfg(feature = "__ffmpeg")]
        Some(Subcommands::SlowMotion(args)) => jxl_oxide_cli::slow_motion::handle_slow_motion(args),
    };

    if let Err(e) = result {
        tracing::error!("{e}");
        std::process::ExitCode::FAILURE
    } else {
        std::process::ExitCode::SUCCESS
    }
}
