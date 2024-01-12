pub mod commands;
pub mod decode;
pub mod error;
#[cfg(feature = "__devtools")]
pub mod generate_fixture;
pub mod info;

pub use commands::{Args, Subcommands};
pub use error::Error;

type Result<T> = std::result::Result<T, Error>;
