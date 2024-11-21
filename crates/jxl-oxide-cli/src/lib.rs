pub mod commands;
pub mod decode;
#[cfg(feature = "__devtools")]
pub mod dump_jbrd;
pub mod error;
#[cfg(feature = "__devtools")]
pub mod generate_fixture;
pub mod info;
#[cfg(feature = "__devtools")]
pub mod progressive;

mod output;

pub use commands::{Args, Subcommands};
pub use error::Error;

type Result<T> = std::result::Result<T, Error>;
