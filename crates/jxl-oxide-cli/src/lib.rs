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
#[cfg(feature = "__ffmpeg")]
pub mod slow_motion;

mod output;

pub use commands::{Args, Subcommands};
pub use error::Error;

type Result<T> = std::result::Result<T, Error>;

#[cfg(feature = "rayon")]
fn create_thread_pool(num_threads: Option<usize>) -> jxl_oxide::JxlThreadPool {
    if num_threads.is_some() {
        jxl_oxide::JxlThreadPool::rayon(num_threads)
    } else {
        tracing::debug!("Using default number of threads");
        jxl_oxide::JxlThreadPool::rayon_global()
    }
}

#[cfg(not(feature = "rayon"))]
fn create_thread_pool(_num_threads: Option<usize>) -> jxl_oxide::JxlThreadPool {
    jxl_oxide::JxlThreadPool::none()
}
