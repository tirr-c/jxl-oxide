mod generic;

/*
#[cfg(
    not(
        any(target_arch = "x86", target_arch = "x86_64"),
    )
)]
*/
use generic as inner;

/*
#[cfg(
    any(target_arch = "x86", target_arch = "x86_64"),
)]
#[path = "x86_64.rs"]
mod inner;
*/

pub use inner::*;
