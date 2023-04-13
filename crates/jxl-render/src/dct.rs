mod consts;
mod generic;

#[cfg(
    not(target_arch = "x86_64")
)]
pub use generic::*;

#[cfg(target_arch = "x86_64")]
mod x86_64;
#[cfg(target_arch = "x86_64")]
pub use x86_64::*;
