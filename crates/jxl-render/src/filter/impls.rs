#[cfg(not(target_arch = "x86_64"))]
mod generic;
#[cfg(target_arch = "x86_64")]
mod x86_64;

#[cfg(not(target_arch = "x86_64"))]
pub use generic::*;

#[cfg(target_arch = "x86_64")]
pub use x86_64::*;
