mod consts;
mod generic;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum DctDirection {
    Forward,
    Inverse,
}

#[cfg(
    not(target_arch = "x86_64")
)]
pub use generic::*;

#[cfg(target_arch = "x86_64")]
mod x86_64;
#[cfg(target_arch = "x86_64")]
pub use x86_64::*;
