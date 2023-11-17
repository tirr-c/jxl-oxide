//! This crate provides a JPEG XL bitstream reader and helper macros. The bitstream reader supports both
//! bare codestream and container format, and it can detect which format to read.

mod container;
mod error;
mod macros;
mod memory;
mod reader;

pub use container::*;
pub use error::{Error, Result};
pub use macros::{unpack_signed, unpack_signed_u64};
pub use memory::Bitstream;
pub use reader::ContainerDetectingReader;

pub trait Bundle<Ctx = ()>: Sized {
    type Error;

    /// Parses a value from the bitstream with the given context.
    fn parse(bitstream: &mut Bitstream<'_>, ctx: Ctx) -> std::result::Result<Self, Self::Error>;
}

pub trait BundleDefault<Ctx = ()>: Sized {
    /// Creates a default value with the given context.
    fn default_with_context(ctx: Ctx) -> Self;
}

impl<T, Ctx> BundleDefault<Ctx> for T
where
    T: Default + Sized,
{
    fn default_with_context(_: Ctx) -> Self {
        Default::default()
    }
}

impl<T, Ctx> Bundle<Ctx> for Option<T>
where
    T: Bundle<Ctx>,
{
    type Error = T::Error;

    fn parse(bitstream: &mut Bitstream, ctx: Ctx) -> std::result::Result<Self, Self::Error> {
        T::parse(bitstream, ctx).map(Some)
    }
}

/// Name type which is read by some JPEG XL headers.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Name(String);

impl<Ctx> Bundle<Ctx> for Name {
    type Error = Error;

    fn parse(bitstream: &mut Bitstream, _: Ctx) -> Result<Self> {
        let len = read_bits!(bitstream, U32(0, u(4), 16 + u(5), 48 + u(10)))? as usize;
        let mut data = vec![0u8; len];
        for b in &mut data {
            *b = bitstream.read_bits(8)? as u8;
        }
        let name = String::from_utf8(data).map_err(|_| Error::NonUtf8Name)?;
        Ok(Self(name))
    }
}

impl std::ops::Deref for Name {
    type Target = String;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for Name {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
