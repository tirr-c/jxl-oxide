//! This crate provides a JPEG XL bitstream reader and helper macros. The bitstream reader supports both
//! bare codestream and container format, and it can detect which format to read.
//!
//! Consumers of this crate can use [`Bitstream::new_detect`] to create a bitstream reader.

use std::io::prelude::*;

mod container;
mod error;
mod macros;
mod reader;

pub use container::*;
pub use error::{Error, Result};
pub use macros::{unpack_signed, unpack_signed_u64};
pub use reader::ContainerDetectingReader;

pub trait Bundle<Ctx = ()>: Sized {
    type Error;

    /// Parses a value from the bitstream with the given context.
    fn parse<R: Read>(bitstream: &mut Bitstream<R>, ctx: Ctx) -> std::result::Result<Self, Self::Error>;
}

pub trait BundleDefault<Ctx = ()>: Sized {
    /// Creates a default value with the given context.
    fn default_with_context(ctx: Ctx) -> Self;
}

impl<T, Ctx> BundleDefault<Ctx> for T where T: Default + Sized {
    fn default_with_context(_: Ctx) -> Self {
        Default::default()
    }
}

impl<T, Ctx> Bundle<Ctx> for Option<T> where T: Bundle<Ctx> {
    type Error = T::Error;

    fn parse<R: Read>(bitstream: &mut Bitstream<R>, ctx: Ctx) -> std::result::Result<Self, Self::Error> {
        T::parse(bitstream, ctx).map(Some)
    }
}

/// JPEG XL bitstream reader.
pub struct Bitstream<R> {
    global_pos: u64,
    buf: Vec<u8>,
    buf_valid_len: usize,
    buf_offset: usize,
    current: u64,
    bits_left: u32,
    reader: R,
}

#[derive(Debug, Copy, Clone)]
pub struct Bookmark(u64);

impl std::ops::Add<u64> for Bookmark {
    type Output = Bookmark;

    fn add(self, rhs: u64) -> Self::Output {
        Bookmark(self.0 + rhs)
    }
}

impl std::ops::AddAssign<u64> for Bookmark {
    fn add_assign(&mut self, rhs: u64) {
        self.0 += rhs;
    }
}

impl<R> std::fmt::Debug for Bitstream<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f
            .debug_struct("Bitstream")
            .field("global_pos", &self.global_pos)
            .field("buf_valid_len", &self.buf_valid_len)
            .field("buf_offset", &self.buf_offset)
            .field("current", &self.current)
            .field("bits_left", &self.bits_left)
            .finish_non_exhaustive()
    }
}

impl<R> Bitstream<ContainerDetectingReader<R>> {
    /// Creates a bitstream reader which detects the container format automatically.
    pub fn new_detect(reader: R) -> Self {
        Self {
            global_pos: 0,
            buf: Vec::new(),
            buf_valid_len: 0,
            buf_offset: 0,
            current: 0,
            bits_left: 0,
            reader: ContainerDetectingReader::new(reader),
        }
    }
}

impl<R> Bitstream<R> {
    /// Creates a bitstream reader which reads bare codestream.
    pub fn new(reader: R) -> Self {
        Self {
            global_pos: 0,
            buf: Vec::new(),
            buf_valid_len: 0,
            buf_offset: 0,
            current: 0,
            bits_left: 0,
            reader,
        }
    }

    pub fn into_inner(self) -> R {
        self.reader
    }

    fn left_in_buffer(&self) -> &[u8] {
        &self.buf[self.buf_offset..self.buf_valid_len]
    }

    pub fn bookmark(&self) -> Bookmark {
        Bookmark(self.global_pos)
    }

    pub fn rewindable(&mut self) -> Bitstream<RewindMarker<'_, R>> {
        let buf = self.buf.clone();
        Bitstream {
            global_pos: self.global_pos,
            buf,
            buf_valid_len: self.buf_valid_len,
            buf_offset: self.buf_offset,
            current: self.current,
            bits_left: self.bits_left,
            reader: RewindMarker {
                read_data: Vec::new(),
                original: self,
            },
        }
    }
}

impl<R: Read> Bitstream<R> {
    const BUF_SIZE: usize = 4096usize;

    fn fill_buf(&mut self) -> Result<()> {
        debug_assert!(self.left_in_buffer().is_empty());
        if self.buf.len() < Self::BUF_SIZE {
            self.buf.resize(Self::BUF_SIZE, 0);
        }

        let count = self.reader.read(&mut self.buf)?;
        if count == 0 {
            let e: std::io::Error = std::io::ErrorKind::UnexpectedEof.into();
            return Err(e.into());
        }

        self.buf_offset = 0;
        self.buf_valid_len = count;
        Ok(())
    }

    fn fill(&mut self) -> Result<()> {
        debug_assert_eq!(self.bits_left, 0);
        let mut buf_left = self.left_in_buffer();
        if buf_left.is_empty() {
            self.fill_buf()?;
            buf_left = self.left_in_buffer();
        }

        let mut buf_current = [0u8; 8];
        let buf_to_read = std::cmp::min(buf_left.len(), 8);
        buf_current[..buf_to_read].copy_from_slice(&buf_left[..buf_to_read]);

        self.buf_offset += buf_to_read;
        self.current = u64::from_le_bytes(buf_current);
        self.bits_left = (buf_to_read * 8) as u32;

        Ok(())
    }

    #[inline]
    fn read_bits_inner(&mut self, n: u32) -> u64 {
        if n == 0 {
            return 0;
        }
        if n == 1 {
            return self.read_single_bit_inner();
        }

        assert!(self.bits_left >= n);
        let mask = (1 << n) - 1;
        let data = self.current & mask;
        self.current >>= n;
        self.bits_left -= n;
        self.global_pos += n as u64;
        data
    }

    #[inline]
    fn read_single_bit_inner(&mut self) -> u64 {
        assert!(self.bits_left > 0);
        let data = self.current & 1;
        self.current >>= 1;
        self.bits_left -= 1;
        self.global_pos += 1;
        data
    }

    pub fn read_bits(&mut self, n: u32) -> Result<u32> {
        if n == 0 {
            return Ok(0);
        }
        if n == 1 {
            let data = self.read_bool()? as u32;
            return Ok(data);
        }

        debug_assert!(n <= 32);

        if self.bits_left >= n {
            Ok(self.read_bits_inner(n) as u32)
        } else {
            let mut bits = self.bits_left;
            let mut ret = self.read_bits_inner(self.bits_left);

            while bits < n {
                self.fill()?;
                let bits_to_read = std::cmp::min(self.bits_left, n - bits);
                let next_bits = self.read_bits_inner(bits_to_read);
                ret |= next_bits << bits;
                bits += bits_to_read;
            }

            Ok(ret as u32)
        }
    }

    /// Read an `U64` as defined in the JPEG XL specification.
    pub fn read_u64(&mut self) -> Result<u64> {
        let selector = self.read_bits(2)?;
        Ok(match selector {
            0 => 0u64,
            1 => self.read_bits(4)? as u64 + 1,
            2 => self.read_bits(8)? as u64 + 17,
            3 => {
                let mut value = self.read_bits(12)? as u64;
                let mut shift = 12u32;
                while self.read_bits(1)? == 1 {
                    if shift == 60 {
                        value |= (self.read_bits(4)? as u64) << shift;
                        break;
                    }
                    value |= (self.read_bits(8)? as u64) << shift;
                    shift += 8;
                }
                value
            },
            _ => unreachable!(),
        })
    }

    /// Reads a `Bool` as defined in the JPEG XL specification.
    pub fn read_bool(&mut self) -> Result<bool> {
        if self.bits_left == 0 {
            self.fill()?;
        }
        Ok(self.read_single_bit_inner() != 0)
    }

    /// Reads an `F16` as defined in the JPEG XL specification, and convert it to `f32`.
    ///
    /// # Errors
    /// Returns `Error::InvalidFloat` if the value is `NaN` or `Infinity`.
    pub fn read_f16_as_f32(&mut self) -> Result<f32> {
        let v = self.read_bits(16)?;
        let neg_bit = (v & 0x8000) << 16;

        if v & 0x7fff == 0 {
            // Zero
            return Ok(f32::from_bits(neg_bit));
        }
        let mantissa = v & 0x3ff; // 10 bits
        let exponent = (v >> 10) & 0x1f; // 5 bits
        if exponent == 0x1f {
            // NaN, Infinity
            Err(Error::InvalidFloat)
        } else if exponent == 0 {
            // Subnormal
            let val = (1.0 / 16384.0) * (mantissa as f32 / 1024.0);
            Ok(if neg_bit != 0 { -val } else { val })
        } else {
            // Normal
            let mantissa = mantissa << 13; // 23 bits
            let exponent = exponent + 112;
            let bitpattern = mantissa | (exponent << 23) | neg_bit;
            Ok(f32::from_bits(bitpattern))
        }
    }

    /// Performs `ZeroPadToByte` as defined in the JPEG XL specification.
    pub fn zero_pad_to_byte(&mut self) -> Result<()> {
        let bits = self.bits_left % 8;
        let data = self.read_bits_inner(bits);
        if data != 0 {
            Err(Error::NonZeroPadding)
        } else {
            Ok(())
        }
    }

    pub fn read_bundle<B: Bundle<()>>(&mut self) -> std::result::Result<B, B::Error> {
        B::parse(self, ())
    }

    pub fn read_bundle_with_ctx<B: Bundle<Ctx>, Ctx>(&mut self, ctx: Ctx) -> std::result::Result<B, B::Error> {
        B::parse(self, ctx)
    }

    pub fn skip_to_bookmark(&mut self, Bookmark(target): Bookmark) -> Result<()> {
        let Some(mut diff) = target.checked_sub(self.global_pos) else {
            return Err(Error::CannotSkip);
        };

        let bits = (self.bits_left as u64 % 8).min(diff) as u32;
        self.read_bits(bits).unwrap();
        diff -= bits as u64;

        let mut buf = vec![0u8; 4096];
        while diff >= 8 {
            let bytes = (diff / 8).min(4096);
            self.read_bytes_aligned(&mut buf[..bytes as usize])?;
            diff -= bytes * 8;
        }

        self.read_bits(diff as u32)?;
        assert_eq!(self.global_pos, target);
        Ok(())
    }

    pub fn read_bytes_aligned(&mut self, mut buf: &mut [u8]) -> Result<()> {
        if self.global_pos % 8 != 0 {
            return Err(Error::NotAligned);
        }
        let direct_read_bytes = ((buf.len() as u64 * 8).min(self.bits_left as u64) / 8) as usize;
        for b in &mut buf[0..direct_read_bytes] {
            *b = self.read_bits_inner(8) as u8;
        }
        buf = &mut buf[direct_read_bytes..];
        if buf.is_empty() {
            return Ok(());
        }

        let byte_copy_len = self.left_in_buffer().len().min(buf.len());
        buf[..byte_copy_len].copy_from_slice(&self.left_in_buffer()[..byte_copy_len]);
        self.buf_offset += byte_copy_len;
        self.global_pos += byte_copy_len as u64 * 8;
        buf = &mut buf[byte_copy_len..];
        if buf.is_empty() {
            return Ok(());
        }

        self.reader.read_exact(buf)?;
        self.global_pos += buf.len() as u64 * 8;
        Ok(())
    }
}

impl<R: Read + Seek> Bitstream<R> {
    pub fn seek_to_bookmark(&mut self, Bookmark(target): Bookmark) -> Result<()> {
        let byte_offset = target / 8;
        let bit_offset = target % 8;

        self.buf_valid_len = 0;
        self.buf_offset = 0;
        self.current = 0;
        self.bits_left = 0;

        self.reader.seek(std::io::SeekFrom::Start(byte_offset))?;
        self.global_pos = byte_offset * 8;
        self.read_bits(bit_offset as u32)?;

        Ok(())
    }
}

pub struct RewindMarker<'b, R> {
    read_data: Vec<u8>,
    original: &'b mut Bitstream<R>,
}

impl<R: Read> Read for RewindMarker<'_, R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self.original.reader.read(buf) {
            Ok(0) => Ok(0),
            Ok(n) => {
                self.read_data.extend_from_slice(&buf[..n]);
                Ok(n)
            },
            Err(e) => Err(e),
        }
    }
}

impl<R> Drop for RewindMarker<'_, R> {
    fn drop(&mut self) {
        let read_data = &self.read_data;
        if read_data.is_empty() {
            return;
        }

        let empty_area = &mut self.original.buf[self.original.buf_valid_len..];
        if empty_area.len() < read_data.len() {
            let left_in_buffer = self.original.left_in_buffer();
            let mut buf = vec![0u8; left_in_buffer.len() + read_data.len()];
            let (l, r) = buf.split_at_mut(left_in_buffer.len());
            l.copy_from_slice(left_in_buffer);
            r.copy_from_slice(read_data);
            self.original.buf_offset = 0;
            self.original.buf_valid_len = buf.len();
            self.original.buf = buf;
        } else {
            empty_area[..read_data.len()].copy_from_slice(read_data);
            self.original.buf_valid_len += read_data.len();
        }
    }
}

impl<R> Bitstream<RewindMarker<'_, R>> {
    pub fn commit(self) {
        let mut marker = self.reader;
        marker.read_data.clear();
        let orig_bitstream = &mut *marker.original;

        orig_bitstream.global_pos = self.global_pos;
        orig_bitstream.buf = self.buf;
        orig_bitstream.buf_valid_len = self.buf_valid_len;
        orig_bitstream.buf_offset = self.buf_offset;
        orig_bitstream.current = self.current;
        orig_bitstream.bits_left = self.bits_left;
        drop(marker);
    }
}

/// Name type which is read by some JPEG XL headers.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Name(String);

impl<Ctx> Bundle<Ctx> for Name {
    type Error = Error;

    fn parse<R: Read>(bitstream: &mut Bitstream<R>, _: Ctx) -> Result<Self> {
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
