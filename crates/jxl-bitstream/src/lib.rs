use std::io::prelude::*;

mod error;
mod macros;

pub mod header;

pub use error::{Error, Result};
pub use macros::{unpack_signed, unpack_signed_u64};

pub trait Bundle<Ctx = ()>: Sized {
    type Error;

    fn parse<R: Read>(bitstream: &mut Bitstream<R>, ctx: Ctx) -> std::result::Result<Self, Self::Error>;
}

pub trait BundleDefault<Ctx = ()>: Sized {
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

impl<R> Bitstream<R> {
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

    fn left_in_buffer(&self) -> &[u8] {
        &self.buf[self.buf_offset..self.buf_valid_len]
    }

    pub fn bookmark(&self) -> Bookmark {
        Bookmark(self.global_pos)
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
        buf_current.copy_from_slice(&buf_left[..buf_to_read]);

        self.buf_offset += buf_to_read;
        self.current = u64::from_le_bytes(buf_current);
        self.bits_left = (buf_to_read * 8) as u32;

        Ok(())
    }

    fn read_bits_inner(&mut self, n: u32) -> u64 {
        if n == 0 {
            return 0;
        }

        assert!(self.bits_left >= n);
        let mask = (1 << n) - 1;
        let data = self.current & mask;
        self.current >>= n;
        self.bits_left -= n;
        self.global_pos += n as u64;
        data
    }

    pub fn read_bits(&mut self, n: u32) -> Result<u32> {
        if n == 0 {
            return Ok(0);
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

    pub fn read_bool(&mut self) -> Result<bool> {
        Ok(self.read_bits(1)? == 1)
    }

    pub fn read_f16_as_f32(&mut self) -> Result<f32> {
        let v = self.read_bits(16)?;
        let mantissa = v & 0x3ff; // 10 bits
        let exponent = (v >> 10) & 0x1f; // 5 bits
        let is_neg = v & 0x8000 != 0;
        if exponent == 0x1f {
            Err(Error::InvalidFloat)
        } else {
            let mantissa = mantissa << 13; // 23 bits
            let exponent = exponent + 112;
            let bitpattern = mantissa | (exponent << 23) | if is_neg { 0x80000000 } else { 0 };
            Ok(f32::from_bits(bitpattern))
        }
    }

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
        while diff > 0 {
            let bits = diff.min(4096 * 8);
            self.read_bits(bits as u32)?;
            diff -= bits;
        }
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
