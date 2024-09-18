use crate::{Error, Result};

/// Bitstream reader with borrowed in-memory buffer.
///
/// Implementation is mostly from [jxl-rs].
///
/// [jxl-rs]: https://github.com/libjxl/jxl-rs
#[derive(Clone)]
pub struct Bitstream<'buf> {
    bytes: &'buf [u8],
    buf: u64,
    num_read_bits: usize,
    remaining_buf_bits: usize,
}

impl std::fmt::Debug for Bitstream<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Bitstream")
            .field(
                "bytes",
                &format_args!(
                    "({} byte{} left)",
                    self.bytes.len(),
                    if self.bytes.len() == 1 { "" } else { "s" },
                ),
            )
            .field("buf", &format_args!("0x{:016x}", self.buf))
            .field("num_read_bits", &self.num_read_bits)
            .field("remaining_buf_bits", &self.remaining_buf_bits)
            .finish()
    }
}

impl<'buf> Bitstream<'buf> {
    /// Create a new bitstream reader.
    #[inline]
    pub fn new(bytes: &'buf [u8]) -> Self {
        Self {
            bytes,
            buf: 0,
            num_read_bits: 0,
            remaining_buf_bits: 0,
        }
    }

    /// Returns the number of bits that are read or skipped.
    #[inline]
    pub fn num_read_bits(&self) -> usize {
        self.num_read_bits
    }
}

impl Bitstream<'_> {
    /// Fills bit buffer from byte buffer.
    #[inline]
    fn refill(&mut self) {
        if let &[b0, b1, b2, b3, b4, b5, b6, b7, ..] = self.bytes {
            let bits = u64::from_le_bytes([b0, b1, b2, b3, b4, b5, b6, b7]);
            self.buf |= bits << self.remaining_buf_bits;
            let read_bytes = (63 - self.remaining_buf_bits) >> 3;
            self.remaining_buf_bits |= 56;
            // SAFETY: read_bytes < 8, self.bytes.len() >= 8 (from the pattern).
            self.bytes = unsafe {
                std::slice::from_raw_parts(
                    self.bytes.as_ptr().add(read_bytes),
                    self.bytes.len() - read_bytes,
                )
            };
        } else {
            self.refill_slow()
        }
    }

    #[inline(never)]
    fn refill_slow(&mut self) {
        while self.remaining_buf_bits < 56 {
            let Some((&b, next)) = self.bytes.split_first() else {
                return;
            };

            self.buf |= (b as u64) << self.remaining_buf_bits;
            self.remaining_buf_bits += 8;
            self.bytes = next;
        }
    }
}

impl Bitstream<'_> {
    /// Peeks bits from bitstream, without consuming them.
    ///
    /// This method refills the bit buffer.
    #[inline]
    pub fn peek_bits(&mut self, n: usize) -> u32 {
        debug_assert!(n <= 32);
        self.refill();
        (self.buf & ((1u64 << n) - 1)) as u32
    }

    /// Peeks bits from bitstream, without consuming them.
    ///
    /// This method refills the bit buffer.
    #[inline]
    pub fn peek_bits_const<const N: usize>(&mut self) -> u32 {
        debug_assert!(N <= 32);
        self.refill();
        (self.buf & ((1u64 << N) - 1)) as u32
    }

    /// Peeks bits from already filled bitstream, without consuming them.
    ///
    /// This method *does not* refill the bit buffer.
    #[inline]
    pub fn peek_bits_prefilled(&mut self, n: usize) -> u32 {
        debug_assert!(n <= 32);
        (self.buf & ((1u64 << n) - 1)) as u32
    }

    /// Peeks bits from already filled bitstream, without consuming them.
    ///
    /// This method *does not* refill the bit buffer.
    #[inline]
    pub fn peek_bits_prefilled_const<const N: usize>(&mut self) -> u32 {
        debug_assert!(N <= 32);
        (self.buf & ((1u64 << N) - 1)) as u32
    }

    /// Consumes bits in bit buffer.
    ///
    /// # Errors
    /// This method returns `Err(Io(std::io::ErrorKind::UnexpectedEof))` when there are not enough
    /// bits in the bit buffer.
    #[inline]
    pub fn consume_bits(&mut self, n: usize) -> Result<()> {
        self.remaining_buf_bits = self
            .remaining_buf_bits
            .checked_sub(n)
            .ok_or(Error::Io(std::io::ErrorKind::UnexpectedEof.into()))?;
        self.num_read_bits += n;
        self.buf >>= n;
        Ok(())
    }

    /// Consumes bits in bit buffer.
    ///
    /// # Errors
    /// This method returns `Err(Io(std::io::ErrorKind::UnexpectedEof))` when there are not enough
    /// bits in the bit buffer.
    #[inline]
    pub fn consume_bits_const<const N: usize>(&mut self) -> Result<()> {
        self.remaining_buf_bits = self
            .remaining_buf_bits
            .checked_sub(N)
            .ok_or(Error::Io(std::io::ErrorKind::UnexpectedEof.into()))?;
        self.num_read_bits += N;
        self.buf >>= N;
        Ok(())
    }

    /// Read and consume bits from bitstream.
    #[inline]
    pub fn read_bits(&mut self, n: usize) -> Result<u32> {
        let ret = self.peek_bits(n);
        self.consume_bits(n)?;
        Ok(ret)
    }

    #[inline(never)]
    pub fn skip_bits(&mut self, mut n: usize) -> Result<()> {
        if let Some(next_remaining_bits) = self.remaining_buf_bits.checked_sub(n) {
            self.num_read_bits += n;
            self.remaining_buf_bits = next_remaining_bits;
            self.buf >>= n;
            return Ok(());
        }

        n -= self.remaining_buf_bits;
        self.num_read_bits += self.remaining_buf_bits;
        self.buf = 0;
        self.remaining_buf_bits = 0;
        if n > self.bytes.len() * 8 {
            self.num_read_bits += self.bytes.len() * 8;
            return Err(Error::Io(std::io::ErrorKind::UnexpectedEof.into()));
        }

        self.num_read_bits += n;
        self.bytes = &self.bytes[n / 8..];
        n %= 8;
        self.refill();
        self.remaining_buf_bits = self
            .remaining_buf_bits
            .checked_sub(n)
            .ok_or(Error::Io(std::io::ErrorKind::UnexpectedEof.into()))?;
        self.buf >>= n;
        Ok(())
    }

    /// Performs `ZeroPadToByte` as defined in the JPEG XL specification.
    pub fn zero_pad_to_byte(&mut self) -> Result<()> {
        let byte_boundary = (self.num_read_bits + 7) / 8 * 8;
        let n = byte_boundary - self.num_read_bits;
        if self.read_bits(n)? != 0 {
            Err(Error::NonZeroPadding)
        } else {
            Ok(())
        }
    }
}

impl Bitstream<'_> {
    /// Reads an `U32` as defined in the JPEG XL specification.
    ///
    /// # Example
    ///
    /// ```
    /// use jxl_bitstream::{Bitstream, U};
    ///
    /// let buf = [0b110010];
    /// let mut bitstream = Bitstream::new(&buf);
    /// let val = bitstream.read_u32(1, U(2), 3 + U(4), 19 + U(8)).expect("failed to read data");
    /// assert_eq!(val, 15);
    /// ```
    #[inline]
    pub fn read_u32(
        &mut self,
        d0: impl Into<U32Specifier>,
        d1: impl Into<U32Specifier>,
        d2: impl Into<U32Specifier>,
        d3: impl Into<U32Specifier>,
    ) -> Result<u32> {
        let d = match self.read_bits(2)? {
            0 => d0.into(),
            1 => d1.into(),
            2 => d2.into(),
            3 => d3.into(),
            _ => unreachable!(),
        };
        match d {
            U32Specifier::Constant(x) => Ok(x),
            U32Specifier::BitsOffset(offset, n) => self.read_bits(n).map(|x| x + offset),
        }
    }

    /// Reads an `U64` as defined in the JPEG XL specification.
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
            }
            _ => unreachable!(),
        })
    }

    /// Reads a `Bool` as defined in the JPEG XL specification.
    #[inline]
    pub fn read_bool(&mut self) -> Result<bool> {
        self.read_bits(1).map(|x| x != 0)
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

    /// Reads an enum as defined in the JPEG XL specification.
    pub fn read_enum<E: TryFrom<u32>>(&mut self) -> Result<E> {
        let v = self.read_u32(0, 1, 2 + U(4), 18 + U(6))?;
        E::try_from(v).map_err(|_| Error::InvalidEnum {
            name: std::any::type_name::<E>(),
            value: v,
        })
    }
}

/// Bit specifier for [`Bitstream::read_u32`].
pub enum U32Specifier {
    Constant(u32),
    BitsOffset(u32, usize),
}

/// Bit count for use in [`Bitstream::read_u32`].
pub struct U(pub usize);

impl From<u32> for U32Specifier {
    fn from(value: u32) -> Self {
        Self::Constant(value)
    }
}

impl From<U> for U32Specifier {
    fn from(value: U) -> Self {
        Self::BitsOffset(0, value.0)
    }
}

impl std::ops::Add<U> for u32 {
    type Output = U32Specifier;

    fn add(self, rhs: U) -> Self::Output {
        U32Specifier::BitsOffset(self, rhs.0)
    }
}
