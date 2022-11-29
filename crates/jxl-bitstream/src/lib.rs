use std::io::prelude::*;

pub mod header;

pub trait Bundle {
    fn parse<R: std::io::Read>(bitstream: &mut crate::Bitstream<R>) -> crate::Result<Self> where Self: Sized;
}

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    NonZeroPadding,
    InvalidFloat,
    InvalidEnum {
        name: &'static str,
        value: u32,
    },
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => {
                write!(f, "I/O error: {}", e)
            },
            Self::NonZeroPadding => {
                write!(f, "PadZeroToByte() read non-zero bits")
            },
            Self::InvalidFloat => {
                write!(f, "F16() read NaN or Infinity")
            },
            Self::InvalidEnum { name, value } => {
                write!(f, "Enum({}) read invalid enum value of {}", name, value)
            },
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

pub type Result<T> = std::result::Result<T, Error>;

#[macro_export]
macro_rules! expand_u32 {
    ($bitstream:ident; $($rest:tt)*) => {
        $bitstream.read_bits(2)
            .and_then(|selector| $crate::expand_u32!(@expand $bitstream, selector, 0; $($rest)*,))
    };
    (@expand $bitstream:ident, $selector:ident, $counter:expr;) => {
        unreachable!()
    };
    (@expand $bitstream:ident, $selector:ident, $counter:expr; $c:literal, $($rest:tt)*) => {
        if $selector == $counter {
            $crate::read_bits!($bitstream, $c)
        } else {
            $crate::expand_u32!(@expand $bitstream, $selector, $counter + 1; $($rest)*)
        }
    };
    (@expand $bitstream:ident, $selector:ident, $counter:expr; u($n:literal), $($rest:tt)*) => {
        if $selector == $counter {
            $crate::read_bits!($bitstream, u($n))
        } else {
            $crate::expand_u32!(@expand $bitstream, $selector, $counter + 1; $($rest)*)
        }
    };
    (@expand $bitstream:ident, $selector:ident, $counter:expr; $c:literal + u($n:literal), $($rest:tt)*) => {
        if $selector == $counter {
            $crate::read_bits!($bitstream, $c + u($n))
        } else {
            $crate::expand_u32!(@expand $bitstream, $selector, $counter + 1; $($rest)*)
        }
    };
}

#[macro_export]
macro_rules! read_bits {
    ($bistream:ident, $c:literal) => {
        $crate::Result::Ok($c)
    };
    ($bitstream:ident, u($n:literal)) => {
        $bitstream.read_bits($n)
    };
    ($bitstream:ident, $c:literal + u($n:literal)) => {
        $bitstream.read_bits($n).map(|v| v.wrapping_add($c))
    };
    ($bitstream:ident, U32($($args:tt)+)) => {
        $crate::expand_u32!($bitstream; $($args)+)
    };
    ($bitstream:ident, U64) => {
        $bitstream.read_bits(2)
            .and_then(|selector| match selector {
                0 => Ok(0u64),
                1 => $crate::read_bits!($bitstream, 1 + u(4)).map(|v| v as u64),
                2 => $crate::read_bits!($bitstream, 17 + u(8)).map(|v| v as u64),
                3 => (|| -> $crate::Result<u64> {
                    let mut value = $bitstream.read_bits(12)? as u64;
                    let mut shift = 12u32;
                    while $bitstream.read_bits(1)? == 1 {
                        if shift == 60 {
                            value |= ($bitstream.read_bits(4)? as u64) << shift;
                            break;
                        }
                        value |= ($bitstream.read_bits(8)? as u64) << shift;
                        shift += 8;
                    }
                    Ok(value)
                })(),
                _ => unreachable!(),
            })
    };
    ($bitstream:ident, F16) => {
        $bitstream.read_f16_as_f32()
    };
    ($bitstream:ident, Bool) => {
        $bitstream.read_bits(1).map(|v| v == 1)
    };
    ($bitstream:ident, Enum($enumtype:ty)) => {
        $crate::read_bits!($bitstream, U32(0, 1, 2 + u(4), 18 + u(6)))
            .and_then(|v| {
                <$enumtype as TryFrom<u32>>::try_from(v).map_err(|_| $crate::Error::InvalidEnum {
                    name: stringify!($enumtype),
                    value: v,
                })
            })
    };
    ($bitstream:ident, ZeroPadToByte) => {
        $bitstream.zero_pad_to_byte()
    };
    ($bitstream:ident, Bundle($bundle:ty)) => {
        <$bundle as $crate::Bundle>::parse($bitstream)
    };
    ($bitstream:ident, Vec[$($inner:tt)*]; $count:expr) => {
        {
            let count = $count as usize;
            (0..count)
                .into_iter()
                .map(|_| $crate::read_bits!($bitstream, $($inner)*))
                .collect::<$crate::Result<Vec<_>>>()
        }
    };
    ($bitstream:ident, Array[$($inner:tt)*]; $count:expr) => {
        (|| -> $crate::Result<[_; $count]> {
            let mut ret = [Default::default(); $count];
            for point in &mut ret {
                *point = $crate::read_bits!($bitstream, $($inner)*)?;
            }
            Ok(ret)
        })()
    };
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

    pub fn global_pos(&self) -> (u64, u32) {
        (self.global_pos / 8, (self.global_pos % 8) as u32)
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
            let mut ret = self.current;
            let mut bits = self.bits_left;
            self.current = 0;
            self.bits_left = 0;

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
}
