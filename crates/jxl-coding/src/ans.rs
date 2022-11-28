use std::io::Read;

use jxl_bitstream::{Bitstream, read_bits};

use crate::{Error, Result};

#[derive(Debug)]
pub struct Histogram {
    dist: Vec<u16>,
    symbols: Vec<u16>,
    offsets: Vec<u16>,
    cutoffs: Vec<u16>,
    log_bucket_size: u32,
}

impl Histogram {
    pub fn parse<R: Read>(bitstream: &mut Bitstream<R>, log_alphabet_size: u32) -> Result<Self> {
        let table_size = 1u32 << log_alphabet_size;
        let log_bucket_size = 12 - log_alphabet_size;
        let bucket_size = 1u32 << log_bucket_size;

        let mut dist = vec![0u16; table_size as usize];
        if read_bits!(bitstream, Bool)? {
            if read_bits!(bitstream, Bool)? {
                let v0 = Self::read_u8(bitstream)?;
                let v1 = Self::read_u8(bitstream)?;
                if v0 == v1 {
                    return Err(Error::InvalidAnsHistogram);
                }
                let prob = bitstream.read_bits(12)? as u16;
                dist[v0 as usize] = prob;
                dist[v1 as usize] = (1u16 << 12) - prob;
            } else {
                let val = Self::read_u8(bitstream)?;
                dist[val as usize] = 1 << 12;
            }
        } else if read_bits!(bitstream, Bool)? {
            let alphabet_size = Self::read_u8(bitstream)? + 1;
        } else {
        }
        todo!()
    }

    fn read_u8<R: Read>(bitstream: &mut Bitstream<R>) -> Result<u8> {
        Ok(if read_bits!(bitstream, Bool)? {
            let n = bitstream.read_bits(3)?;
            ((1 << n) + bitstream.read_bits(n)?) as u8
        } else {
            0
        })
    }
}

impl Histogram {
    fn map_alias(&self, idx: u16) -> (u16, u16) {
        let i = (idx >> self.log_bucket_size) as usize;
        let pos = idx & ((1 << self.log_bucket_size) - 1);
        if pos >= self.cutoffs[i] {
            (self.symbols[i], self.offsets[i] + pos)
        } else {
            (i as u16, pos)
        }
    }

    pub fn read_symbol<R: Read>(&self, bitstream: &mut Bitstream<R>, state: &mut u32) -> Result<u16> {
        let idx = (*state & 0xfff) as u16;
        let (symbol, offset) = self.map_alias(idx);
        *state = (*state >> 12) * (self.dist[symbol as usize] as u32) + offset as u32;
        if *state < (1 << 16) {
            *state = (*state << 16) | bitstream.read_bits(16)?;
        }
        Ok(symbol)
    }
}
