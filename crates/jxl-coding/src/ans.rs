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
        let table_size = 1u16 << log_alphabet_size;
        let log_bucket_size = 12 - log_alphabet_size;
        let bucket_size = 1u16 << log_bucket_size;

        let alphabet_size;
        let mut dist = vec![0u16; table_size as usize];
        if read_bits!(bitstream, Bool)? {
            alphabet_size = table_size as usize;
            if read_bits!(bitstream, Bool)? {
                // binary
                let v0 = Self::read_u8(bitstream)?;
                let v1 = Self::read_u8(bitstream)?;
                if v0 == v1 {
                    return Err(Error::InvalidAnsHistogram);
                }
                let prob = bitstream.read_bits(12)? as u16;
                dist[v0 as usize] = prob;
                dist[v1 as usize] = (1u16 << 12) - prob;
            } else {
                // unary
                let val = Self::read_u8(bitstream)?;
                dist[val as usize] = 1 << 12;
            }
        } else if read_bits!(bitstream, Bool)? {
            // evenly distributed
            alphabet_size = Self::read_u8(bitstream)? as usize + 1;
            let base = (1usize << 12) / alphabet_size;
            let leftover = (1usize << 12) % alphabet_size;
            dist[0..leftover].fill(base as u16 + 1);
            dist[leftover..alphabet_size].fill(base as u16);
        } else {
            // compressed distribution info
            let mut len = 0u32;
            while len < 3 {
                if read_bits!(bitstream, Bool)? {
                    len += 1;
                } else {
                    break;
                }
            }
            let shift = (bitstream.read_bits(len)? + (1 << len) - 1) as u16;
            if shift > 13 {
                return Err(Error::InvalidAnsHistogram);
            }
            alphabet_size = Self::read_u8(bitstream)? as usize + 3;
            let mut repeat_ranges = Vec::new();
            let histogram = crate::prefix::Histogram::prefix_code_for_ans();

            let mut omit_data = None;
            let mut idx = 0;
            while idx < alphabet_size {
                dist[idx] = histogram.read_symbol(bitstream)?;
                if dist[idx] == 13 {
                    let repeat_count = Self::read_u8(bitstream)? as usize + 4;
                    repeat_ranges.push(idx..(idx + repeat_count));
                    idx += repeat_count;
                    dist[idx] = 0;
                    continue;
                }
                match &mut omit_data {
                    Some((log, pos)) => {
                        if dist[idx] > *log {
                            *log = dist[idx];
                            *pos = idx;
                        }
                    },
                    data => {
                        *data = Some((dist[idx], idx));
                    },
                }
            }
            let (_, omit_pos) = omit_data.unwrap();
            if dist.get(omit_pos + 1) == Some(&13) {
                return Err(Error::InvalidAnsHistogram);
            }

            let mut acc = 0;
            for (idx, code) in dist.iter_mut().enumerate() {
                if *code == 0 {
                    continue;
                }
                if idx == omit_pos {
                    continue;
                }
                if *code > 1 {
                    let zeros = *code - 1;
                    let bitcount = (shift - ((12 - zeros) >> 1)).clamp(0, zeros);
                    *code = (1 << zeros) + ((bitstream.read_bits(bitcount as u32)? as u16) << (zeros - bitcount));
                }
                acc += *code;
                if acc > (1 << 12) {
                    return Err(Error::InvalidAnsHistogram);
                }
            }
            dist[omit_pos] = (1 << 12) - acc;
        }

        if let Some(single_sym_idx) = dist.iter().position(|&d| d == 1 << 12) {
            let symbols = vec![single_sym_idx as u16; table_size as usize];
            let offsets = (0..table_size).map(|i| (bucket_size * i) as u16).collect();
            let cutoffs = vec![0u16; table_size as usize];
            return Ok(Self {
                dist,
                symbols,
                offsets,
                cutoffs,
                log_bucket_size,
            });
        }

        let mut cutoffs = dist.clone();
        let mut symbols = (0..(alphabet_size as u16)).collect::<Vec<_>>();
        symbols.resize(table_size as usize, 0);
        let mut offsets = vec![0u16; table_size as usize];

        let mut underfull = Vec::new();
        let mut overfull = Vec::new();
        for (idx, d) in dist.iter().enumerate() {
            match d.cmp(&bucket_size) {
                std::cmp::Ordering::Less => underfull.push(idx),
                std::cmp::Ordering::Equal => {},
                std::cmp::Ordering::Greater => overfull.push(idx),
            }
        }
        while let (Some(o), Some(u)) = (overfull.pop(), underfull.pop()) {
            let by = bucket_size - cutoffs[u];
            cutoffs[o] -= by;
            symbols[u] = o as u16;
            offsets[u] = cutoffs[o];
            match cutoffs[o].cmp(&bucket_size) {
                std::cmp::Ordering::Less => underfull.push(o),
                std::cmp::Ordering::Equal => {},
                std::cmp::Ordering::Greater => overfull.push(o),
            }
        }

        for idx in 0..(table_size as usize) {
            if cutoffs[idx] == bucket_size {
                symbols[idx] = idx as u16;
                offsets[idx] = 0;
                cutoffs[idx] = 0;
            } else {
                offsets[idx] -= cutoffs[idx];
            }
        }

        Ok(Self {
            dist,
            symbols,
            offsets,
            cutoffs,
            log_bucket_size,
        })
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
