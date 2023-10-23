use jxl_bitstream::{Bitstream, read_bits};

use crate::{Error, Result};

#[derive(Debug)]
pub struct Histogram {
    buckets: Vec<Bucket>,
    log_bucket_size: u32,
    single_symbol: Option<u16>,
}

#[derive(Debug)]
struct Bucket {
    dist: u16,
    alias_symbol: u16,
    alias_offset: u16,
    alias_cutoff: u16,
}

impl Histogram {
    pub fn parse(bitstream: &mut Bitstream, log_alphabet_size: u32) -> Result<Self> {
        let table_size = (1u16 << log_alphabet_size) as usize;
        let log_bucket_size = 12 - log_alphabet_size;
        let bucket_size = 1u16 << log_bucket_size;

        let alphabet_size;
        let mut dist = vec![0u16; table_size];
        if read_bits!(bitstream, Bool)? {
            if read_bits!(bitstream, Bool)? {
                // binary
                let v0 = Self::read_u8(bitstream)? as usize;
                let v1 = Self::read_u8(bitstream)? as usize;
                if v0 == v1 {
                    return Err(Error::InvalidAnsHistogram);
                }
                alphabet_size = v0.max(v1) + 1;
                if alphabet_size > table_size {
                    return Err(Error::InvalidAnsHistogram);
                }

                let prob = bitstream.read_bits(12)? as u16;
                dist[v0] = prob;
                dist[v1] = (1u16 << 12) - prob;
            } else {
                // unary
                let val = Self::read_u8(bitstream)? as usize;
                alphabet_size = val + 1;
                if alphabet_size > table_size {
                    return Err(Error::InvalidAnsHistogram);
                }

                dist[val] = 1 << 12;
            }
        } else if read_bits!(bitstream, Bool)? {
            // evenly distributed
            alphabet_size = Self::read_u8(bitstream)? as usize + 1;
            if alphabet_size > table_size {
                return Err(Error::InvalidAnsHistogram);
            }

            let base = (1usize << 12) / alphabet_size;
            let leftover = (1usize << 12) % alphabet_size;
            dist[0..leftover].fill(base as u16 + 1);
            dist[leftover..alphabet_size].fill(base as u16);
        } else {
            // compressed distribution info
            let mut len = 0usize;
            while len < 3 {
                if read_bits!(bitstream, Bool)? {
                    len += 1;
                } else {
                    break;
                }
            }
            let shift = (bitstream.read_bits(len)? + (1 << len) - 1) as i16;
            if shift > 13 {
                return Err(Error::InvalidAnsHistogram);
            }
            alphabet_size = Self::read_u8(bitstream)? as usize + 3;
            if alphabet_size > table_size {
                return Err(Error::InvalidAnsHistogram);
            }
            let mut repeat_ranges = Vec::new();

            let mut omit_data = None;
            let mut idx = 0;
            while idx < alphabet_size {
                dist[idx] = read_prefix(bitstream)?;
                if dist[idx] == 13 {
                    let repeat_count = Self::read_u8(bitstream)? as usize + 4;
                    if idx + repeat_count > alphabet_size {
                        return Err(Error::InvalidAnsHistogram);
                    }
                    repeat_ranges.push(idx..(idx + repeat_count));
                    idx += repeat_count;
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
                idx += 1;
            }
            let Some((_, omit_pos)) = omit_data else {
                return Err(Error::InvalidAnsHistogram);
            };
            if dist.get(omit_pos + 1) == Some(&13) {
                return Err(Error::InvalidAnsHistogram);
            }

            let mut repeat_range_idx = 0usize;
            let mut acc = 0;
            let mut prev_dist = 0u16;
            for (idx, code) in dist.iter_mut().enumerate() {
                if repeat_range_idx < repeat_ranges.len() && repeat_ranges[repeat_range_idx].start <= idx {
                    if repeat_ranges[repeat_range_idx].end == idx {
                        repeat_range_idx += 1;
                    } else {
                        *code = prev_dist;
                        acc += *code;
                        if acc > (1 << 12) {
                            return Err(Error::InvalidAnsHistogram);
                        }
                        continue;
                    }
                }

                if *code == 0 {
                    prev_dist = 0;
                    continue;
                }
                if idx == omit_pos {
                    prev_dist = 0;
                    continue;
                }
                if *code > 1 {
                    let zeros = (*code - 1) as i16;
                    let bitcount = (shift - ((12 - zeros) >> 1)).clamp(0, zeros);
                    *code = (1 << zeros) + ((bitstream.read_bits(bitcount as usize)? as u16) << (zeros - bitcount));
                }
                prev_dist = *code;
                acc += *code;
                if acc > (1 << 12) {
                    return Err(Error::InvalidAnsHistogram);
                }
            }
            dist[omit_pos] = (1 << 12) - acc;
        }

        if let Some(single_sym_idx) = dist.iter().position(|&d| d == 1 << 12) {
            let buckets = dist.into_iter()
                .enumerate()
                .map(|(i, dist)| Bucket {
                    dist,
                    alias_symbol: single_sym_idx as u16,
                    alias_offset: bucket_size * i as u16,
                    alias_cutoff: 0,
                })
                .collect();
            return Ok(Self {
                buckets,
                log_bucket_size,
                single_symbol: Some(single_sym_idx as u16),
            });
        }

        let mut buckets: Vec<_> = dist
            .into_iter()
            .enumerate()
            .map(|(i, dist)| Bucket {
                dist,
                alias_symbol: if i < alphabet_size { i as u16 } else { 0 },
                alias_offset: 0,
                alias_cutoff: dist,
            })
            .collect();

        let mut underfull = Vec::new();
        let mut overfull = Vec::new();
        for (idx, &Bucket { dist, .. }) in buckets.iter().enumerate() {
            match dist.cmp(&bucket_size) {
                std::cmp::Ordering::Less => underfull.push(idx),
                std::cmp::Ordering::Equal => {},
                std::cmp::Ordering::Greater => overfull.push(idx),
            }
        }
        while let (Some(o), Some(u)) = (overfull.pop(), underfull.pop()) {
            let by = bucket_size - buckets[u].alias_cutoff;
            buckets[o].alias_cutoff -= by;
            buckets[u].alias_symbol = o as u16;
            buckets[u].alias_offset = buckets[o].alias_cutoff;
            match buckets[o].alias_cutoff.cmp(&bucket_size) {
                std::cmp::Ordering::Less => underfull.push(o),
                std::cmp::Ordering::Equal => {},
                std::cmp::Ordering::Greater => overfull.push(o),
            }
        }

        for (idx, bucket) in buckets.iter_mut().enumerate() {
            if bucket.alias_cutoff == bucket_size {
                bucket.alias_symbol = idx as u16;
                bucket.alias_offset = 0;
                bucket.alias_cutoff = 0;
            } else {
                bucket.alias_offset -= bucket.alias_cutoff;
            }
        }

        Ok(Self {
            buckets,
            log_bucket_size,
            single_symbol: None,
        })
    }

    fn read_u8(bitstream: &mut Bitstream) -> Result<u8> {
        Ok(if read_bits!(bitstream, Bool)? {
            let n = bitstream.read_bits(3)? as usize;
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
        let bucket = &self.buckets[i];
        if pos >= bucket.alias_cutoff {
            (bucket.alias_symbol, bucket.alias_offset + pos)
        } else {
            (i as u16, pos)
        }
    }

    pub fn read_symbol(&self, bitstream: &mut Bitstream, state: &mut u32) -> Result<u16> {
        let idx = (*state & 0xfff) as u16;
        let (symbol, offset) = self.map_alias(idx);
        *state = (*state >> 12) * (self.buckets[symbol as usize].dist as u32) + offset as u32;
        if *state < (1 << 16) {
            match bitstream.read_bits(16) {
                Ok(bits) => {
                    *state = (*state << 16) | bits;
                },
                Err(e) => return Err(e.into()),
            }
        }
        Ok(symbol)
    }

    #[inline]
    pub fn single_symbol(&self) -> Option<u16> {
        self.single_symbol
    }
}

fn read_prefix(bitstream: &mut Bitstream) -> Result<u16> {
    Ok(match bitstream.read_bits(3)? {
        0 => 10,
        1 => {
            for val in [4, 0, 11, 13] {
                if bitstream.read_bool()? {
                    return Ok(val);
                }
            }
            12
        },
        2 => 7,
        3 => {
            if bitstream.read_bool()? {
                1
            } else {
                3
            }
        },
        4 => 6,
        5 => 8,
        6 => 9,
        7 => {
            if bitstream.read_bool()? {
                2
            } else {
                5
            }
        },
        _ => unreachable!(),
    })
}
