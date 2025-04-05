use jxl_bitstream::Bitstream;

use crate::{CodingResult, Error};

#[derive(Debug)]
pub struct Histogram {
    buckets: Vec<Bucket>,
    log_bucket_size: u32,
    bucket_mask: u32,
    single_symbol: Option<u32>,
}

// Ported from libjxl. log_alphabet_size <= 8 and log_bucket_size <= 7, so u8 is sufficient for
// symbols and cutoffs. alias_dist_xor allows branchless dist computation for alias symbol.
#[derive(Debug, Copy, Clone)]
#[repr(C)]
struct Bucket {
    alias_symbol: u8,
    alias_cutoff: u8,
    dist: u16,
    alias_offset: u16,
    alias_dist_xor: u16,
}

impl Histogram {
    // log_alphabet_size: 5 + u(2)
    pub fn parse(bitstream: &mut Bitstream, log_alphabet_size: u32) -> CodingResult<Self> {
        #[derive(Debug)]
        struct WorkingBucket {
            dist: u16,
            alias_symbol: u16,
            alias_offset: u16,
            alias_cutoff: u16,
        }

        debug_assert!((5..=8).contains(&log_alphabet_size));
        let table_size = (1u16 << log_alphabet_size) as usize;
        // 4 <= log_bucket_size <= 7
        let log_bucket_size = 12 - log_alphabet_size;
        let bucket_size = 1u16 << log_bucket_size;

        let alphabet_size;
        let mut dist = vec![0u16; table_size];
        if bitstream.read_bool()? {
            if bitstream.read_bool()? {
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
        } else if bitstream.read_bool()? {
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
                if bitstream.read_bool()? {
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
                    }
                    data => {
                        *data = Some((dist[idx], idx));
                    }
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
                if repeat_range_idx < repeat_ranges.len()
                    && repeat_ranges[repeat_range_idx].start <= idx
                {
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
                    *code = (1 << zeros)
                        + ((bitstream.read_bits(bitcount as usize)? as u16) << (zeros - bitcount));
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
            let buckets = dist
                .into_iter()
                .enumerate()
                .map(|(i, dist)| Bucket {
                    dist,
                    alias_symbol: single_sym_idx as u8,
                    alias_offset: bucket_size * i as u16,
                    alias_cutoff: 0,
                    alias_dist_xor: dist ^ (1 << 12),
                })
                .collect();
            return Ok(Self {
                buckets,
                log_bucket_size,
                bucket_mask: (1 << log_bucket_size) - 1,
                single_symbol: Some(single_sym_idx as u32),
            });
        }

        let mut buckets: Vec<_> = dist
            .into_iter()
            .enumerate()
            .map(|(i, dist)| WorkingBucket {
                dist,
                alias_symbol: if i < alphabet_size { i as u16 } else { 0 },
                alias_offset: 0,
                alias_cutoff: dist,
            })
            .collect();

        let mut underfull = Vec::new();
        let mut overfull = Vec::new();
        for (idx, &WorkingBucket { dist, .. }) in buckets.iter().enumerate() {
            match dist.cmp(&bucket_size) {
                std::cmp::Ordering::Less => underfull.push(idx),
                std::cmp::Ordering::Equal => {}
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
                std::cmp::Ordering::Equal => {}
                std::cmp::Ordering::Greater => overfull.push(o),
            }
        }

        let buckets = buckets
            .iter()
            .enumerate()
            .map(|(idx, bucket)| {
                if bucket.alias_cutoff == bucket_size {
                    Bucket {
                        dist: bucket.dist,
                        alias_symbol: idx as u8,
                        alias_offset: 0,
                        alias_cutoff: 0,
                        alias_dist_xor: 0,
                    }
                } else {
                    Bucket {
                        dist: bucket.dist,
                        alias_symbol: bucket.alias_symbol as u8,
                        alias_offset: bucket.alias_offset - bucket.alias_cutoff,
                        alias_cutoff: bucket.alias_cutoff as u8,
                        alias_dist_xor: bucket.dist ^ buckets[bucket.alias_symbol as usize].dist,
                    }
                }
            })
            .collect();

        Ok(Self {
            buckets,
            log_bucket_size,
            bucket_mask: (1 << log_bucket_size) - 1,
            single_symbol: None,
        })
    }

    fn read_u8(bitstream: &mut Bitstream) -> CodingResult<u8> {
        Ok(if bitstream.read_bool()? {
            let n = bitstream.read_bits(3)?;
            ((1 << n) + bitstream.read_bits(n as usize)?) as u8
        } else {
            0
        })
    }
}

impl Histogram {
    #[inline(always)]
    pub fn read_symbol(&self, bitstream: &mut Bitstream, state: &mut u32) -> CodingResult<u32> {
        assert_eq!(std::mem::size_of::<Bucket>(), 8);
        let is_le = usize::from_le(1) == 1;

        let idx = *state & 0xfff;
        let i = (idx >> self.log_bucket_size) as usize;
        let pos = idx & self.bucket_mask;
        // SAFETY: idx is 12 bits, buckets.len() << log_bucket_size == 1 << 12.
        let bucket = unsafe { *self.buckets.get_unchecked(i) };
        // SAFETY: all bit patterns are valid.
        let bucket_int = unsafe { std::mem::transmute::<Bucket, u64>(bucket) };

        // Ported from libjxl; this makes map_alias branchless.
        let (alias_symbol, alias_cutoff, dist) = if is_le {
            (
                (bucket_int & 0xff) as usize,
                ((bucket_int >> 8) & 0xff) as u32,
                ((bucket_int >> 16) & 0xffff) as u32,
            )
        } else {
            (
                bucket.alias_symbol as usize,
                bucket.alias_cutoff as u32,
                bucket.dist as u32,
            )
        };

        let map_to_alias = pos >= alias_cutoff;
        let (offset, dist_xor) = if is_le {
            let cond_bucket = if map_to_alias { bucket_int } else { 0 };
            (
                (cond_bucket >> 32) as u32 & 0xffff,
                (cond_bucket >> 48) as u32,
            )
        } else if map_to_alias {
            (bucket.alias_offset as u32, bucket.alias_dist_xor as u32)
        } else {
            (0, 0)
        };

        let dist = dist ^ dist_xor;
        let symbol = if map_to_alias { alias_symbol } else { i };
        let offset = offset + pos;

        let next_state = (*state >> 12) * dist + offset;
        let appended_state = (next_state << 16) | bitstream.peek_bits_const::<16>();
        let select_appended = next_state < (1 << 16);
        *state = if select_appended {
            appended_state
        } else {
            next_state
        };
        bitstream.consume_bits(if select_appended { 16 } else { 0 })?;
        Ok(symbol as u32)
    }

    #[inline]
    pub fn single_symbol(&self) -> Option<u32> {
        self.single_symbol
    }
}

fn read_prefix(bitstream: &mut Bitstream) -> CodingResult<u16> {
    Ok(match bitstream.read_bits(3)? {
        0 => 10,
        1 => {
            for val in [4, 0, 11, 13] {
                if bitstream.read_bool()? {
                    return Ok(val);
                }
            }
            12
        }
        2 => 7,
        3 => {
            if bitstream.read_bool()? {
                1
            } else {
                3
            }
        }
        4 => 6,
        5 => 8,
        6 => 9,
        7 => {
            if bitstream.read_bool()? {
                2
            } else {
                5
            }
        }
        _ => unreachable!(),
    })
}
