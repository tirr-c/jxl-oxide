//! Prefix code based on Brotli
use jxl_bitstream::{read_bits, Bitstream};

use crate::{Error, Result};

const MAX_TOPLEVEL_BITS: usize = 10;

#[derive(Debug)]
pub struct Histogram {
    toplevel_bits: usize,
    toplevel_mask: u32,
    toplevel_entries: Vec<Entry>,
    second_level_entries: Vec<Entry>,
}

#[derive(Debug, Copy, Clone, Default)]
struct Entry {
    nested: bool,
    bits_or_mask: u8,
    symbol_or_offset: u16,
}

impl Histogram {
    fn with_code_lengths(code_lengths: Vec<u8>) -> Result<Self> {
        let mut syms_for_length = Vec::with_capacity(15);
        for (sym, len) in code_lengths.into_iter().enumerate() {
            let sym = sym as u16;
            if len > 0 {
                if syms_for_length.len() < len as usize {
                    syms_for_length.resize_with(len as usize, Vec::new);
                }
                syms_for_length[len as usize - 1].push(sym);
            }
        }

        let toplevel_bits = syms_for_length.len().min(MAX_TOPLEVEL_BITS);
        let mut entries = vec![Entry::default(); 1 << toplevel_bits];
        let mut current_bits = 0u16;
        for (idx, syms) in syms_for_length.iter().enumerate().take(toplevel_bits) {
            let shifts = toplevel_bits - 1 - idx;
            for &sym in syms {
                let entry = Entry {
                    nested: false,
                    bits_or_mask: (idx + 1) as u8,
                    symbol_or_offset: sym,
                };
                entries[current_bits as usize..][..(1 << shifts)].fill(entry);
                current_bits += 1u16 << shifts;
            }
        }

        let mut second_level_entries = Vec::new();
        if toplevel_bits < syms_for_length.len() {
            let mut remaining_entries = Vec::new();
            let mut remaining_entry_bits = 0usize;
            for (idx, syms) in syms_for_length.iter().enumerate().skip(toplevel_bits) {
                if syms.is_empty() {
                    continue;
                }

                let chunk_size_bits = idx + 1 - toplevel_bits;
                let chunk_size = 1usize << chunk_size_bits;
                let mut chunk = Vec::with_capacity(chunk_size);
                if !remaining_entries.is_empty() {
                    let mult = 1usize << (chunk_size_bits - remaining_entry_bits);
                    for entry in remaining_entries {
                        for _ in 0..mult {
                            chunk.push(entry);
                        }
                    }
                }
                for &sym in syms {
                    let entry = Entry {
                        nested: false,
                        bits_or_mask: (idx + 1) as u8,
                        symbol_or_offset: sym,
                    };
                    chunk.push(entry);
                    if chunk.len() == chunk_size {
                        entries[current_bits as usize] = Entry {
                            nested: true,
                            bits_or_mask: (chunk_size - 1) as u8,
                            symbol_or_offset: second_level_entries.len() as u16,
                        };
                        vec_reverse_bits(&chunk, &mut second_level_entries);
                        current_bits += 1;
                        chunk = Vec::with_capacity(chunk_size);
                    }
                }
                remaining_entries = chunk;
                remaining_entry_bits = chunk_size_bits;
            }

            if !remaining_entries.is_empty() {
                return Err(Error::InvalidPrefixHistogram);
            }
        }

        if current_bits == 1 << toplevel_bits {
            let mut toplevel_entries = Vec::with_capacity(entries.len());
            vec_reverse_bits(&entries, &mut toplevel_entries);
            Ok(Self {
                toplevel_bits,
                toplevel_mask: (1 << toplevel_bits) - 1,
                toplevel_entries,
                second_level_entries,
            })
        } else {
            Err(Error::InvalidPrefixHistogram)
        }
    }

    fn with_single_symbol(symbol: u16) -> Self {
        let entry = Entry {
            nested: false,
            bits_or_mask: 0,
            symbol_or_offset: symbol,
        };
        Self {
            toplevel_bits: 0,
            toplevel_mask: 0,
            toplevel_entries: vec![entry],
            second_level_entries: Vec::new(),
        }
    }

    pub fn parse(bitstream: &mut Bitstream, alphabet_size: u32) -> Result<Self> {
        if alphabet_size == 1 {
            return Ok(Self::with_single_symbol(0));
        }

        let hskip = read_bits!(bitstream, u(2))?;
        if hskip == 1 {
            Self::parse_simple(bitstream, alphabet_size)
        } else {
            Self::parse_complex(bitstream, alphabet_size, hskip)
        }
    }

    fn parse_simple(bitstream: &mut Bitstream, alphabet_size: u32) -> Result<Self> {
        let alphabet_bits = alphabet_size.next_power_of_two().trailing_zeros() as usize;
        let nsym = read_bits!(bitstream, u(2))? + 1;
        let it = match nsym {
            1 => {
                let sym = bitstream.read_bits(alphabet_bits)?;
                if sym >= alphabet_size {
                    return Err(Error::InvalidPrefixHistogram);
                }
                return Ok(Self::with_single_symbol(sym as u16));
            }
            2 => {
                let syms = [
                    0,
                    0,
                    bitstream.read_bits(alphabet_bits)? as usize,
                    bitstream.read_bits(alphabet_bits)? as usize,
                ];

                syms.into_iter().zip([0u8, 0, 1u8, 1])
            }
            3 => {
                let syms = [
                    0,
                    bitstream.read_bits(alphabet_bits)? as usize,
                    bitstream.read_bits(alphabet_bits)? as usize,
                    bitstream.read_bits(alphabet_bits)? as usize,
                ];

                syms.into_iter().zip([0u8, 1, 2, 2])
            }
            4 => {
                let syms = [
                    bitstream.read_bits(alphabet_bits)? as usize,
                    bitstream.read_bits(alphabet_bits)? as usize,
                    bitstream.read_bits(alphabet_bits)? as usize,
                    bitstream.read_bits(alphabet_bits)? as usize,
                ];
                let tree_selector = bitstream.read_bool()?;

                if tree_selector {
                    syms.into_iter().zip([1u8, 2, 3, 3])
                } else {
                    syms.into_iter().zip([2u8, 2, 2, 2])
                }
            }
            _ => unreachable!(),
        };

        let mut code_lengths = vec![0u8; alphabet_size as usize];
        for (sym, len) in it {
            if let Some(out) = code_lengths.get_mut(sym) {
                *out = len;
            } else {
                return Err(Error::InvalidPrefixHistogram);
            }
        }
        Self::with_code_lengths(code_lengths)
    }

    fn parse_complex(bitstream: &mut Bitstream, alphabet_size: u32, hskip: u32) -> Result<Self> {
        const CODE_LENGTH_ORDER: [usize; 18] =
            [1, 2, 3, 4, 0, 5, 17, 6, 16, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let mut code_length_code_lengths = [0u8; 18];
        let mut bitacc = 0usize;

        let mut nonzero_count = 0;
        let mut nonzero_sym = 0;
        for idx in CODE_LENGTH_ORDER.into_iter().skip(hskip as usize) {
            // Read single code length code.
            let base = read_bits!(bitstream, U32(0, 4, 3, 8))?;
            let len = if base == 8 {
                if read_bits!(bitstream, Bool)? {
                    if read_bits!(bitstream, Bool)? {
                        // 1111
                        5
                    } else {
                        // 0111
                        1
                    }
                } else {
                    // 011
                    2
                }
            } else {
                base
            };

            code_length_code_lengths[idx] = len;
            if len != 0 {
                nonzero_count += 1;
                nonzero_sym = idx;
                bitacc += 32 >> len;

                match bitacc.cmp(&32) {
                    std::cmp::Ordering::Less => {}
                    std::cmp::Ordering::Equal => break,
                    std::cmp::Ordering::Greater => return Err(Error::InvalidPrefixHistogram),
                }
            }
        }

        let code_length_histogram = if nonzero_count == 1 {
            Histogram::with_single_symbol(nonzero_sym as u16)
        } else if bitacc != 32 {
            return Err(Error::InvalidPrefixHistogram);
        } else {
            Histogram::with_code_lengths(code_length_code_lengths.to_vec())?
        };

        let mut code_lengths = vec![0u8; alphabet_size as usize];
        let mut bitacc = 0usize;

        let mut prev_sym = 8u8;
        let mut last_nonzero_sym = 8u8;
        let mut last_repeat_count = 0usize;

        let mut repeat_count = 0usize;
        let mut repeat_sym = 0u8;
        for len in &mut code_lengths {
            if repeat_count > 0 {
                *len = repeat_sym;
                repeat_count -= 1;
            } else {
                let sym = code_length_histogram.read_symbol(bitstream)? as u8;
                match sym {
                    0 => {}
                    1..=15 => {
                        *len = sym;
                        last_nonzero_sym = sym;
                    }
                    16 => {
                        repeat_count = bitstream.peek_bits_prefilled(2) as usize + 3;
                        bitstream.consume_bits(2)?;
                        if prev_sym == 16 {
                            repeat_count += last_repeat_count * 3 - 8;
                            last_repeat_count += repeat_count;
                        } else {
                            last_repeat_count = repeat_count;
                        }
                        repeat_sym = last_nonzero_sym;

                        *len = repeat_sym;
                        repeat_count -= 1;
                    }
                    17 => {
                        repeat_count = bitstream.peek_bits_prefilled(3) as usize + 3;
                        bitstream.consume_bits(3)?;
                        if prev_sym == 17 {
                            repeat_count += last_repeat_count * 7 - 16;
                            last_repeat_count += repeat_count;
                        } else {
                            last_repeat_count = repeat_count;
                        }
                        repeat_sym = 0;

                        *len = repeat_sym;
                        repeat_count -= 1;
                    }
                    _ => unreachable!(),
                }
                prev_sym = sym;
            }

            if *len != 0 {
                bitacc += 32768 >> *len;

                if bitacc > 32768 {
                    return Err(Error::InvalidPrefixHistogram);
                } else if bitacc == 32768 && repeat_count == 0 {
                    break;
                }
            }
        }

        if bitacc != 32768 || repeat_count > 0 {
            return Err(Error::InvalidPrefixHistogram);
        }
        Self::with_code_lengths(code_lengths)
    }
}

impl Histogram {
    #[inline(always)]
    pub fn read_symbol(&self, bitstream: &mut Bitstream) -> Result<u32> {
        let Self {
            toplevel_bits,
            toplevel_mask,
            ref toplevel_entries,
            ref second_level_entries,
        } = *self;
        let mut peeked = bitstream.peek_bits_const::<15>();
        let toplevel_offset = peeked & toplevel_mask;
        let toplevel_entry = toplevel_entries[toplevel_offset as usize];
        peeked >>= toplevel_bits;
        if toplevel_entry.nested {
            let chunk_offset = peeked & (toplevel_entry.bits_or_mask as u32);
            let second_level_offset = toplevel_entry.symbol_or_offset as u32 + chunk_offset;
            let second_level_entry = second_level_entries[second_level_offset as usize];
            bitstream.consume_bits(second_level_entry.bits_or_mask as usize)?;
            Ok(second_level_entry.symbol_or_offset as u32)
        } else {
            bitstream.consume_bits(toplevel_entry.bits_or_mask as usize)?;
            Ok(toplevel_entry.symbol_or_offset as u32)
        }
    }

    #[inline]
    pub fn single_symbol(&self) -> Option<u32> {
        if let &[Entry {
            nested: false,
            bits_or_mask: 0,
            symbol_or_offset: symbol,
        }] = &*self.toplevel_entries
        {
            Some(symbol as u32)
        } else {
            None
        }
    }
}

fn vec_reverse_bits(v: &[Entry], out: &mut Vec<Entry>) {
    let len = v.len();
    debug_assert!(len.is_power_of_two());
    let bits = len.trailing_zeros();
    let shift = usize::BITS - bits;
    for idx in 0..len {
        let rev_idx = idx.reverse_bits() >> shift;
        let entry = v[rev_idx];
        out.push(entry);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn type_size() {
        assert_eq!(std::mem::size_of::<super::Entry>(), 4);
    }
}
