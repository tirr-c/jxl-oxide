//! Prefix code based on Brotli
use jxl_bitstream::{read_bits, Bitstream};

use crate::{Error, Result};

#[derive(Debug)]
pub struct Histogram {
    toplevel_entries: Box<[Entry]>,
    second_level_entries: Box<[LeafEntry]>,
}

#[derive(Debug, Copy, Clone, Default)]
struct LeafEntry {
    bits: u32,
    symbol: u32,
}

#[derive(Debug, Copy, Clone)]
enum Entry {
    Leaf { bits: u32, symbol: u32 },
    Nested { mask: u32, offset: u32 },
}

impl Default for Entry {
    fn default() -> Self {
        Self::Leaf { bits: 0, symbol: 0 }
    }
}

impl Histogram {
    fn with_code_lengths(code_lengths: Vec<u8>) -> Result<Self> {
        let mut syms_for_length = Vec::with_capacity(15);
        for (sym, len) in code_lengths.into_iter().enumerate() {
            let sym = sym as u32;
            if len > 0 {
                if syms_for_length.len() < len as usize {
                    syms_for_length.resize_with(len as usize, Vec::new);
                }
                syms_for_length[len as usize - 1].push(sym);
            }
        }

        let toplevel_bits = {
            let mut numer = 0usize;
            let mut denom = 1usize;
            for syms in &syms_for_length {
                numer = numer * 2 + syms.len();
                denom *= 2;
                if numer * 100 >= denom * 99 {
                    break;
                }
            }
            denom.trailing_zeros().max(4) as usize
        };

        let mut entries = vec![Entry::default(); 1 << toplevel_bits];
        let mut current_bits = 0u16;
        for (idx, syms) in syms_for_length.iter().enumerate().take(toplevel_bits) {
            let shifts = toplevel_bits - 1 - idx;
            for &sym in syms {
                let entry = Entry::Leaf {
                    bits: (idx + 1) as u32,
                    symbol: sym,
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
                    let entry = LeafEntry {
                        bits: (idx + 1) as u32,
                        symbol: sym,
                    };
                    chunk.push(entry);
                    if chunk.len() == chunk_size {
                        entries[current_bits as usize] = Entry::Nested {
                            mask: (chunk_size - 1) as u32,
                            offset: second_level_entries.len() as u32,
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
                toplevel_entries: toplevel_entries.into_boxed_slice(),
                second_level_entries: second_level_entries.into_boxed_slice(),
            })
        } else {
            Err(Error::InvalidPrefixHistogram)
        }
    }

    fn with_single_symbol(symbol: u32) -> Self {
        let entry = Entry::Leaf { bits: 0, symbol };
        Self {
            toplevel_entries: vec![entry].into_boxed_slice(),
            second_level_entries: Vec::new().into_boxed_slice(),
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
        let alphabet_bits = alphabet_size.next_power_of_two().trailing_zeros();
        let nsym = read_bits!(bitstream, u(2))? + 1;
        let it = match nsym {
            1 => {
                let sym = bitstream.read_bits(alphabet_bits)?;
                if sym >= alphabet_size {
                    return Err(Error::InvalidPrefixHistogram);
                }
                return Ok(Self::with_single_symbol(sym));
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
            Histogram::with_single_symbol(nonzero_sym as u32)
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
                let sym = code_length_histogram.read_symbol(bitstream) as u8;
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

        bitstream.check_in_bounds()?;
        if bitacc != 32768 || repeat_count > 0 {
            return Err(Error::InvalidPrefixHistogram);
        }
        Self::with_code_lengths(code_lengths)
    }
}

impl Histogram {
    #[inline(always)]
    pub fn read_symbol(&self, bitstream: &mut Bitstream) -> u32 {
        let Self {
            toplevel_entries,
            second_level_entries,
        } = self;
        let peeked = bitstream.bit_buffer_u32();
        debug_assert!(toplevel_entries.len().is_power_of_two());
        let toplevel_mask = toplevel_entries.len() - 1;
        let toplevel_offset = peeked as usize & toplevel_mask;
        let toplevel_entry = unsafe { *toplevel_entries.get_unchecked(toplevel_offset) };
        let (bits, symbol) = match toplevel_entry {
            Entry::Leaf { bits, symbol } => (bits, symbol),
            Entry::Nested { mask, offset } => {
                let toplevel_bits = toplevel_mask.trailing_ones();
                let chunk_offset = (peeked >> toplevel_bits) & mask;
                let second_level_offset = offset + chunk_offset;
                let entry = second_level_entries[second_level_offset as usize];
                (entry.bits, entry.symbol)
            }
        };

        bitstream.consume_bits_silent(bits);
        symbol
    }

    #[inline]
    pub fn single_symbol(&self) -> Option<u32> {
        if let &[Entry::Leaf { bits: 0, symbol }] = &*self.toplevel_entries {
            Some(symbol)
        } else {
            None
        }
    }
}

fn vec_reverse_bits<T: Copy>(v: &[T], out: &mut Vec<T>) {
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
