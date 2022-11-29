//! Prefix code based on Brotli
use std::collections::BTreeMap;
use std::io::Read;

use jxl_bitstream::{Bitstream, read_bits};

use crate::{Error, Result};

#[derive(Debug)]
pub struct Histogram {
    instr: TreeInstr,
}

#[derive(Debug)]
enum TreeInstr {
    Leaf(u16),
    Read {
        bits: u8,
        children: Vec<TreeInstr>,
    },
}

impl Histogram {
    fn with_code_lengths(code_lengths: Vec<u8>) -> Result<Self> {
        let mut syms_for_length = BTreeMap::new();
        for (sym, len) in code_lengths.into_iter().enumerate() {
            let sym = sym as u16;
            if len > 0 {
                syms_for_length
                    .entry(len)
                    .or_insert_with(Vec::new)
                    .push(sym);
            }
        }

        let it = syms_for_length
            .into_iter()
            .flat_map(|(len, syms)| {
                syms.into_iter().map(move |sym| (len, sym))
            });

        let mut node_stack = Vec::new();
        let mut current_len = 0u8;
        let mut current_children = Vec::new();
        let mut target_bits = 0u8;
        for (len, sym) in it {
            if current_len == len {
                current_children.push(TreeInstr::Leaf(sym));
                if current_children.len() == (1 << target_bits) {
                    let node = TreeInstr::Read {
                        bits: target_bits,
                        children: current_children,
                    };
                    let (parent_len, parent_children) = node_stack
                        .pop()
                        .ok_or(Error::InvalidPrefixHistogram)?;
                    current_len = parent_len;
                    current_children = parent_children;
                    current_children.push(node);
                }
            } else {
                target_bits = len - current_len;
                node_stack.push((current_len, current_children));
                current_children = vec![TreeInstr::Leaf(sym)];
                current_len = len;
            }
        }

        if !node_stack.is_empty() {
            Err(Error::InvalidPrefixHistogram)
        } else if let Some(instr) = current_children.pop() {
            if current_children.is_empty() {
                Ok(Self { instr })
            } else {
                Err(Error::InvalidPrefixHistogram)
            }
        } else {
            Err(Error::InvalidPrefixHistogram)
        }
    }

    fn with_single_symbol(symbol: u16) -> Self {
        Self {
            instr: TreeInstr::Leaf(symbol),
        }
    }

    pub fn parse<R: Read>(bitstream: &mut Bitstream<R>, alphabet_size: u32) -> Result<Self> {
        let hskip = read_bits!(bitstream, u(2))?;
        if hskip == 1 {
            Self::parse_simple(bitstream, alphabet_size)
        } else {
            Self::parse_complex(bitstream, alphabet_size, hskip)
        }
    }

    fn parse_simple<R: Read>(bitstream: &mut Bitstream<R>, alphabet_size: u32) -> Result<Self> {
        let alphabet_bits = (alphabet_size - 1).next_power_of_two().trailing_zeros();
        let nsym = read_bits!(bitstream, u(2))? + 1;
        let it = match nsym {
            1 => {
                let sym = bitstream.read_bits(alphabet_bits)?;
                if sym >= alphabet_size {
                    return Err(Error::InvalidPrefixHistogram);
                }
                return Ok(Self::with_single_symbol(sym as u16));
            },
            2 => {
                let syms = [
                    0,
                    0,
                    bitstream.read_bits(alphabet_bits)? as usize,
                    bitstream.read_bits(alphabet_bits)? as usize,
                ];

                syms.into_iter().zip([0u8, 0, 1u8, 1])
            },
            3 => {
                let syms = [
                    0,
                    bitstream.read_bits(alphabet_bits)? as usize,
                    bitstream.read_bits(alphabet_bits)? as usize,
                    bitstream.read_bits(alphabet_bits)? as usize,
                ];

                syms.into_iter().zip([0u8, 1, 2, 2])
            },
            4 => {
                let syms = [
                    bitstream.read_bits(alphabet_bits)? as usize,
                    bitstream.read_bits(alphabet_bits)? as usize,
                    bitstream.read_bits(alphabet_bits)? as usize,
                    bitstream.read_bits(alphabet_bits)? as usize,
                ];
                let tree_selector = read_bits!(bitstream, Bool)?;

                if tree_selector {
                    syms.into_iter().zip([2u8, 2, 2, 2])
                } else {
                    syms.into_iter().zip([1u8, 2, 3, 3])
                }
            },
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

    fn parse_complex<R: Read>(bitstream: &mut Bitstream<R>, alphabet_size: u32, hskip: u32) -> Result<Self> {
        const CODE_LENGTH_ORDER: [usize; 18] = [
            1, 2, 3, 4, 0, 5,
            17, 6, 16, 7, 8, 9,
            10, 11, 12, 13, 14, 15,
        ];
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
                    std::cmp::Ordering::Less => {},
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
                        repeat_count = bitstream.read_bits(2)? as usize + 3;
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
                        repeat_count = bitstream.read_bits(3)? as usize + 3;
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
                    _ => unreachable!()
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
    #[inline]
    pub fn read_symbol<R: Read>(&self, bitstream: &mut Bitstream<R>) -> Result<u16> {
        self.instr.read_symbol(bitstream)
    }
}

impl TreeInstr {
    fn read_symbol<R: Read>(&self, bitstream: &mut Bitstream<R>) -> Result<u16> {
        match self {
            Self::Leaf(sym) => Ok(*sym),
            Self::Read { bits, children } => {
                let selector = bitstream.read_bits(*bits as u32)? as usize;
                children[selector].read_symbol(bitstream)
            },
        }
    }
}

impl Histogram {
    pub(crate) fn prefix_code_for_ans() -> Self {
        use TreeInstr::*;

        let instr = Read {
            bits: 3,
            children: vec![
                Leaf(10),
                Leaf(6),
                Leaf(7),
                Leaf(9),
                Read {
                    bits: 1,
                    children: vec![
                        Read {
                            bits: 1,
                            children: vec![
                                Read {
                                    bits: 1,
                                    children: vec![
                                        Read {
                                            bits: 1,
                                            children: vec![
                                                Leaf(12),
                                                Leaf(13),
                                            ],
                                        },
                                        Leaf(11),
                                    ],
                                },
                                Leaf(0),
                            ],
                        },
                        Leaf(4),
                    ],
                },
                Leaf(8),
                Read {
                    bits: 1,
                    children: vec![
                        Leaf(3),
                        Leaf(1),
                    ],
                },
                Read {
                    bits: 1,
                    children: vec![
                        Leaf(5),
                        Leaf(2),
                    ],
                },
            ],
        };
        Self { instr }
    }
}
