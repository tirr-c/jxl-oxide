use jxl_bitstream::{Bitstream, U};
use jxl_oxide_common::Bundle;

#[derive(Debug)]
pub(crate) struct HuffmanCode {
    pub(crate) is_ac: bool,
    pub(crate) id: u8,
    pub(crate) is_last: bool,
    pub(crate) counts: [u8; 17],
    pub(crate) values: Vec<u8>,
}

impl HuffmanCode {
    pub fn encoded_len(&self) -> usize {
        1 + 16 + self.values.len() - 1
    }

    pub fn build(&self) -> BuiltHuffmanTable {
        let counts = &self.counts;
        let values = &self.values;

        let mut lengths = vec![0u8; values.len()];
        let mut next_lengths = &mut *lengths;
        for (len, &count) in counts.iter().enumerate() {
            let len = len as u8;
            let count = count as usize;

            let (buf, next) = next_lengths.split_at_mut(count);
            buf.fill(len);
            next_lengths = next;
        }
        lengths.pop();

        let mut bits = Vec::with_capacity(values.len());
        let mut next_code = 0u64;
        let mut prev_len = lengths[0];
        for &len in &lengths {
            let shift_len = 64 - len;
            if len != prev_len {
                next_code <<= 1;
                prev_len = len;
            }
            bits.push(next_code << shift_len);
            next_code += 1;
        }

        let mut reordered_lengths = vec![0u8; 256];
        let mut reordered_bits = vec![0u64; 256];
        for (&value, (length, bit)) in values.iter().zip(std::iter::zip(lengths, bits)) {
            let idx = value as usize;
            reordered_lengths[idx] = length;
            reordered_bits[idx] = bit;
        }

        BuiltHuffmanTable {
            lengths: reordered_lengths,
            bits: reordered_bits,
        }
    }
}

impl Bundle for HuffmanCode {
    type Error = jxl_bitstream::Error;

    fn parse(bitstream: &mut Bitstream, _: ()) -> Result<Self, Self::Error> {
        let is_ac = bitstream.read_bool()?;
        let id = bitstream.read_bits(2)? as u8;
        let is_last = bitstream.read_bool()?;

        let mut counts = [0u8; 17];
        let mut sum_counts = 0u32;
        for count in &mut counts {
            let x = bitstream.read_u32(0, 1, 2 + U(3), U(8))?;
            sum_counts += x;
            *count = x as u8;
        }
        let values = (0..sum_counts)
            .map(|_| {
                bitstream
                    .read_u32(U(2), 4 + U(2), 8 + U(4), 1 + U(8))
                    .map(|x| x as u8)
            })
            .collect::<Result<_, _>>()?;

        Ok(Self {
            is_ac,
            id,
            is_last,
            counts,
            values,
        })
    }
}

pub(crate) struct BuiltHuffmanTable {
    lengths: Vec<u8>,
    bits: Vec<u64>,
}

impl std::fmt::Debug for BuiltHuffmanTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        struct Helper<'a>(&'a BuiltHuffmanTable);
        impl std::fmt::Debug for Helper<'_> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                let entries = std::iter::zip(&self.0.lengths, &self.0.bits).enumerate().filter_map(|(sym, (&len, &bit))| {
                    if len == 0 {
                        return None;
                    }

                    let bit = bit >> (64 - len);
                    let bits_str = format!("{bit:b}");
                    let pad = len as usize - bits_str.len();
                    let mut s = "0".repeat(pad);
                    s.push_str(&bits_str);
                    Some((sym, s))
                });

                f.debug_map().entries(entries).finish()
            }
        }

        f.debug_struct("BulitHuffmanTable").field("mapping", &Helper(self)).finish()
    }
}

impl BuiltHuffmanTable {
    pub fn lookup(&self, symbol: u8) -> (u8, u64) {
        let idx = symbol as usize;
        debug_assert_ne!(self.lengths[idx], 0);
        (self.lengths[idx], self.bits[idx])
    }
}
