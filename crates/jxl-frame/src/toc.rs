use std::io::Read;
use jxl_bitstream::{
    read_bits,
    Bitstream,
    Bundle,
};
use crate::Result;

#[derive(Debug)]
pub struct Toc {
    bookmark: jxl_bitstream::Bookmark,
    num_lf_groups: usize,
    num_groups: usize,
    has_hf_global: bool,
    offsets: Vec<u64>,
    sizes: Vec<u32>,
    total_size: u64,
}

impl Toc {
    pub fn bookmark(&self) -> jxl_bitstream::Bookmark {
        self.bookmark
    }

    pub fn is_single_entry(&self) -> bool {
        self.offsets.len() <= 1
    }

    pub fn lf_global_byte_offset(&self) -> (u64, u32) {
        (self.offsets[0], self.sizes[0])
    }

    pub fn lf_group_byte_offset(&self, idx: u32) -> (u64, u32) {
        if self.is_single_entry() {
            panic!("cannot obtain LfGroup offset of single entry frame");
        } else if (idx as usize) >= self.num_lf_groups {
            panic!("index out of range: {} >= {} (num_lf_groups)", idx, self.num_lf_groups);
        } else {
            let idx = idx as usize + 1;
            (self.offsets[idx], self.sizes[idx])
        }
    }

    pub fn hf_global_byte_offset(&self) -> (u64, u32) {
        if self.has_hf_global {
            let idx = self.num_lf_groups + 1;
            (self.offsets[idx], self.sizes[idx])
        } else {
            panic!("this frame does not have HfGlobal offset");
        }
    }

    pub fn pass_group_byte_offset(&self, pass_idx: u32, group_idx: u32) -> (u64, u32) {
        if self.is_single_entry() {
            panic!("cannot obtain PassGroup offset of single entry frame");
        } else {
            let mut idx = 1 + self.num_lf_groups;
            if self.has_hf_global {
                idx += 1;
            }
            idx += (pass_idx as usize * self.num_groups) + group_idx as usize;
            (self.offsets[idx], self.sizes[idx])
        }
    }

    pub fn total_byte_size(&self) -> u64 {
        self.total_size
    }
}

impl Bundle<&crate::FrameHeader> for Toc {
    type Error = crate::Error;

    fn parse<R: Read>(bitstream: &mut Bitstream<R>, ctx: &crate::FrameHeader) -> Result<Self> {
        let num_groups = ctx.num_groups();
        let num_passes = ctx.passes.num_passes;

        let entry_count = if num_groups == 1 && num_passes == 1 {
            1
        } else {
            1 + ctx.num_lf_groups() + 1 + num_groups * num_passes
        };

        let permutated_toc = bitstream.read_bool()?;
        let permutation = if permutated_toc {
            let mut decoder = jxl_coding::Decoder::parse(bitstream, 8)?;
            let end = decoder.read_varint(bitstream, get_context(entry_count))?;

            let mut lehmer = vec![0u32; end as usize];
            let mut prev_val = 0u32;
            for val in &mut lehmer {
                *val = decoder.read_varint(bitstream, get_context(prev_val))?;
                prev_val = *val;
            }

            let mut temp = (0..(entry_count as usize)).collect::<Vec<_>>();
            let mut permutation = Vec::with_capacity(entry_count as usize);
            for idx in lehmer {
                let idx = idx as usize;
                if idx >= temp.len() {
                    return Err(crate::Error::InvalidTocPermutation);
                }
                permutation.push(temp.remove(idx));
            }

            permutation
        } else {
            Vec::new()
        };

        bitstream.zero_pad_to_byte()?;
        let sizes = (0..entry_count)
            .map(|_| read_bits!(bitstream, U32(u(10), 1024 + u(14), 17408 + u(22), 4211712 + u(30))))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let mut offsets = Vec::with_capacity(sizes.len());
        let mut acc = 0u64;
        for &size in &sizes {
            offsets.push(acc);
            acc += size as u64;
        }
        let (sizes, offsets) = if permutated_toc {
            let mut new_sizes = Vec::with_capacity(sizes.len());
            let mut new_offsets = Vec::with_capacity(offsets.len());
            for idx in permutation {
                new_sizes.push(sizes[idx]);
                new_offsets.push(offsets[idx]);
            }
            (new_sizes, new_offsets)
        } else {
            (sizes, offsets)
        };

        bitstream.zero_pad_to_byte()?;
        let bookmark = bitstream.bookmark();
        Ok(Self {
            bookmark,
            num_lf_groups: ctx.num_lf_groups() as usize,
            num_groups: num_groups as usize,
            has_hf_global: entry_count > 1 && ctx.encoding == crate::header::Encoding::VarDct,
            sizes,
            offsets,
            total_size: acc,
        })
    }
}

fn get_context(x: u32) -> u32 {
    add_log2_ceil(x).min(7)
}

fn add_log2_ceil(x: u32) -> u32 {
    (x + 1).next_power_of_two().trailing_zeros()
}
