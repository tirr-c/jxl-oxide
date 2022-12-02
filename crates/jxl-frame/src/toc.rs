use std::io::Read;
use jxl_bitstream::{
    read_bits,
    Bitstream,
    Bundle,
};
use crate::Result;

pub struct Toc {
    bookmark: jxl_bitstream::Bookmark,
    num_lf_groups: usize,
    num_groups: usize,
    has_hf_global: bool,
    offsets: Vec<u64>,
    sizes: Vec<u32>,
    linear_groups: Vec<(TocGroupKind, u32)>,
    total_size: u64,
}

impl std::fmt::Debug for Toc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f
            .debug_struct("Toc")
            .field("bookmark", &self.bookmark)
            .field("num_lf_groups", &self.num_lf_groups)
            .field("num_groups", &self.num_groups)
            .field("has_hf_global", &self.has_hf_global)
            .field("total_size", &self.total_size)
            .field(
                "offsets",
                &format_args!(
                    "({} {})",
                    self.offsets.len(),
                    if self.offsets.len() == 1 { "entry" } else { "entries" },
                ),
            )
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum TocGroupKind {
    All,
    LfGlobal,
    LfGroup(u32),
    HfGlobal,
    GroupPass {
        pass_idx: u32,
        group_idx: u32,
    },
}

impl Toc {
    pub fn bookmark(&self) -> jxl_bitstream::Bookmark {
        self.bookmark
    }

    pub fn is_single_entry(&self) -> bool {
        self.offsets.len() <= 1
    }

    pub fn lf_global_byte_offset(&self) -> (u64, u32) {
        if self.is_single_entry() {
            (0, self.total_size as u32)
        } else {
            (self.offsets[0], self.sizes[0])
        }
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
        let has_hf_global = ctx.encoding == crate::header::Encoding::VarDct;

        let entry_count = if num_groups == 1 && num_passes == 1 {
            1
        } else {
            1 + ctx.num_lf_groups() + (has_hf_global as u32) + num_groups * num_passes
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

        let section_kinds = if entry_count == 1 {
            vec![TocGroupKind::All]
        } else {
            let mut out = Vec::with_capacity(entry_count as usize);
            out.push(TocGroupKind::LfGlobal);
            for idx in 0..ctx.num_lf_groups() {
                out.push(TocGroupKind::LfGroup(idx));
            }
            if has_hf_global {
                out.push(TocGroupKind::HfGlobal);
            }
            for pass_idx in 0..num_passes {
                for group_idx in 0..num_groups {
                    out.push(TocGroupKind::GroupPass { pass_idx, group_idx });
                }
            }
            out
        };

        let (sizes, offsets, linear_groups) = if permutated_toc {
            let mut new_sizes = Vec::with_capacity(sizes.len());
            let mut new_offsets = Vec::with_capacity(offsets.len());
            let mut new_section_kinds = vec![TocGroupKind::All; section_kinds.len()];
            for (section_kind, idx) in section_kinds.into_iter().zip(permutation) {
                new_sizes.push(sizes[idx]);
                new_offsets.push(offsets[idx]);
                new_section_kinds[idx] = section_kind;
            }
            let linear_groups = new_section_kinds.into_iter().zip(sizes.iter().copied()).collect();
            (new_sizes, new_offsets, linear_groups)
        } else {
            let linear_groups = section_kinds.into_iter().zip(sizes.iter().copied()).collect();
            (sizes, offsets, linear_groups)
        };

        bitstream.zero_pad_to_byte()?;
        let bookmark = bitstream.bookmark();
        Ok(Self {
            bookmark,
            num_lf_groups: ctx.num_lf_groups() as usize,
            num_groups: num_groups as usize,
            has_hf_global: entry_count > 1 && has_hf_global,
            sizes,
            offsets,
            linear_groups,
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
