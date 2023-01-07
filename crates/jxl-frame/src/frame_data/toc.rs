use std::io::Read;
use jxl_bitstream::{
    read_bits,
    Bitstream,
    Bookmark,
    Bundle,
};
use crate::Result;

pub struct Toc {
    num_lf_groups: usize,
    num_groups: usize,
    groups: Vec<TocGroup>,
    bitstream_order: Vec<usize>,
    total_size: u64,
}

impl std::fmt::Debug for Toc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f
            .debug_struct("Toc")
            .field("num_lf_groups", &self.num_lf_groups)
            .field("num_groups", &self.num_groups)
            .field("total_size", &self.total_size)
            .field(
                "groups",
                &format_args!(
                    "({} {})",
                    self.groups.len(),
                    if self.groups.len() == 1 { "entry" } else { "entries" },
                ),
            )
            .field(
                "bitstream_order",
                &format_args!(
                    "({})",
                    if self.bitstream_order.is_empty() { "empty" } else { "non-empty" },
                ),
            )
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Copy, Clone)]
pub struct TocGroup {
    pub kind: TocGroupKind,
    pub offset: Bookmark,
    pub size: u32,
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

impl Ord for TocGroupKind {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            (x, y) if x == y => std::cmp::Ordering::Equal,
            (Self::All, _) => std::cmp::Ordering::Less,
            (_, Self::All) => std::cmp::Ordering::Greater,
            (Self::LfGlobal, _) => std::cmp::Ordering::Less,
            (_, Self::LfGlobal) => std::cmp::Ordering::Greater,
            (Self::LfGroup(g_self), Self::LfGroup(g_other)) => g_self.cmp(g_other),
            (Self::LfGroup(_), _) => std::cmp::Ordering::Less,
            (_, Self::LfGroup(_)) => std::cmp::Ordering::Greater,
            (Self::HfGlobal, _) => std::cmp::Ordering::Less,
            (_, Self::HfGlobal) => std::cmp::Ordering::Greater,
            (Self::GroupPass { pass_idx: p_self, group_idx: g_self },
             Self::GroupPass { pass_idx: p_other, group_idx: g_other }) =>
                p_self.cmp(p_other).then(g_self.cmp(g_other))
        }
    }
}

impl PartialOrd for TocGroupKind {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Toc {
    pub fn bookmark(&self) -> Bookmark {
        self.groups[0].offset
    }

    pub fn is_single_entry(&self) -> bool {
        self.groups.len() <= 1
    }

    fn group(&self, idx: usize) -> TocGroup {
        let idx = if self.bitstream_order.is_empty() {
            idx
        } else {
            self.bitstream_order[idx]
        };
        self.groups[idx]
    }

    pub fn lf_global(&self) -> TocGroup {
        self.group(0)
    }

    pub fn lf_group(&self, idx: u32) -> TocGroup {
        if self.is_single_entry() {
            panic!("cannot obtain LfGroup offset of single entry frame");
        } else if (idx as usize) >= self.num_lf_groups {
            panic!("index out of range: {} >= {} (num_lf_groups)", idx, self.num_lf_groups);
        } else {
            self.group(idx as usize + 1)
        }
    }

    pub fn hf_global(&self) -> TocGroup {
        self.group(self.num_lf_groups + 1)
    }

    pub fn pass_group(&self, pass_idx: u32, group_idx: u32) -> TocGroup {
        if self.is_single_entry() {
            panic!("cannot obtain PassGroup offset of single entry frame");
        } else {
            let mut idx = 1 + self.num_lf_groups + 1;
            idx += (pass_idx as usize * self.num_groups) + group_idx as usize;
            self.group(idx)
        }
    }

    pub fn total_byte_size(&self) -> u64 {
        self.total_size
    }

    pub fn iter_bitstream_order(&self) -> impl Iterator<Item = TocGroup> {
        let groups = if self.bitstream_order.is_empty() {
            self.groups.clone()
        } else {
            self.bitstream_order.iter().map(|&idx| self.groups[idx]).collect()
        };
        groups.into_iter()
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
        bitstream.zero_pad_to_byte()?;

        let mut offsets = Vec::with_capacity(sizes.len());
        let mut acc = bitstream.bookmark();
        let mut total_size = 0u64;
        for &size in &sizes {
            offsets.push(acc);
            acc += size as u64 * 8;
            total_size += size as u64;
        }

        let section_kinds = if entry_count == 1 {
            vec![TocGroupKind::All]
        } else {
            let mut out = Vec::with_capacity(entry_count as usize);
            out.push(TocGroupKind::LfGlobal);
            for idx in 0..ctx.num_lf_groups() {
                out.push(TocGroupKind::LfGroup(idx));
            }
            out.push(TocGroupKind::HfGlobal);
            for pass_idx in 0..num_passes {
                for group_idx in 0..num_groups {
                    out.push(TocGroupKind::GroupPass { pass_idx, group_idx });
                }
            }
            out
        };

        let groups = sizes
            .into_iter()
            .zip(offsets)
            .zip(section_kinds)
            .map(|((size, offset), kind)| TocGroup {
                kind,
                offset,
                size,
            })
            .collect::<Vec<_>>();
        let bitstream_order = if permutated_toc {
            permutation
        } else {
            Vec::new()
        };

        Ok(Self {
            num_lf_groups: ctx.num_lf_groups() as usize,
            num_groups: num_groups as usize,
            groups,
            bitstream_order,
            total_size,
        })
    }
}

fn get_context(x: u32) -> u32 {
    add_log2_ceil(x).min(7)
}

fn add_log2_ceil(x: u32) -> u32 {
    (x + 1).next_power_of_two().trailing_zeros()
}
