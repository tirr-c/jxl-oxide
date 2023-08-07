use std::io::Read;
use jxl_bitstream::{
    read_bits,
    Bitstream,
    Bookmark,
    Bundle,
};
use crate::Result;

/// Table of contents of a frame.
///
/// Frame data are organized in groups. TOC specified the size and order of each group, and it is
/// decoded after the frame header.
pub struct Toc {
    num_lf_groups: usize,
    num_groups: usize,
    groups: Vec<TocGroup>,
    bitstream_to_original: Vec<usize>,
    original_to_bitstream: Vec<usize>,
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
                    if self.bitstream_to_original.is_empty() { "empty" } else { "non-empty" },
                ),
            )
            .finish_non_exhaustive()
    }
}

/// Information about a group in TOC.
#[derive(Debug, Copy, Clone)]
pub struct TocGroup {
    /// Kind of the group.
    pub kind: TocGroupKind,
    /// Offset within the bitstream.
    pub offset: Bookmark,
    /// Size of the group.
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
    /// Returns the offset to the beginning of the data.
    pub fn bookmark(&self) -> Bookmark {
        let idx = self.bitstream_to_original.first().copied().unwrap_or(0);
        self.groups[idx].offset
    }

    /// Returns whether the frame has only one group.
    pub fn is_single_entry(&self) -> bool {
        self.groups.len() <= 1
    }

    pub fn group_index_bitstream_order(&self, kind: TocGroupKind) -> usize {
        let original_order = match kind {
            TocGroupKind::All if self.is_single_entry() => 0,
            _ if self.is_single_entry() => panic!("Cannot request group type of {:?} for single-group frame", kind),
            TocGroupKind::All => panic!("Cannot request group type of All for multi-group frame"),
            TocGroupKind::LfGlobal => 0,
            TocGroupKind::LfGroup(lf_group_idx) => 1 + lf_group_idx as usize,
            TocGroupKind::HfGlobal => 1 + self.num_lf_groups,
            TocGroupKind::GroupPass { pass_idx, group_idx } =>
                1 + self.num_lf_groups + 1 + pass_idx as usize * self.num_groups + group_idx as usize,
        };

        if self.original_to_bitstream.is_empty() {
            original_order
        } else {
            self.original_to_bitstream[original_order]
        }
    }

    /// Returns the total size of the frame data in bytes.
    pub fn total_byte_size(&self) -> u64 {
        self.total_size
    }

    pub fn iter_bitstream_order(&self) -> impl Iterator<Item = TocGroup> + Send {
        let groups = if self.bitstream_to_original.is_empty() {
            self.groups.clone()
        } else {
            self.bitstream_to_original.iter().map(|&idx| self.groups[idx]).collect()
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

        if entry_count > 65536 {
            return Err(jxl_bitstream::Error::ValidationFailed(
                "Too many TOC entries"
            ).into());
        }

        let permutated_toc = bitstream.read_bool()?;
        let permutation = if permutated_toc {
            let mut decoder = jxl_coding::Decoder::parse(bitstream, 8)?;
            decoder.begin(bitstream)?;
            let permutation = jxl_coding::read_permutation(bitstream, &mut decoder, entry_count, 0)?;
            decoder.finalize()?;
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

        let (offsets, sizes, bitstream_to_original, original_to_bitstream) = if permutated_toc {
            let mut bitstream_to_original = vec![0usize; permutation.len()];
            let mut offsets_out = Vec::with_capacity(permutation.len());
            let mut sizes_out = Vec::with_capacity(permutation.len());
            for (idx, &perm) in permutation.iter().enumerate() {
                offsets_out.push(offsets[perm]);
                sizes_out.push(sizes[perm]);
                bitstream_to_original[perm] = idx;
            }
            (offsets_out, sizes_out, bitstream_to_original, permutation)
        } else {
            (offsets, sizes, Vec::new(), Vec::new())
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

        Ok(Self {
            num_lf_groups: ctx.num_lf_groups() as usize,
            num_groups: num_groups as usize,
            groups,
            bitstream_to_original,
            original_to_bitstream,
            total_size,
        })
    }
}
