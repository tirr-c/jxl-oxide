use std::collections::{HashMap, HashSet};
use std::collections::BTreeMap;
use std::io::Read;

use header::Encoding;
use jxl_bitstream::{read_bits, Bitstream, Bundle};
use jxl_image::Headers;

mod error;
pub mod filter;
pub mod data;
pub mod header;

pub use error::{Error, Result};
pub use header::FrameHeader;
pub use data::Toc;

use crate::data::*;

#[derive(Debug, Copy, Clone)]
enum GroupInstr {
    Read(usize, TocGroup),
    Decode(usize),
}

#[derive(Debug)]
pub struct Frame<'a> {
    image_header: &'a Headers,
    header: FrameHeader,
    toc: Toc,
    plan: Vec<GroupInstr>,
    next_instr: usize,
    buf_slot: HashMap<usize, (TocGroupKind, Vec<u8>)>,
    data: FrameData,
    pass_shifts: BTreeMap<u32, (i32, i32)>,
}

impl<'a> Bundle<&'a Headers> for Frame<'a> {
    type Error = crate::Error;

    fn parse<R: Read>(bitstream: &mut Bitstream<R>, image_header: &'a Headers) -> Result<Self> {
        bitstream.zero_pad_to_byte()?;
        let header = read_bits!(bitstream, Bundle(FrameHeader), image_header)?;
        let toc = read_bits!(bitstream, Bundle(Toc), &header)?;
        let data = FrameData::new(&header);

        let passes = &header.passes;
        let mut pass_shifts = BTreeMap::new();
        let mut maxshift = 3i32;
        for (&downsample, &last_pass) in passes.downsample.iter().zip(&passes.last_pass) {
            let minshift = downsample.trailing_zeros() as i32;
            pass_shifts.insert(last_pass, (minshift, maxshift));
            maxshift = minshift;
        }
        pass_shifts.insert(header.passes.num_passes - 1, (0i32, maxshift));

        let mut plan = Vec::new();
        if toc.is_single_entry() {
            let group = toc.lf_global();
            plan.push(GroupInstr::Read(0, group));
            plan.push(GroupInstr::Decode(0));
        } else {
            let groups = toc.iter_bitstream_order();
            let mut read_slot = HashMap::new();
            let mut decoded_slots = HashSet::new();
            let mut need_lf_global = true;
            let mut need_hf_global = header.encoding == Encoding::VarDct;
            let mut next_slot_idx = 0usize;
            for group in groups {
                if !need_hf_global && group.kind == TocGroupKind::HfGlobal {
                    continue;
                }

                let current_slot_idx = next_slot_idx;
                plan.push(GroupInstr::Read(current_slot_idx, group));
                next_slot_idx += 1;

                let mut update_lf_groups = false;
                let mut update_pass_groups = false;
                match group.kind {
                    TocGroupKind::All => panic!("unexpected TocGroupKind::All"),
                    TocGroupKind::LfGlobal => {
                        plan.push(GroupInstr::Decode(current_slot_idx));
                        decoded_slots.insert(group.kind);
                        update_lf_groups = true;
                        need_lf_global = false;
                    },
                    TocGroupKind::HfGlobal => {
                        if need_lf_global {
                            read_slot.insert(group.kind, current_slot_idx);
                        } else {
                            plan.push(GroupInstr::Decode(current_slot_idx));
                            decoded_slots.insert(group.kind);
                            update_pass_groups = true;
                            need_hf_global = false;
                        }
                    },
                    TocGroupKind::LfGroup(_) => {
                        if need_lf_global {
                            read_slot.insert(group.kind, current_slot_idx);
                        } else {
                            plan.push(GroupInstr::Decode(current_slot_idx));
                            decoded_slots.insert(group.kind);
                            update_pass_groups = true;
                        }
                    },
                    TocGroupKind::GroupPass { group_idx, .. } => {
                        let lf_group_idx = header.lf_group_idx_from_group_idx(group_idx);
                        if need_lf_global || need_hf_global || !decoded_slots.contains(&TocGroupKind::LfGroup(lf_group_idx)) {
                            read_slot.insert(group.kind, current_slot_idx);
                        } else {
                            plan.push(GroupInstr::Decode(current_slot_idx));
                            decoded_slots.insert(group.kind);
                        }
                    },
                }

                if update_lf_groups {
                    let mut decoded = Vec::new();
                    for (&kind, &slot_idx) in &read_slot {
                        if let TocGroupKind::LfGroup(_) = kind {
                            plan.push(GroupInstr::Decode(slot_idx));
                            decoded.push(kind);
                        }
                    }
                    for kind in decoded {
                        read_slot.remove(&kind);
                        decoded_slots.insert(kind);
                    }
                    update_pass_groups = true;
                }

                if update_pass_groups && !need_hf_global {
                    let mut decoded = Vec::new();
                    for (&kind, &slot_idx) in &read_slot {
                        if let TocGroupKind::GroupPass { group_idx, .. } = kind {
                            let lf_group_idx = header.lf_group_idx_from_group_idx(group_idx);
                            if decoded_slots.contains(&TocGroupKind::LfGroup(lf_group_idx)) {
                                plan.push(GroupInstr::Decode(slot_idx));
                                decoded.push(kind);
                            }
                        }
                    }
                    for kind in decoded {
                        read_slot.remove(&kind);
                        decoded_slots.insert(kind);
                    }
                }
            }
        }

        Ok(Self {
            image_header,
            header,
            toc,
            plan,
            next_instr: 0,
            buf_slot: HashMap::new(),
            data,
            pass_shifts,
        })
    }
}

impl Frame<'_> {
    pub fn header(&self) -> &FrameHeader {
        &self.header
    }

    pub fn toc(&self) -> &Toc {
        &self.toc
    }

    pub fn data(&self) -> &FrameData {
        &self.data
    }
}

impl Frame<'_> {
    pub fn load_progressive<R: Read>(
        &mut self,
        bitstream: &mut Bitstream<R>,
        filter_fn: impl FnMut(&FrameHeader, &FrameData, TocGroupKind) -> bool,
    ) -> Result<bool> {
        let span = tracing::span!(tracing::Level::TRACE, "Frame::load_progressive");
        let _guard = span.enter();

        let result = self.load_with_filter(bitstream, filter_fn);
        match result {
            Err(e) if e.unexpected_eof() => Ok(false),
            Err(e) => Err(e),
            Ok(()) => Ok(true),
        }
    }

    pub fn load_with_filter<R: Read>(
        &mut self,
        bitstream: &mut Bitstream<R>,
        mut filter_fn: impl FnMut(&FrameHeader, &FrameData, TocGroupKind) -> bool,
    ) -> Result<()> {
        let span = tracing::span!(tracing::Level::TRACE, "Frame::load_with_filter");
        let _guard = span.enter();

        while let Some(&instr) = self.plan.get(self.next_instr) {
            self.process_instr(bitstream, instr, &mut filter_fn)?;
            self.next_instr += 1;
        }

        Ok(())
    }

    #[inline]
    pub fn load<R: Read> (
        &mut self,
        bitstream: &mut Bitstream<R>,
    ) -> Result<()> {
        self.load_with_filter(bitstream, |_, _, _| true)
    }

    fn process_instr<R: Read>(
        &mut self,
        bitstream: &mut Bitstream<R>,
        instr: GroupInstr,
        mut filter_fn: impl FnMut(&FrameHeader, &FrameData, TocGroupKind) -> bool,
    ) -> Result<()> {
        let span = tracing::span!(tracing::Level::TRACE, "Frame::process_instr", instr = format_args!("{:?}", instr));
        let _guard = span.enter();

        match instr {
            GroupInstr::Read(slot_idx, group) => {
                tracing::trace!(group_kind = format_args!("{:?}", group.kind), "Reading group into memory");
                bitstream.skip_to_bookmark(group.offset)?;

                let mut b = bitstream.rewindable();
                let mut buf = vec![0u8; group.size as usize];
                b.read_bytes_aligned(&mut buf)?;
                b.commit();

                self.buf_slot.insert(slot_idx, (group.kind, buf));
            },
            GroupInstr::Decode(slot_idx) => {
                let (kind, buf) = self.buf_slot.remove(&slot_idx).expect("specified slot is not present");
                tracing::trace!(group_kind = format_args!("{:?}", kind), "Decoding group");
                if !filter_fn(&self.header, &self.data, kind) {
                    return Ok(());
                }

                let mut bitstream = Bitstream::new(std::io::Cursor::new(buf));
                self.load_single(&mut bitstream, kind)?;
            },
        }
        Ok(())
    }

    fn load_single<R: Read>(
        &mut self,
        bitstream: &mut Bitstream<R>,
        kind: TocGroupKind,
    ) -> Result<()> {
        match kind {
            TocGroupKind::All => {
                self.read_merged_group(bitstream)?;
            },
            TocGroupKind::LfGlobal => {
                self.data.lf_global = Some(self.read_lf_global(bitstream)?);
            },
            TocGroupKind::LfGroup(lf_group_idx) => {
                let lf_global = self.data.lf_global.as_ref().expect("invalid decode plan: LfGlobal not decoded");
                self.data.lf_group.insert(lf_group_idx, self.read_lf_group(bitstream, lf_global, lf_group_idx)?);
            },
            TocGroupKind::HfGlobal => {
                let lf_global = self.data.lf_global.as_ref().expect("invalid decode plan: LfGlobal not decoded");
                self.data.hf_global = Some(self.read_hf_global(bitstream, lf_global)?);
            },
            TocGroupKind::GroupPass { pass_idx, group_idx } => {
                let lf_global = self.data.lf_global.as_ref().expect("invalid decode plan: LfGlobal not decoded");
                let lf_group_idx = self.header.lf_group_idx_from_group_idx(group_idx);
                let lf_group = self.data.lf_group.get(&lf_group_idx).expect("invalid decode plan: LfGroup not decoded");
                let hf_global = self.data.hf_global.as_ref().expect("invalid decode plan: HfGlobal not decoded");

                let group = self.read_group_pass(
                    bitstream,
                    lf_global,
                    lf_group,
                    hf_global.as_ref(),
                    pass_idx,
                    group_idx,
                )?;
                self.data.group_pass.insert((pass_idx, group_idx), group);
            },
        }
        Ok(())
    }

    pub fn load_cropped<R: Read>(
        &mut self,
        bitstream: &mut Bitstream<R>,
        region: Option<(u32, u32, u32, u32)>,
    ) -> Result<()> {
        let span = tracing::span!(tracing::Level::TRACE, "Frame::load_cropped");
        let _guard = span.enter();

        let mut region = if region.is_some() && self.header.have_crop {
            region.map(|(left, top, width, height)| (
                left.saturating_add_signed(-self.header.x0),
                top.saturating_add_signed(-self.header.y0),
                width,
                height,
            ))
        } else {
            region
        };
        let mut region_adjust_done = false;

        let crop_filter_fn = |frame_header: &FrameHeader, frame_data: &FrameData, kind| {
            if !region_adjust_done {
                let Some(lf_global) = frame_data.lf_global.as_ref() else {
                    return true;
                };
                if lf_global.gmodular.modular.has_delta_palette() {
                    if region.take().is_some() {
                        tracing::debug!("GlobalModular has delta palette, forcing full decode");
                    }
                } else if lf_global.gmodular.modular.has_squeeze() {
                    if let Some((left, top, width, height)) = &mut region {
                        *width += *left;
                        *height += *top;
                        *left = 0;
                        *top = 0;
                        tracing::debug!("GlobalModular has squeeze, decoding from top-left");
                    }
                }
                if let Some(region) = &region {
                    tracing::debug!("Cropped decoding: {:?}", region);
                }
                region_adjust_done = true;
            }

            let Some(region) = region else { return true; };

            match kind {
                TocGroupKind::LfGroup(lf_group_idx) => {
                    frame_header.is_lf_group_collides_region(lf_group_idx, region)
                },
                TocGroupKind::GroupPass { group_idx, .. } => {
                    frame_header.is_group_collides_region(group_idx, region)
                },
                _ => true,
            }
        };

        self.load_with_filter(bitstream, crop_filter_fn)
    }

    pub fn load_all<R: Read>(&mut self, bitstream: &mut Bitstream<R>) -> Result<()> {
        self.load_cropped(bitstream, None)
    }

    pub fn read_lf_global<R: Read>(&mut self, bitstream: &mut Bitstream<R>) -> Result<LfGlobal> {
        read_bits!(bitstream, Bundle(LfGlobal), (self.image_header, &self.header))
    }

    pub fn read_lf_group<R: Read>(&self, bitstream: &mut Bitstream<R>, lf_global: &LfGlobal, lf_group_idx: u32) -> Result<LfGroup> {
        let lf_group_params = LfGroupParams::new(&self.header, lf_global, lf_group_idx);
        read_bits!(bitstream, Bundle(LfGroup), lf_group_params)
    }

    pub fn read_hf_global<R: Read>(&self, bitstream: &mut Bitstream<R>, lf_global: &LfGlobal) -> Result<Option<HfGlobal>> {
        let has_hf_global = self.header.encoding == crate::header::Encoding::VarDct;
        let hf_global = if has_hf_global {
            let params = HfGlobalParams::new(&self.image_header.metadata, &self.header, lf_global);
            Some(HfGlobal::parse(bitstream, params)?)
        } else {
            None
        };
        Ok(hf_global)
    }

    pub fn read_group_pass<R: Read>(
        &self,
        bitstream: &mut Bitstream<R>,
        lf_global: &LfGlobal,
        lf_group: &LfGroup,
        hf_global: Option<&HfGlobal>,
        pass_idx: u32,
        group_idx: u32,
    ) -> Result<PassGroup> {
        let shift = self.pass_shifts.get(&pass_idx).copied();
        let params = PassGroupParams::new(
            &self.header,
            lf_global,
            lf_group,
            hf_global,
            pass_idx,
            group_idx,
            shift,
        );
        read_bits!(bitstream, Bundle(PassGroup), params)
    }

    pub fn read_merged_group<R: Read>(&mut self, bitstream: &mut Bitstream<R>) -> Result<()> {
        let lf_global = self.read_lf_global(bitstream)?;
        let lf_group = self.read_lf_group(bitstream, &lf_global, 0)?;
        let hf_global = self.read_hf_global(bitstream, &lf_global)?;
        let group_pass = self.read_group_pass(bitstream, &lf_global, &lf_group, hf_global.as_ref(), 0, 0)?;

        self.data.lf_global = Some(lf_global);
        self.data.lf_group.insert(0, lf_group);
        self.data.hf_global = Some(hf_global);
        self.data.group_pass.insert((0, 0), group_pass);

        Ok(())
    }

    pub fn complete(&mut self) -> Result<()> {
        self.data.complete()?;
        Ok(())
    }
}

#[derive(Debug)]
pub struct FrameData {
    pub lf_global: Option<LfGlobal>,
    pub lf_group: BTreeMap<u32, LfGroup>,
    pub hf_global: Option<Option<HfGlobal>>,
    pub group_pass: BTreeMap<(u32, u32), PassGroup>,
}

impl FrameData {
    fn new(frame_header: &FrameHeader) -> Self {
        let has_hf_global = frame_header.encoding == crate::header::Encoding::VarDct;
        let hf_global = if has_hf_global {
            None
        } else {
            Some(None)
        };

        Self {
            lf_global: None,
            lf_group: Default::default(),
            hf_global,
            group_pass: Default::default(),
        }
    }

    fn complete(&mut self) -> Result<&mut Self> {
        let Self {
            lf_global,
            lf_group,
            group_pass,
            ..
        } = self;

        let Some(lf_global) = lf_global else {
            return Err(Error::IncompleteFrameData { field: "lf_global" });
        };
        for lf_group in lf_group.values_mut() {
            let mlf_group = std::mem::take(&mut lf_group.mlf_group);
            lf_global.gmodular.modular.copy_from_modular(mlf_group);
        }
        for group in group_pass.values_mut() {
            let modular = std::mem::take(&mut group.modular);
            lf_global.gmodular.modular.copy_from_modular(modular);
        }
        lf_global.apply_modular_inverse_transform();
        Ok(self)
    }
}
