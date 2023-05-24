//! This crate provides types related to JPEG XL frames.
//!
//! A JPEG XL image contains one or more frames. A frame represents single unit of image that can
//! be displayed or referenced by other frames.
//!
//! A frame consists of a few components:
//! - [Frame header][FrameHeader].
//! - [Table of contents (TOC)][data::Toc].
//! - [Actual frame data][FrameData], in the following order, potentially permuted as specified in
//!   the TOC:
//!   - one [`LfGlobal`][data::LfGlobal],
//!   - [`num_lf_groups`] [`LfGroup`][data::LfGroup]'s, in raster order,
//!   - one [`HfGlobal`][data::HfGlobal], potentially empty for Modular frames, and
//!   - [`num_passes`] times [`num_groups`] [`PassGroup`][data::PassGroup]'s, in raster order.
//!
//! [`num_lf_groups`]: FrameHeader::num_lf_groups
//! [`num_groups`]: FrameHeader::num_groups
//! [`num_passes`]: header::Passes::num_passes
use std::collections::{HashMap, HashSet};
use std::collections::BTreeMap;
use std::io::Read;

use header::Encoding;
use jxl_bitstream::{read_bits, Bitstream, Bundle};
use jxl_image::ImageHeader;

mod error;
pub mod filter;
pub mod data;
pub mod header;

pub use error::{Error, Result};
pub use header::FrameHeader;

use crate::data::*;

/// JPEG XL frame.
///
/// A frame represents a single unit of image that can be displayed or referenced by other frames.
#[derive(Debug)]
pub struct Frame<'a> {
    image_header: &'a ImageHeader,
    header: FrameHeader,
    toc: Toc,
    plan: Vec<GroupInstr>,
    next_instr: usize,
    buf_slot: HashMap<usize, (TocGroupKind, Vec<u8>)>,
    data: FrameData,
    pass_shifts: BTreeMap<u32, (i32, i32)>,
}

#[derive(Debug, Copy, Clone)]
enum GroupInstr {
    Read(usize, TocGroup),
    Decode(usize),
    ProgressiveScan {
        pass_idx: Option<u32>,
        downsample_factor: u32,
        done: bool,
    },
}

/// Result of progressive loading of a frame.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum ProgressiveResult {
    /// More data is needed to complete a frame or a progressive scan.
    NeedMoreData,
    /// A progressive scan is ready to be rendered.
    SingleScan {
        /// Pass index, `None` if the scan is from the LF image.
        pass_idx: Option<u32>,
        /// Downsample factor of the scan.
        downsample_factor: u32,
        /// Whether the scan completes an image with the given downsample factor.
        done: bool,
    },
    /// A frame is fully loaded.
    FrameComplete,
}

impl<'a> Bundle<&'a ImageHeader> for Frame<'a> {
    type Error = crate::Error;

    fn parse<R: Read>(bitstream: &mut Bitstream<R>, image_header: &'a ImageHeader) -> Result<Self> {
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

        Ok(Self {
            image_header,
            header,
            toc,
            plan: Vec::new(),
            next_instr: 0,
            buf_slot: HashMap::new(),
            data,
            pass_shifts,
        })
    }
}

impl Frame<'_> {
    fn prepare_default_plan(&mut self) {
        let header = &self.header;
        let passes = &header.passes;
        let toc = &self.toc;
        let plan = &mut self.plan;
        if toc.is_single_entry() {
            let group = toc.lf_global();
            plan.push(GroupInstr::Read(0, group));
            plan.push(GroupInstr::Decode(0));
        } else {
            let groups = toc.iter_bitstream_order();
            let num_lf_groups = header.num_lf_groups() as usize;
            let num_groups = header.num_groups() as usize;

            let mut read_slot = HashMap::new();
            let mut decoded_slots = HashSet::new();
            let mut need_lf_global = true;
            let mut need_hf_global = header.encoding == Encoding::VarDct;
            let mut next_slot_idx = 0usize;
            let mut lf_group_count = 0usize;
            let mut group_count_per_pass = (0..passes.num_passes)
                .map(|pass| (pass, 0usize))
                .collect::<BTreeMap<_, _>>();
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
                            lf_group_count += 1;
                            update_pass_groups = true;
                        }
                    },
                    TocGroupKind::GroupPass { pass_idx, group_idx } => {
                        let lf_group_idx = header.lf_group_idx_from_group_idx(group_idx);
                        if need_lf_global || need_hf_global || !decoded_slots.contains(&TocGroupKind::LfGroup(lf_group_idx)) {
                            read_slot.insert(group.kind, current_slot_idx);
                        } else {
                            plan.push(GroupInstr::Decode(current_slot_idx));
                            decoded_slots.insert(group.kind);
                            *group_count_per_pass
                                .get_mut(&pass_idx)
                                .unwrap() += 1;
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
                    lf_group_count += decoded.len();
                    for kind in decoded {
                        read_slot.remove(&kind);
                        decoded_slots.insert(kind);
                    }
                    update_pass_groups = true;
                }

                if update_pass_groups && !need_hf_global {
                    let mut decoded = Vec::new();
                    for (&kind, &slot_idx) in &read_slot {
                        if let TocGroupKind::GroupPass { pass_idx, group_idx } = kind {
                            let lf_group_idx = header.lf_group_idx_from_group_idx(group_idx);
                            if decoded_slots.contains(&TocGroupKind::LfGroup(lf_group_idx)) {
                                plan.push(GroupInstr::Decode(slot_idx));
                                decoded.push(kind);
                                *group_count_per_pass
                                    .get_mut(&pass_idx)
                                    .unwrap() += 1;
                            }
                        }
                    }
                    for kind in decoded {
                        read_slot.remove(&kind);
                        decoded_slots.insert(kind);
                    }
                }

                if lf_group_count == num_lf_groups && !need_hf_global {
                    let done = passes.downsample.first().copied().unwrap_or(1) != 8;
                    plan.push(GroupInstr::ProgressiveScan {
                        pass_idx: None,
                        downsample_factor: 8,
                        done,
                    });
                    lf_group_count += 1;
                }
                if lf_group_count > num_lf_groups {
                    while let Some((&pass_idx, &v)) = group_count_per_pass.first_key_value() {
                        if v == num_groups {
                            let search_result = passes.last_pass.binary_search(&pass_idx);
                            let factor_idx = match search_result {
                                Ok(v) | Err(v) => v,
                            };
                            let done = search_result.is_ok() || passes.last_pass.len() == factor_idx;
                            let downsample_factor = passes.downsample
                                .get(factor_idx)
                                .copied()
                                .unwrap_or(1);
                            plan.push(GroupInstr::ProgressiveScan {
                                pass_idx: Some(pass_idx),
                                downsample_factor,
                                done,
                            });
                            group_count_per_pass.pop_first();
                        } else {
                            break;
                        }
                    }
                }
            }
        }

        if let Some(GroupInstr::ProgressiveScan { .. }) = plan.last() {
            plan.pop();
        }
    }
}

impl Frame<'_> {
    /// Returns the frame header.
    pub fn header(&self) -> &FrameHeader {
        &self.header
    }

    /// Returns the TOC.
    ///
    /// See the documentation of [`Toc`] for details.
    pub fn toc(&self) -> &Toc {
        &self.toc
    }

    /// Returns the frame data.
    pub fn data(&self) -> &FrameData {
        &self.data
    }
}

impl Frame<'_> {
    /// Adjusts the cropping region of the image to the actual decoding region of the frame.
    ///
    /// The cropping region of the *image* needs to be adjusted to be used in a *frame*, for a few
    /// reasons:
    /// - A frame may be blended to the canvas with offset, which makes the image and the frame
    ///   have different coordinates.
    /// - Some filters reference other samples, which requires padding to the region.
    ///
    /// This method takes care of those and adjusts the given region appropriately.
    pub fn adjust_region(&self, (left, top, width, height): &mut (u32, u32, u32, u32)) {
        if self.header.have_crop {
            *left = left.saturating_add_signed(-self.header.x0);
            *top = top.saturating_add_signed(-self.header.y0);
        };

        let mut padding = 0u32;
        if self.header.restoration_filter.gab.enabled() {
            tracing::debug!("Gabor-like filter requires padding of 1 pixel");
            padding = 1;
        }
        if self.header.restoration_filter.epf.enabled() {
            tracing::debug!("Edge-preserving filter requires padding of 3 pixels");
            padding = 3;
        }
        if padding > 0 {
            let delta_w = (*left).min(padding);
            let delta_h = (*top).min(padding);
            *left -= delta_w;
            *top -= delta_h;
            *width += delta_w + padding;
            *height += delta_h + padding;
        }
    }

    /// Loads the data of the frame with the given TOC filter. A group is loaded if the filter
    /// returns `true` for a given TOC group.
    ///
    /// If `progressive` is true, then the method pauses loading the data at the next progressive
    /// scan marker.
    pub fn load_with_filter<R: Read>(
        &mut self,
        bitstream: &mut Bitstream<R>,
        progressive: bool,
        mut filter_fn: impl FnMut(&FrameHeader, &FrameData, TocGroupKind) -> bool,
    ) -> Result<ProgressiveResult> {
        let span = tracing::span!(tracing::Level::TRACE, "Frame::load_with_filter");
        let _guard = span.enter();

        if self.plan.is_empty() {
            self.prepare_default_plan();
        }

        while let Some(&instr) = self.plan.get(self.next_instr) {
            let result = self.process_instr(bitstream, instr, &mut filter_fn);
            match result {
                Err(e) if e.unexpected_eof() => return Ok(ProgressiveResult::NeedMoreData),
                result => result?,
            }

            self.next_instr += 1;
            if progressive {
                if let GroupInstr::ProgressiveScan { pass_idx, downsample_factor, done } = instr {
                    return Ok(ProgressiveResult::SingleScan {
                        pass_idx,
                        downsample_factor,
                        done,
                    });
                }
            }
        }

        self.data.complete()?;
        Ok(ProgressiveResult::FrameComplete)
    }

    /// Loads the data of the frame with the given cropping region of the frame.
    ///
    /// The region is expected in the frame coordinate. Use [`adjust_region`][Self::adjust_region]
    /// to convert from the region of the image.
    pub fn load_cropped<R: Read>(
        &mut self,
        bitstream: &mut Bitstream<R>,
        region: Option<(u32, u32, u32, u32)>,
    ) -> Result<()> {
        let span = tracing::span!(tracing::Level::TRACE, "Frame::load_cropped");
        let _guard = span.enter();

        self.load_with_filter(bitstream, false, crop_filter(region)).map(drop)
    }

    /// Loads all data of the frame.
    pub fn load_all<R: Read>(&mut self, bitstream: &mut Bitstream<R>) -> Result<()> {
        self.load_cropped(bitstream, None)
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
                let &(kind, ref buf) = self.buf_slot.get(&slot_idx).unwrap();
                tracing::trace!(group_kind = format_args!("{:?}", kind), "Decoding group");
                if !filter_fn(&self.header, &self.data, kind) {
                    return Ok(());
                }

                let mut bitstream = Bitstream::new(std::io::Cursor::new(buf));
                self.data.load_single(&mut bitstream, kind, self.image_header, &self.header, &self.pass_shifts)?;
            },
            GroupInstr::ProgressiveScan { downsample_factor, done, .. } => {
                tracing::debug!(downsample_factor, done, "Single progressive scan");
            },
        }
        Ok(())
    }
}

/// Data of a frame.
#[derive(Debug)]
pub struct FrameData {
    pub lf_global: Option<LfGlobal>,
    pub lf_group: HashMap<u32, LfGroup>,
    pub hf_global: Option<Option<HfGlobal>>,
    pub group_pass: HashMap<(u32, u32), PassGroup>,
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

impl FrameData {
    fn load_single<R: Read>(
        &mut self,
        bitstream: &mut Bitstream<R>,
        kind: TocGroupKind,
        image_header: &ImageHeader,
        frame_header: &FrameHeader,
        pass_shifts: &BTreeMap<u32, (i32, i32)>,
    ) -> Result<()> {
        match kind {
            TocGroupKind::All => {
                let shift = pass_shifts.get(&0).copied();
                self.read_merged_group(bitstream, shift, image_header, frame_header)?;
            },
            TocGroupKind::LfGlobal => {
                self.lf_global = Some(self.read_lf_global(bitstream, image_header, frame_header)?);
            },
            TocGroupKind::LfGroup(lf_group_idx) => {
                let lf_global = self.lf_global.as_ref().expect("invalid decode plan: LfGlobal not decoded");
                self.lf_group.insert(lf_group_idx, self.read_lf_group(bitstream, lf_global, lf_group_idx, frame_header)?);
            },
            TocGroupKind::HfGlobal => {
                let lf_global = self.lf_global.as_ref().expect("invalid decode plan: LfGlobal not decoded");
                self.hf_global = Some(self.read_hf_global(bitstream, lf_global, image_header, frame_header)?);
            },
            TocGroupKind::GroupPass { pass_idx, group_idx } => {
                let lf_global = self.lf_global.as_ref().expect("invalid decode plan: LfGlobal not decoded");
                let lf_group_idx = frame_header.lf_group_idx_from_group_idx(group_idx);
                let lf_group = self.lf_group.get(&lf_group_idx).expect("invalid decode plan: LfGroup not decoded");
                let hf_global = self.hf_global.as_ref().expect("invalid decode plan: HfGlobal not decoded");

                let shift = pass_shifts.get(&pass_idx).copied();
                let group = self.read_group_pass(
                    bitstream,
                    lf_global,
                    lf_group,
                    hf_global.as_ref(),
                    pass_idx,
                    group_idx,
                    shift,
                    frame_header,
                )?;
                self.group_pass.insert((pass_idx, group_idx), group);
            },
        }
        Ok(())
    }

    fn read_lf_global<R: Read>(
        &mut self,
        bitstream: &mut Bitstream<R>,
        image_header: &ImageHeader,
        frame_header: &FrameHeader,
    ) -> Result<LfGlobal> {
        read_bits!(bitstream, Bundle(LfGlobal), (image_header, frame_header))
    }

    fn read_lf_group<R: Read>(
        &self,
        bitstream: &mut Bitstream<R>,
        lf_global: &LfGlobal,
        lf_group_idx: u32,
        frame_header: &FrameHeader,
    ) -> Result<LfGroup> {
        let lf_group_params = LfGroupParams::new(frame_header, lf_global, lf_group_idx);
        read_bits!(bitstream, Bundle(LfGroup), lf_group_params)
    }

    fn read_hf_global<R: Read>(
        &self,
        bitstream: &mut Bitstream<R>,
        lf_global: &LfGlobal,
        image_header: &ImageHeader,
        frame_header: &FrameHeader,
    ) -> Result<Option<HfGlobal>> {
        let has_hf_global = frame_header.encoding == crate::header::Encoding::VarDct;
        let hf_global = if has_hf_global {
            let params = HfGlobalParams::new(&image_header.metadata, frame_header, lf_global);
            Some(HfGlobal::parse(bitstream, params)?)
        } else {
            None
        };
        Ok(hf_global)
    }

    #[allow(clippy::too_many_arguments)]
    fn read_group_pass<R: Read>(
        &self,
        bitstream: &mut Bitstream<R>,
        lf_global: &LfGlobal,
        lf_group: &LfGroup,
        hf_global: Option<&HfGlobal>,
        pass_idx: u32,
        group_idx: u32,
        shift: Option<(i32, i32)>,
        frame_header: &FrameHeader,
    ) -> Result<PassGroup> {
        let params = PassGroupParams::new(
            frame_header,
            lf_global,
            lf_group,
            hf_global,
            pass_idx,
            group_idx,
            shift,
        );
        read_bits!(bitstream, Bundle(PassGroup), params)
    }

    fn read_merged_group<R: Read>(
        &mut self,
        bitstream: &mut Bitstream<R>,
        shift: Option<(i32, i32)>,
        image_header: &ImageHeader,
        frame_header: &FrameHeader,
    ) -> Result<()> {
        let lf_global = self.read_lf_global(bitstream, image_header, frame_header)?;
        let lf_group = self.read_lf_group(bitstream, &lf_global, 0, frame_header)?;
        let hf_global = self.read_hf_global(bitstream, &lf_global, image_header, frame_header)?;
        let group_pass = self.read_group_pass(bitstream, &lf_global, &lf_group, hf_global.as_ref(), 0, 0, shift, frame_header)?;

        self.lf_global = Some(lf_global);
        self.lf_group.insert(0, lf_group);
        self.hf_global = Some(hf_global);
        self.group_pass.insert((0, 0), group_pass);

        Ok(())
    }
}

/// Creates a filter that loads only groups within the given cropping region of the frame.
///
/// The region is expected in the frame coordinate. Use [`Frame::adjust_region`] to convert from
/// the region of the image.
pub fn crop_filter(adjusted_region: Option<(u32, u32, u32, u32)>) -> impl for<'a, 'b> FnMut(&'a FrameHeader, &'b FrameData, TocGroupKind) -> bool {
    let mut region = adjusted_region;
    let mut region_adjust_done = false;

    move |frame_header: &FrameHeader, frame_data: &FrameData, kind| {
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
    }
}
