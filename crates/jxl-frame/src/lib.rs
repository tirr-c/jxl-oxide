use std::{collections::BTreeMap, io::Cursor};
use std::io::Read;

use header::Encoding;
use jxl_bitstream::{read_bits, Bitstream, Bundle, header::Headers};

mod error;
pub mod filter;
pub mod data;
pub mod header;

pub use error::{Error, Result};
pub use header::FrameHeader;
pub use data::Toc;

use crate::data::*;

#[derive(Debug)]
pub struct Frame<'a> {
    image_header: &'a Headers,
    header: FrameHeader,
    toc: Toc,
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

        Ok(Self {
            image_header,
            header,
            toc,
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
    pub fn load_cropped<R: Read>(
        &mut self,
        bitstream: &mut Bitstream<R>,
        region: Option<(u32, u32, u32, u32)>,
    ) -> Result<()> {
        let span = tracing::span!(tracing::Level::TRACE, "Frame::load_cropped");
        let _guard = span.enter();

        if self.toc.is_single_entry() {
            tracing::debug!("Frame has single TOC entry");
            self.read_merged_group(bitstream)?;
            return Ok(());
        }

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

        let mut it = self.toc.iter_bitstream_order();
        let mut pending_lf_groups = Vec::new();
        let mut pending_groups = Vec::new();
        let mut hf_global_bitstream = None;
        let mut lf_global = None;
        let mut found_hf_global = self.header.encoding == Encoding::Modular;
        for group in &mut it {
            bitstream.skip_to_bookmark(group.offset)?;
            let mut buf = vec![0u8; group.size as usize];
            bitstream.read_bytes_aligned(&mut buf)?;
            let mut bitstream = Bitstream::new(Cursor::new(buf));
            match group.kind {
                TocGroupKind::LfGlobal => {
                    lf_global = Some(self.read_lf_global(&mut bitstream)?);
                },
                TocGroupKind::LfGroup(lf_group_idx) => {
                    pending_lf_groups.push((lf_group_idx, bitstream));
                },
                TocGroupKind::HfGlobal => {
                    hf_global_bitstream = Some(bitstream);
                    found_hf_global = true;
                },
                TocGroupKind::GroupPass { pass_idx, group_idx } => {
                    pending_groups.push((pass_idx, group_idx, Some(bitstream), None));
                },
                _ => {},
            }

            if lf_global.is_some() && found_hf_global && pending_lf_groups.len() >= self.header.num_lf_groups() as usize {
                break;
            }
        }

        self.data.lf_global = lf_global;
        let lf_global = self.data.lf_global.as_ref().unwrap();
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

        for (lf_group_idx, mut bitstream) in pending_lf_groups {
            if let Some(region) = region {
                if !self.header.is_lf_group_collides_region(lf_group_idx, region) {
                    continue;
                }
            }
            let lf_group = self.read_lf_group(&mut bitstream, lf_global, lf_group_idx)?;
            self.data.lf_group.insert(lf_group_idx, lf_group);
        }

        if let Some(mut bitstream) = hf_global_bitstream {
            self.data.hf_global = Some(self.read_hf_global(&mut bitstream, lf_global)?);
        }
        let hf_global = self.data.hf_global.as_ref().unwrap().as_ref();

        let it = it.filter_map(|v| {
            let (pass_idx, group_idx) = match v.kind {
                TocGroupKind::GroupPass { pass_idx, group_idx } => (pass_idx, group_idx),
                TocGroupKind::HfGlobal => return None,
                _ => unreachable!(),
            };
            Some((pass_idx, group_idx, None, Some(v.offset)))
        });

        for (pass_idx, group_idx, local_bitstream, offset) in pending_groups.into_iter().chain(it) {
            if let Some(region) = region {
                if !self.header.is_group_collides_region(group_idx, region) {
                    continue;
                }
            }

            let lf_group_idx = self.header.lf_group_idx_from_group_idx(group_idx);
            let lf_group = self.data.lf_group.get(&lf_group_idx).unwrap();
            let group_pass = if let Some(mut bitstream) = local_bitstream {
                self.read_group_pass(&mut bitstream, lf_global, lf_group, hf_global, pass_idx, group_idx)?
            } else {
                bitstream.skip_to_bookmark(offset.unwrap())?;
                self.read_group_pass(bitstream, lf_global, lf_group, hf_global, pass_idx, group_idx)?
            };
            self.data.group_pass.insert((pass_idx, group_idx), group_pass);
        }

        Ok(())
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
