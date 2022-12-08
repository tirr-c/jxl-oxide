use std::collections::BTreeMap;
use std::io::{Read, Seek};
use jxl_bitstream::{read_bits, Bitstream, Bundle, header::Headers};
use crate::{
    data::*,
    toc::{TocGroup, TocGroupKind},
    Error,
    FrameHeader,
    Result,
    Toc,
};

#[derive(Debug)]
pub struct Frame<'a> {
    image_header: &'a Headers,
    header: FrameHeader,
    toc: Toc,
    data: FrameData,
    pending_groups: BTreeMap<TocGroupKind, Vec<u8>>,
}

impl<'a> Bundle<&'a Headers> for Frame<'a> {
    type Error = crate::Error;

    fn parse<R: Read>(bitstream: &mut Bitstream<R>, image_header: &Headers) -> Result<Self> {
        bitstream.zero_pad_to_byte()?;
        let header = read_bits!(bitstream, Bundle(FrameHeader), image_header)?;
        let toc = read_bits!(bitstream, Bundle(Toc), &header)?;
        let data = FrameData::new(&header);
        Ok(Self {
            image_header,
            header,
            toc,
            data,
            pending_groups: Default::default(),
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
    pub fn load_all<R: Read>(&mut self, bitstream: &mut Bitstream<R>) -> Result<()> {
        todo!()
    }

    fn read_group<R: Read>(&mut self, bitstream: &mut Bitstream<R>, group: TocGroup) -> Result<()> {
        bitstream.skip_to_bookmark(group.offset)?;
        match group.kind {
            TocGroupKind::All => {
                let has_hf_global = self.header.encoding == crate::header::Encoding::VarDct;

                // self.read_group(bitstream, TocGroupKind::LfGlobal)?;
                // self.read_group(bitstream, TocGroupKind::LfGroup(0))?;
                if has_hf_global {
                    // self.read_group(bitstream, TocGroupKind::HfGlobal)?;
                }
                // self.read_group(bitstream, TocGroupKind::GroupPass { pass_idx: 0, group_idx: 0 })?;
                Ok(())
            },
            TocGroupKind::LfGlobal => {
                let lf_global = read_bits!(bitstream, Bundle(crate::data::LfGlobal), (self.image_header, &self.header))?;
                self.data.lf_global(lf_global);
                self.try_pending_blocks()?;
                Ok(())
            },
            TocGroupKind::LfGroup(lf_group_idx) => {
                let lf_group = todo!();
                self.data.lf_group(lf_group_idx, lf_group);
                Ok(())
            },
            TocGroupKind::HfGlobal => {
                let hf_global = todo!();
                self.data.hf_global(hf_global);
                Ok(())
            },
            TocGroupKind::GroupPass { pass_idx, group_idx } => {
                let group_pass = todo!();
                self.data.group_pass(pass_idx, group_idx, group_pass);
                Ok(())
            },
        }
    }

    fn try_pending_blocks(&mut self) -> Result<()> {
        todo!()
    }

    pub fn complete(&mut self) -> Result<()> {
        self.data.complete(&self.header)?;
        Ok(())
    }
}

#[derive(Debug)]
pub enum FrameData {
    Partial {
        lf_global: Option<LfGlobal>,
        lf_group: BTreeMap<u32, LfGroup>,
        hf_global: Option<Option<HfGlobal>>,
        group_pass: BTreeMap<(u32, u32), PassGroup>,
    },
    Complete {
        lf_global: LfGlobal,
        lf_group: Vec<LfGroup>,
        hf_global: Option<HfGlobal>,
        group_pass: Vec<PassGroup>,
    },
}

impl FrameData {
    fn new(frame_header: &FrameHeader) -> Self {
        let has_hf_global = frame_header.encoding == crate::header::Encoding::VarDct;
        let hf_global = if has_hf_global {
            None
        } else {
            Some(None)
        };

        Self::Partial {
            lf_global: None,
            lf_group: Default::default(),
            hf_global,
            group_pass: Default::default(),
        }
    }

    fn lf_global(&mut self, lf_global: LfGlobal) -> &mut Self {
        let Self::Partial { lf_global: target @ None, .. } = self else {
            panic!()
        };
        *target = Some(lf_global);
        self
    }

    fn lf_group(&mut self, lf_group_idx: u32, lf_group: LfGroup) -> &mut Self {
        let Self::Partial { lf_group: target, .. } = self else {
            panic!()
        };
        target.entry(lf_group_idx).or_insert(lf_group);
        self
    }

    fn hf_global(&mut self, hf_global: HfGlobal) -> &mut Self {
        let Self::Partial { hf_global: target @ None, .. } = self else {
            panic!()
        };
        *target = Some(Some(hf_global));
        self
    }

    fn group_pass(&mut self, pass_idx: u32, group_idx: u32, group_pass: PassGroup) -> &mut Self {
        let Self::Partial { group_pass: target, .. } = self else {
            panic!()
        };
        target.entry((pass_idx, group_idx)).or_insert(group_pass);
        self
    }

    fn complete(&mut self, frame_header: &FrameHeader) -> Result<&mut Self> {
        let num_groups = frame_header.num_groups() as usize;
        let num_passes = frame_header.passes.num_passes as usize;
        let num_lf_groups = frame_header.num_lf_groups() as usize;

        let Self::Partial { lf_global, lf_group, hf_global, group_pass } = self else {
            return Ok(self);
        };

        if lf_global.is_none() {
            return Err(Error::IncompleteFrameData { field: "lf_global" });
        }
        if lf_group.len() < num_lf_groups {
            return Err(Error::IncompleteFrameData { field: "lf_group" });
        }
        if hf_global.is_none() {
            return Err(Error::IncompleteFrameData { field: "hf_global" });
        }
        if group_pass.len() < num_groups * num_passes {
            return Err(Error::IncompleteFrameData { field: "group_pass" });
        }

        let lf_global = lf_global.take().unwrap();
        let lf_group = std::mem::take(lf_group)
            .into_values()
            .collect();
        let hf_global = hf_global.take().unwrap();
        let group_pass = std::mem::take(group_pass)
            .into_values()
            .collect();
        *self = Self::Complete {
            lf_global,
            lf_group,
            hf_global,
            group_pass,
        };

        Ok(self)
    }
}
