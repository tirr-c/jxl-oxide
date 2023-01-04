use std::collections::BTreeMap;
use std::io::{Read, Seek};
use jxl_bitstream::{read_bits, Bitstream, Bundle, header::Headers};
use crate::{
    frame_data::*,
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

    fn parse<R: Read>(bitstream: &mut Bitstream<R>, image_header: &'a Headers) -> Result<Self> {
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

    pub fn read_group<R: Read>(&mut self, bitstream: &mut Bitstream<R>, group: TocGroup) -> Result<()> {
        bitstream.skip_to_bookmark(group.offset)?;
        match group.kind {
            TocGroupKind::All => {
                let has_hf_global = self.header.encoding == crate::header::Encoding::VarDct;

                let lf_global = read_bits!(bitstream, Bundle(LfGlobal), (self.image_header, &self.header))?;
                self.data.set_lf_global(lf_global);
                // self.read_group(bitstream, TocGroupKind::LfGroup(0))?;
                if has_hf_global {
                    // self.read_group(bitstream, TocGroupKind::HfGlobal)?;
                }
                // self.read_group(bitstream, TocGroupKind::GroupPass { pass_idx: 0, group_idx: 0 })?;
                Ok(())
            },
            TocGroupKind::LfGlobal => {
                let lf_global = read_bits!(bitstream, Bundle(LfGlobal), (self.image_header, &self.header))?;
                self.data.set_lf_global(lf_global);
                self.try_pending_blocks()?;
                Ok(())
            },
            TocGroupKind::LfGroup(lf_group_idx) => {
                let Some(lf_global) = self.data.lf_global() else {
                    let mut buf = vec![0u8; group.size as usize];
                    bitstream.read_bytes_aligned(&mut buf)?;
                    self.pending_groups.insert(group.kind, buf);
                    return Ok(());
                };
                let lf_group = todo!();
                self.data.set_lf_group(lf_group_idx, lf_group);
                Ok(())
            },
            TocGroupKind::HfGlobal => {
                let hf_global = todo!();
                self.data.set_hf_global(hf_global);
                self.try_pending_blocks()?;
                Ok(())
            },
            TocGroupKind::GroupPass { pass_idx, group_idx } => {
                let (Some(lf_global), Some(hf_global)) = (self.data.lf_global(), self.data.hf_global()) else {
                    let mut buf = vec![0u8; group.size as usize];
                    bitstream.read_bytes_aligned(&mut buf)?;
                    self.pending_groups.insert(group.kind, buf);
                    return Ok(());
                };
                let group_pass = todo!();
                self.data.set_group_pass(pass_idx, group_idx, group_pass);
                Ok(())
            },
        }
    }

    fn try_pending_blocks(&mut self) -> Result<()> {
        // TODO: parse pending blocks
        Ok(())
    }

    pub fn complete(&mut self) -> Result<()> {
        self.data.complete(&self.header)?;
        Ok(())
    }
}

#[derive(Debug)]
pub struct FrameData {
    lf_global: Option<LfGlobal>,
    lf_group: BTreeMap<u32, LfGroup>,
    hf_global: Option<Option<HfGlobal>>,
    group_pass: BTreeMap<(u32, u32), PassGroup>,
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

    fn lf_global(&self) -> Option<&LfGlobal> {
        self.lf_global.as_ref()
    }

    fn set_lf_global(&mut self, lf_global: LfGlobal) -> &mut Self {
        self.lf_global = Some(lf_global);
        self
    }

    fn lf_group(&self, lf_group_idx: u32) -> Option<&LfGroup> {
        self.lf_group.get(&lf_group_idx)
    }

    fn set_lf_group(&mut self, lf_group_idx: u32, lf_group: LfGroup) -> &mut Self {
        self.lf_group.entry(lf_group_idx).or_insert(lf_group);
        self
    }

    fn hf_global(&self) -> Option<Option<&HfGlobal>> {
        self.hf_global.as_ref().map(|x| x.as_ref())
    }

    fn set_hf_global(&mut self, hf_global: HfGlobal) -> &mut Self {
        self.hf_global = Some(Some(hf_global));
        self
    }

    fn group_pass(&self, pass_idx: u32, group_idx: u32) -> Option<&PassGroup> {
        self.group_pass.get(&(pass_idx, group_idx))
    }

    fn set_group_pass(&mut self, pass_idx: u32, group_idx: u32, group_pass: PassGroup) -> &mut Self {
        self.group_pass.entry((pass_idx, group_idx)).or_insert(group_pass);
        self
    }

    fn complete(&mut self, frame_header: &FrameHeader) -> Result<&mut Self> {
        let num_groups = frame_header.num_groups() as usize;
        let num_passes = frame_header.passes.num_passes as usize;
        let num_lf_groups = frame_header.num_lf_groups() as usize;

        let Self {
            lf_global,
            lf_group,
            hf_global,
            group_pass,
        } = self;

        let Some(lf_global) = lf_global else {
            return Err(Error::IncompleteFrameData { field: "lf_global" });
        };
        for lf_group in lf_group.values() {
            // TODO: copy modular into lf_global
        }
        for group in group_pass.values() {
            // TODO: copy modular into lf_global
        }

        lf_global.apply_modular_inverse_transform();

        // TODO: perform vardct

        Ok(self)
    }
}
