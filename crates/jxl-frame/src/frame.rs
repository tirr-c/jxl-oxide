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
    pass_shifts: BTreeMap<u32, (i32, i32)>,
    pending_groups: BTreeMap<TocGroupKind, Vec<u8>>,
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
        if self.toc.is_single_entry() {
            let group = self.toc.lf_global();
            self.read_group(bitstream, group)?;
            return Ok(());
        }

        for group in self.toc.iter_bitstream_order() {
            self.read_group(bitstream, group)?;
        }

        Ok(())
    }

    pub fn read_group<R: Read>(&mut self, bitstream: &mut Bitstream<R>, group: TocGroup) -> Result<()> {
        bitstream.skip_to_bookmark(group.offset)?;
        let has_hf_global = self.header.encoding == crate::header::Encoding::VarDct;
        match group.kind {
            TocGroupKind::All => {
                let lf_global = read_bits!(bitstream, Bundle(LfGlobal), (self.image_header, &self.header))?;
                self.data.set_lf_global(lf_global);

                let lf_group = {
                    let lf_global = self.data.lf_global().unwrap();
                    let lf_group_params = LfGroupParams::new(&self.header, lf_global, 0);
                    read_bits!(bitstream, Bundle(LfGroup), lf_group_params)?
                };
                self.data.set_lf_group(0, lf_group);

                if has_hf_global {
                    let hf_global = todo!();
                    self.data.set_hf_global(hf_global);
                }

                let lf_global = self.data.lf_global().unwrap();
                let hf_global = self.data.hf_global().unwrap();
                let params = PassGroupParams::new(
                    &self.header,
                    lf_global,
                    hf_global,
                    0,
                    0,
                    Some((0, 3)),
                );
                let group_pass = read_bits!(bitstream, Bundle(PassGroup), params)?;
                self.data.set_group_pass(0, 0, group_pass);

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
                let lf_group_params = LfGroupParams::new(&self.header, lf_global, lf_group_idx);
                let lf_group = read_bits!(bitstream, Bundle(LfGroup), lf_group_params)?;
                self.data.set_lf_group(lf_group_idx, lf_group);
                Ok(())
            },
            TocGroupKind::HfGlobal => {
                if has_hf_global {
                    let hf_global = todo!();
                    self.data.set_hf_global(hf_global);
                    self.try_pending_blocks()?;
                }
                Ok(())
            },
            TocGroupKind::GroupPass { pass_idx, group_idx } => {
                let (Some(lf_global), Some(hf_global)) = (self.data.lf_global(), self.data.hf_global()) else {
                    let mut buf = vec![0u8; group.size as usize];
                    bitstream.read_bytes_aligned(&mut buf)?;
                    self.pending_groups.insert(group.kind, buf);
                    return Ok(());
                };

                let shift = self.pass_shifts.get(&pass_idx).copied();
                let params = PassGroupParams::new(
                    &self.header,
                    lf_global,
                    hf_global,
                    pass_idx,
                    group_idx,
                    shift,
                );
                let group_pass = read_bits!(bitstream, Bundle(PassGroup), params)?;
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
        for lf_group in std::mem::take(lf_group).into_values() {
            lf_global.gmodular.modular.copy_from_modular(lf_group.mlf_group);
        }
        for group in std::mem::take(group_pass).into_values() {
            lf_global.gmodular.modular.copy_from_modular(group.modular);
        }

        lf_global.apply_modular_inverse_transform();

        // TODO: perform vardct

        Ok(self)
    }
}
