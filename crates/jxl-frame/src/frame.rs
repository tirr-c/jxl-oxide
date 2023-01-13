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

    #[cfg(feature = "mt")]
    pub fn load_all_par<R: Read + Send>(&mut self, bitstream: &mut Bitstream<R>) -> Result<()> {
        use rayon::prelude::*;

        if self.toc.is_single_entry() {
            let group = self.toc.lf_global();
            self.read_group(bitstream, group)?;
            return Ok(());
        }

        let mut lf_global = self.data.lf_global.take();
        let mut hf_global = self.data.hf_global.take();

        let (lf_group_tx, lf_group_rx) = crossbeam_channel::unbounded();
        let (pass_group_tx, pass_group_rx) = crossbeam_channel::unbounded();

        let mut it = self.toc.iter_bitstream_order();
        while lf_global.is_none() || hf_global.is_none() {
            let group = it.next().expect("lf_global or hf_global not found?");
            bitstream.skip_to_bookmark(group.offset)?;

            match group.kind {
                TocGroupKind::LfGlobal => {
                    lf_global = Some(self.read_lf_global(bitstream)?);
                },
                TocGroupKind::HfGlobal => {
                    hf_global = Some(self.read_hf_global(bitstream)?);
                },
                TocGroupKind::LfGroup(lf_group_idx) => {
                    let mut buf = vec![0u8; group.size as usize];
                    bitstream.read_bytes_aligned(&mut buf)?;
                    lf_group_tx.send((lf_group_idx, buf)).unwrap();
                },
                TocGroupKind::GroupPass { pass_idx, group_idx } => {
                    let mut buf = vec![0u8; group.size as usize];
                    bitstream.read_bytes_aligned(&mut buf)?;
                    pass_group_tx.send((pass_idx, group_idx, buf)).unwrap();
                },
                _ => unreachable!(),
            }
        }

        self.data.lf_global = lf_global;
        self.data.hf_global = hf_global;
        let lf_global = self.data.lf_global.as_ref().unwrap();
        let hf_global = self.data.hf_global.as_ref().unwrap().as_ref();

        let mut lf_groups = Ok(BTreeMap::new());
        let mut pass_groups = Ok(BTreeMap::new());
        let io_result = rayon::scope(|scope| -> Result<()> {
            let lf_group_tx = lf_group_tx;
            let pass_group_tx = pass_group_tx;

            scope.spawn(|_| {
                lf_groups = lf_group_rx
                    .into_iter()
                    .par_bridge()
                    .map(|(lf_group_idx, buf)| {
                        let mut bitstream = Bitstream::new(std::io::Cursor::new(buf));
                        let lf_group = self.read_lf_group(&mut bitstream, lf_global, lf_group_idx)?;
                        Ok((lf_group_idx, lf_group))
                    })
                    .collect::<Result<BTreeMap<_, _>>>();
            });
            scope.spawn(|_| {
                pass_groups = pass_group_rx
                    .into_iter()
                    .par_bridge()
                    .map(|(pass_idx, group_idx, buf)| {
                        let mut bitstream = Bitstream::new(std::io::Cursor::new(buf));
                        let pass_group = self.read_group_pass(&mut bitstream, lf_global, hf_global, pass_idx, group_idx)?;
                        Ok(((pass_idx, group_idx), pass_group))
                    })
                    .collect::<Result<BTreeMap<_, _>>>();
            });

            for group in it {
                bitstream.skip_to_bookmark(group.offset)?;
                let mut buf = vec![0u8; group.size as usize];
                bitstream.read_bytes_aligned(&mut buf)?;

                match group.kind {
                    TocGroupKind::LfGroup(lf_group_idx) =>
                        lf_group_tx.send((lf_group_idx, buf)).unwrap(),
                    TocGroupKind::GroupPass { pass_idx, group_idx } =>
                        pass_group_tx.send((pass_idx, group_idx, buf)).unwrap(),
                    _ => { /* ignore */ },
                }
            }

            Ok(())
        });

        io_result?;
        self.data.lf_group = lf_groups?;
        self.data.group_pass = pass_groups?;

        Ok(())
    }

    pub fn read_lf_global<R: Read>(&mut self, bitstream: &mut Bitstream<R>) -> Result<LfGlobal> {
        read_bits!(bitstream, Bundle(LfGlobal), (self.image_header, &self.header))
    }

    pub fn read_lf_group<R: Read>(&self, bitstream: &mut Bitstream<R>, lf_global: &LfGlobal, lf_group_idx: u32) -> Result<LfGroup> {
        let lf_group_params = LfGroupParams::new(&self.header, lf_global, lf_group_idx);
        read_bits!(bitstream, Bundle(LfGroup), lf_group_params)
    }

    pub fn read_hf_global<R: Read>(&self, bitstream: &mut Bitstream<R>) -> Result<Option<HfGlobal>> {
        let has_hf_global = self.header.encoding == crate::header::Encoding::VarDct;
        let hf_global = if has_hf_global {
            todo!()
        } else {
            None
        };
        Ok(hf_global)
    }

    pub fn read_group_pass<R: Read>(&self, bitstream: &mut Bitstream<R>, lf_global: &LfGlobal, hf_global: Option<&HfGlobal>, pass_idx: u32, group_idx: u32) -> Result<PassGroup> {
        let shift = self.pass_shifts.get(&pass_idx).copied();
        let params = PassGroupParams::new(
            &self.header,
            lf_global,
            hf_global,
            pass_idx,
            group_idx,
            shift,
        );
        read_bits!(bitstream, Bundle(PassGroup), params)
    }

    pub fn read_group<R: Read>(&mut self, bitstream: &mut Bitstream<R>, group: TocGroup) -> Result<()> {
        bitstream.skip_to_bookmark(group.offset)?;
        let has_hf_global = self.header.encoding == crate::header::Encoding::VarDct;
        match group.kind {
            TocGroupKind::All => {
                let lf_global = self.read_lf_global(bitstream)?;
                let lf_group = self.read_lf_group(bitstream, &lf_global, 0)?;
                let hf_global = self.read_hf_global(bitstream)?;
                let group_pass = self.read_group_pass(bitstream, &lf_global, hf_global.as_ref(), 0, 0)?;

                self.data.lf_global = Some(lf_global);
                self.data.lf_group.insert(0, lf_group);
                self.data.hf_global = Some(hf_global);
                self.data.group_pass.insert((0, 0), group_pass);

                Ok(())
            },
            TocGroupKind::LfGlobal => {
                let lf_global = read_bits!(bitstream, Bundle(LfGlobal), (self.image_header, &self.header))?;
                self.data.lf_global = Some(lf_global);
                self.try_pending_blocks()?;
                Ok(())
            },
            TocGroupKind::LfGroup(lf_group_idx) => {
                let Some(lf_global) = &self.data.lf_global else {
                    let mut buf = vec![0u8; group.size as usize];
                    bitstream.read_bytes_aligned(&mut buf)?;
                    self.pending_groups.insert(group.kind, buf);
                    return Ok(());
                };
                let lf_group = self.read_lf_group(bitstream, lf_global, lf_group_idx)?;
                self.data.lf_group.insert(lf_group_idx, lf_group);
                Ok(())
            },
            TocGroupKind::HfGlobal => {
                let hf_global = self.read_hf_global(bitstream)?;
                self.data.hf_global = Some(hf_global);
                Ok(())
            },
            TocGroupKind::GroupPass { pass_idx, group_idx } => {
                let (Some(lf_global), Some(hf_global)) = (&self.data.lf_global, &self.data.hf_global) else {
                    let mut buf = vec![0u8; group.size as usize];
                    bitstream.read_bytes_aligned(&mut buf)?;
                    self.pending_groups.insert(group.kind, buf);
                    return Ok(());
                };

                let group_pass = self.read_group_pass(bitstream, lf_global, hf_global.as_ref(), pass_idx, group_idx)?;
                self.data.group_pass.insert((pass_idx, group_idx), group_pass);
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

    pub fn rgba_be_interleaved<F>(&self, f: F) -> Result<()>
    where
        F: FnMut(&[u8]) -> Result<()>,
    {
        let bit_depth = self.image_header.metadata.bit_depth.bits_per_sample();
        let modular_channels = self.data.lf_global.as_ref().unwrap().gmodular.modular.image().channel_data();
        let alpha = self.image_header.metadata.alpha();

        let (rgb, a) = if self.header.encoding == crate::header::Encoding::VarDct {
            todo!()
        } else {
            let rgb = [&modular_channels[0], &modular_channels[1], &modular_channels[2]];
            let a = alpha.map(|idx| &modular_channels[3 + idx]);
            (rgb, a)
        };

        crate::image::rgba_be_interleaved(rgb, a, bit_depth, f)
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

    fn complete(&mut self, frame_header: &FrameHeader) -> Result<&mut Self> {
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
