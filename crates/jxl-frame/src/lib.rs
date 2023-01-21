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
        if self.toc.is_single_entry() {
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
                eprintln!("GlobalModular has delta palette, forcing full decode");
            }
        } else if lf_global.gmodular.modular.has_squeeze() {
            if let Some((left, top, width, height)) = &mut region {
                *width += *left;
                *height += *top;
                *left = 0;
                *top = 0;
                eprintln!("GlobalModular has squeeze, decoding from top-left");
            }
        }
        if let Some(region) = &region {
            eprintln!("Cropped decoding: {:?}", region);
        }

        for (lf_group_idx, mut bitstream) in pending_lf_groups {
            if let Some(region) = region {
                let lf_group_dim = self.header.lf_group_dim();
                let lf_group_per_row = self.header.lf_groups_per_row();
                let group_left = (lf_group_idx % lf_group_per_row) * lf_group_dim;
                let group_top = (lf_group_idx / lf_group_per_row) * lf_group_dim;
                if !is_aabb_collides(region, (group_left, group_top, lf_group_dim, lf_group_dim)) {
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

        let it = it.map(|v| {
            let TocGroupKind::GroupPass { pass_idx, group_idx } = v.kind else { panic!() };
            (pass_idx, group_idx, None, Some(v.offset))
        });

        for (pass_idx, group_idx, local_bitstream, offset) in pending_groups.into_iter().chain(it) {
            if let Some(region) = region {
                let group_dim = self.header.group_dim();
                let group_per_row = self.header.groups_per_row();
                let group_left = (group_idx % group_per_row) * group_dim;
                let group_top = (group_idx / group_per_row) * group_dim;
                if !is_aabb_collides(region, (group_left, group_top, group_dim, group_dim)) {
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

    #[cfg(feature = "mt")]
    pub fn load_cropped_par<R: Read + Send>(
        &mut self,
        bitstream: &mut Bitstream<R>,
        region: Option<(u32, u32, u32, u32)>,
    ) -> Result<()> {
        use rayon::prelude::*;

        if self.toc.is_single_entry() {
            let group = self.toc.lf_global();
            bitstream.skip_to_bookmark(group.offset)?;
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

        let mut lf_global = self.data.lf_global.take();
        let mut hf_global = self.data.hf_global.take();
        let mut hf_bitstream = None;

        let (lf_group_tx, lf_group_rx) = crossbeam_channel::unbounded();
        let (pass_group_tx, pass_group_rx) = crossbeam_channel::unbounded();

        let mut it = self.toc.iter_bitstream_order();
        while lf_global.is_none() || hf_global.is_none() {
            let group = it.next().expect("lf_global or hf_global not found?");
            bitstream.skip_to_bookmark(group.offset)?;

            match group.kind {
                TocGroupKind::LfGlobal => {
                    let mut buf = vec![0u8; group.size as usize];
                    bitstream.read_bytes_aligned(&mut buf)?;
                    let mut bitstream = Bitstream::new(std::io::Cursor::new(buf));
                    lf_global = Some(self.read_lf_global(&mut bitstream)?);

                    if let Some(mut hf_bitstream) = hf_bitstream.take() {
                        let lf_global = lf_global.as_ref().unwrap();
                        hf_global = Some(self.read_hf_global(&mut hf_bitstream, lf_global)?);
                    }
                },
                TocGroupKind::HfGlobal => {
                    let mut buf = vec![0u8; group.size as usize];
                    bitstream.read_bytes_aligned(&mut buf)?;
                    let mut bitstream = Bitstream::new(std::io::Cursor::new(buf));

                    if let Some(lf_global) = &lf_global {
                        hf_global = Some(self.read_hf_global(&mut bitstream, lf_global)?);
                    } else {
                        hf_bitstream = Some(bitstream);
                    }
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

        if lf_global.gmodular.modular.has_delta_palette() {
            if region.take().is_some() {
                eprintln!("GlobalModular has delta palette, forcing full decode");
            }
        } else if lf_global.gmodular.modular.has_squeeze() {
            if let Some((left, top, width, height)) = &mut region {
                *width += *left;
                *height += *top;
                *left = 0;
                *top = 0;
                eprintln!("GlobalModular has squeeze, decoding from top-left");
            }
        }
        if let Some(region) = &region {
            eprintln!("Cropped decoding: {:?}", region);
        }

        let mut lf_groups = Ok(BTreeMap::new());
        let mut pass_groups = Ok(BTreeMap::new());
        let io_result = rayon::scope(|scope| -> Result<()> {
            let lf_group_tx = lf_group_tx;
            let pass_group_tx = pass_group_tx;

            scope.spawn(|_| {
                lf_groups = lf_group_rx
                    .into_iter()
                    .par_bridge()
                    .filter(|(lf_group_idx, _)| {
                        let Some(region) = region else { return true; };
                        let lf_group_dim = self.header.lf_group_dim();
                        let lf_group_per_row = self.header.lf_groups_per_row();
                        let group_left = (lf_group_idx % lf_group_per_row) * lf_group_dim;
                        let group_top = (lf_group_idx / lf_group_per_row) * lf_group_dim;
                        is_aabb_collides(region, (group_left, group_top, lf_group_dim, lf_group_dim))
                    })
                    .map(|(lf_group_idx, buf)| {
                        let mut bitstream = Bitstream::new(std::io::Cursor::new(buf));
                        let lf_group = self.read_lf_group(&mut bitstream, lf_global, lf_group_idx)?;
                        Ok((lf_group_idx, lf_group))
                    })
                    .collect::<Result<BTreeMap<_, _>>>();
                let Ok(lf_groups) = &lf_groups else { return; };

                pass_groups = pass_group_rx
                    .into_iter()
                    .par_bridge()
                    .filter(|(_, group_idx, _)| {
                        let Some(region) = region else { return true; };
                        let group_dim = self.header.group_dim();
                        let group_per_row = self.header.groups_per_row();
                        let group_left = (group_idx % group_per_row) * group_dim;
                        let group_top = (group_idx / group_per_row) * group_dim;
                        is_aabb_collides(region, (group_left, group_top, group_dim, group_dim))
                    })
                    .map(|(pass_idx, group_idx, buf)| {
                        let mut bitstream = Bitstream::new(std::io::Cursor::new(buf));
                        let lf_group_id = self.header.lf_group_idx_from_group_idx(group_idx);
                        let lf_group = lf_groups.get(&lf_group_id).unwrap();
                        let pass_group = self.read_group_pass(&mut bitstream, lf_global, lf_group, hf_global, pass_idx, group_idx)?;
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

    #[cfg(feature = "mt")]
    pub fn load_all_par<R: Read + Send>(&mut self, bitstream: &mut Bitstream<R>) -> Result<()> {
        self.load_cropped_par(bitstream, None)
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

        jxl_grid::rgba_be_interleaved(rgb, a, bit_depth, f)
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

fn is_aabb_collides(rect0: (u32, u32, u32, u32), rect1: (u32, u32, u32, u32)) -> bool {
    let (x0, y0, w0, h0) = rect0;
    let (x1, y1, w1, h1) = rect1;
    (x0 < x1 + w1) && (x0 + w0 > x1) && (y0 < y1 + h1) && (y0 + h0 > y1)
}
