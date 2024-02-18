use jxl_bitstream::{define_bundle, read_bits, Bitstream, Bundle};
use jxl_grid::{AllocTracker, CutGrid, SimpleGrid};
use jxl_threadpool::JxlThreadPool;

use super::{
    predictor::{Predictor, PredictorState, WpHeader},
    ModularChannelInfo,
};
use crate::{image::TransformedGrid, Error, Result, Sample};

mod rct;
mod squeeze;

#[derive(Debug, Clone)]
pub enum TransformInfo {
    Rct(Rct),
    Palette(Palette),
    Squeeze(Squeeze),
}

impl TransformInfo {
    pub(super) fn prepare_transform_info(
        &mut self,
        channels: &mut super::ModularChannels,
    ) -> Result<()> {
        match self {
            Self::Rct(rct) => rct.transform_channel_info(channels),
            Self::Palette(pal) => {
                pal.transform_channel_info::<i16>(channels, &mut Vec::new(), None)
            }
            Self::Squeeze(sq) => {
                sq.set_default_params(channels);
                sq.transform_channel_info::<i16>(channels, None)
            }
        }
    }

    pub(super) fn prepare_meta_channels<S: Sample>(
        &self,
        meta_channels: &mut Vec<SimpleGrid<S>>,
        tracker: Option<&AllocTracker>,
    ) -> Result<()> {
        if let Self::Palette(pal) = self {
            meta_channels.insert(
                0,
                SimpleGrid::with_alloc_tracker(
                    pal.nb_colours as usize,
                    pal.num_c as usize,
                    tracker,
                )?,
            );
        }
        Ok(())
    }

    pub(super) fn transform_channels<'dest, S: Sample>(
        &self,
        channels: &mut super::ModularChannels,
        meta_channel_grids: &mut Vec<CutGrid<'dest, S>>,
        grids: &mut Vec<TransformedGrid<'dest, S>>,
    ) -> Result<()> {
        match self {
            Self::Rct(rct) => rct.transform_channel_info(channels),
            Self::Palette(pal) => {
                pal.transform_channel_info(channels, meta_channel_grids, Some(grids))
            }
            Self::Squeeze(sq) => sq.transform_channel_info(channels, Some(grids)),
        }
    }

    pub(super) fn inverse<S: Sample>(
        &self,
        grids: &mut Vec<TransformedGrid<'_, S>>,
        bit_depth: u32,
        pool: &JxlThreadPool,
    ) {
        match self {
            Self::Rct(rct) => rct.inverse(grids, pool),
            Self::Palette(pal) => pal.inverse(grids, bit_depth),
            Self::Squeeze(sq) => sq.inverse(grids, pool),
        }
    }

    pub fn is_palette(&self) -> bool {
        matches!(self, Self::Palette(_))
    }

    pub fn is_squeeze(&self) -> bool {
        matches!(self, Self::Squeeze(_))
    }
}

impl Bundle<&WpHeader> for TransformInfo {
    type Error = crate::Error;

    fn parse(bitstream: &mut Bitstream, wp_header: &WpHeader) -> crate::Result<Self> {
        let tr = bitstream.read_bits(2)?;
        match tr {
            0 => read_bits!(bitstream, Bundle(Rct)).map(Self::Rct),
            1 => read_bits!(bitstream, Bundle(Palette), wp_header).map(Self::Palette),
            2 => read_bits!(bitstream, Bundle(Squeeze)).map(Self::Squeeze),
            value => Err(crate::Error::Bitstream(jxl_bitstream::Error::InvalidEnum {
                name: "TransformId",
                value,
            })),
        }
    }
}

define_bundle! {
    #[derive(Debug, Clone)]
    pub struct Rct error(crate::Error) {
        begin_c: ty(U32(u(3), 8 + u(6), 72 + u(10), 1096 + u(13))),
        rct_type: ty(U32(6, u(2), 2 + u(4), 10 + u(6))),
    }

    #[derive(Debug, Clone)]
    pub struct Squeeze error(crate::Error) {
        num_sq: ty(U32(0, 1 + u(4), 9 + u(6), 41 + u(8))),
        sp: ty(Vec[Bundle(SqueezeParams)]; num_sq),
    }

    #[derive(Debug, Clone)]
    struct SqueezeParams error(crate::Error) {
        horizontal: ty(Bool),
        in_place: ty(Bool),
        begin_c: ty(U32(u(3), 8 + u(6), 72 + u(10), 1096 + u(13))),
        num_c: ty(U32(1, 2, 3, 4 + u(4))),
    }
}

#[derive(Debug, Clone)]
pub struct Palette {
    begin_c: u32,
    num_c: u32,
    nb_colours: u32,
    nb_deltas: u32,
    d_pred: Predictor,
    wp_header: Option<WpHeader>,
}

impl Bundle<&WpHeader> for Palette {
    type Error = crate::Error;

    fn parse(bitstream: &mut Bitstream, wp_header: &WpHeader) -> Result<Self> {
        let begin_c = read_bits!(bitstream, U32(u(3), 8 + u(6), 72 + u(10), 1096 + u(13)))?;
        let num_c = read_bits!(bitstream, U32(1, 3, 4, 1 + u(13)))?;
        let nb_colours = read_bits!(
            bitstream,
            U32(u(8), 256 + u(10), 1280 + u(12), 5376 + u(16))
        )?;
        let nb_deltas = read_bits!(bitstream, U32(0, 1 + u(8), 257 + u(10), 1281 + u(16)))?;
        let d_pred = Predictor::try_from(bitstream.read_bits(4)?)?;
        Ok(Self {
            begin_c,
            num_c,
            nb_colours,
            nb_deltas,
            d_pred,
            wp_header: if d_pred == Predictor::SelfCorrecting {
                Some(wp_header.clone())
            } else {
                None
            },
        })
    }
}

impl Rct {
    fn transform_channel_info(&self, channels: &mut super::ModularChannels) -> Result<()> {
        let begin_c = self.begin_c;
        let end_c = self.begin_c + 3;
        if end_c as usize > channels.info.len() {
            return Err(Error::InvalidRctParams);
        }

        let ModularChannelInfo { width, height, .. } = channels.info[begin_c as usize];
        for info in &channels.info[(begin_c + 1) as usize..end_c as usize] {
            if width != info.width || height != info.height {
                return Err(Error::InvalidRctParams);
            }
        }

        Ok(())
    }

    fn inverse<S: Sample>(&self, grids: &mut [TransformedGrid<'_, S>], pool: &JxlThreadPool) {
        let permutation = self.rct_type / 7;
        let ty = self.rct_type % 7;

        let begin_c = self.begin_c as usize;
        let mut channels = grids.iter_mut().skip(begin_c);
        let a = channels.next().unwrap().grid_mut();
        let b = channels.next().unwrap().grid_mut();
        let c = channels.next().unwrap().grid_mut();

        let a = {
            let g = a.subgrid_mut(.., ..);
            let width = g.width();
            g.into_groups(width, 8)
        };
        let b = {
            let g = b.subgrid_mut(.., ..);
            let width = g.width();
            g.into_groups(width, 8)
        };
        let c = {
            let g = c.subgrid_mut(.., ..);
            let width = g.width();
            g.into_groups(width, 8)
        };
        let groups = a
            .into_iter()
            .zip(b)
            .zip(c)
            .map(|((a, b), c)| (a, b, c))
            .collect::<Vec<_>>();
        pool.for_each_vec(groups, |(mut a, mut b, mut c)| {
            rct::inverse_rct(permutation, ty, [&mut a, &mut b, &mut c]);
        })
    }
}

impl Palette {
    #[rustfmt::skip]
    const DELTA_PALETTE: [[i16; 3]; 72] = [
        [0, 0, 0], [4, 4, 4], [11, 0, 0], [0, 0, -13], [0, -12, 0], [-10, -10, -10],
        [-18, -18, -18], [-27, -27, -27], [-18, -18, 0], [0, 0, -32], [-32, 0, 0], [-37, -37, -37],
        [0, -32, -32], [24, 24, 45], [50, 50, 50], [-45, -24, -24], [-24, -45, -45], [0, -24, -24],
        [-34, -34, 0], [-24, 0, -24], [-45, -45, -24], [64, 64, 64], [-32, 0, -32], [0, -32, 0],
        [-32, 0, 32], [-24, -45, -24], [45, 24, 45], [24, -24, -45], [-45, -24, 24], [80, 80, 80],
        [64, 0, 0], [0, 0, -64], [0, -64, -64], [-24, -24, 45], [96, 96, 96], [64, 64, 0],
        [45, -24, -24], [34, -34, 0], [112, 112, 112], [24, -45, -45], [45, 45, -24], [0, -32, 32],
        [24, -24, 45], [0, 96, 96], [45, -24, 24], [24, -45, -24], [-24, -45, 24], [0, -64, 0],
        [96, 0, 0], [128, 128, 128], [64, 0, 64], [144, 144, 144], [96, 96, 0], [-36, -36, 36],
        [45, -24, -45], [45, -45, -24], [0, 0, -96], [0, 128, 128], [0, 96, 0], [45, 24, -45],
        [-128, 0, 0], [24, -45, 24], [-45, 24, -45], [64, 0, -64], [64, -64, -64], [96, 0, 96],
        [45, -45, 24], [24, 45, -45], [64, 64, -64], [128, 128, 0], [0, 0, -128], [-24, 45, -45],
    ];

    fn transform_channel_info<'dest, S: Sample>(
        &self,
        channels: &mut super::ModularChannels,
        meta_channel_grids: &mut Vec<CutGrid<'dest, S>>,
        grids: Option<&mut Vec<TransformedGrid<'dest, S>>>,
    ) -> Result<()> {
        let begin_c = self.begin_c;
        let end_c = begin_c + self.num_c;
        if end_c as usize > channels.info.len() {
            return Err(Error::InvalidPaletteParams);
        }
        if begin_c < channels.nb_meta_channels {
            if end_c > channels.nb_meta_channels {
                return Err(Error::InvalidPaletteParams);
            }
            channels.nb_meta_channels = channels.nb_meta_channels + 2 - self.num_c;
        } else {
            channels.nb_meta_channels += 1;
        }

        let ModularChannelInfo { width, height, .. } = channels.info[begin_c as usize];
        for info in &channels.info[(begin_c as usize + 1)..end_c as usize] {
            if info.width != width || info.height != height {
                return Err(Error::InvalidPaletteParams);
            }
        }

        channels
            .info
            .drain((begin_c as usize + 1)..(end_c as usize));
        channels.info.insert(
            0,
            ModularChannelInfo::new_shifted(self.nb_colours, self.num_c, -1, -1),
        );

        if let Some(grids) = grids {
            let members = grids
                .drain((begin_c as usize + 1)..(end_c as usize))
                .collect::<Vec<_>>();
            grids[begin_c as usize].merge(members);

            let palette_grid = meta_channel_grids.pop().unwrap();
            grids.insert(0, TransformedGrid::from(palette_grid));
        }
        Ok(())
    }

    fn inverse_once<S: Sample>(
        &self,
        c: usize,
        palette: &CutGrid<'_, S>,
        indices: Option<&CutGrid<'_, S>>,
        grid: &mut CutGrid<'_, S>,
        bit_depth: u32,
    ) {
        let nb_deltas = self.nb_deltas as i32;
        let nb_colors = self.nb_colours as i32;

        let mut need_delta = Vec::new();
        let width = grid.width();
        let height = grid.height();
        for y in 0..height {
            for x in 0..width {
                let index = if let Some(indices) = indices {
                    indices.get(x, y)
                } else {
                    grid.get(x, y)
                }
                .to_i32();
                let sample = grid.get_mut(x, y);
                if index < nb_deltas {
                    need_delta.push((x, y));
                }
                if (0..nb_colors).contains(&index) {
                    *sample = palette.get(index as usize, c);
                } else if index >= nb_colors {
                    let value = index;
                    let index = index - nb_colors;
                    if index < 64 {
                        *sample = S::from_i32(
                            ((value >> (2 * c)) % 4) * ((1i32 << bit_depth) - 1) / 4
                                + (1i32 << bit_depth.saturating_sub(3)),
                        );
                    } else {
                        let mut index = index - 64;
                        for _ in 0..c {
                            index /= 5;
                        }
                        *sample = S::from_i32((index % 5) * ((1i32 << bit_depth) - 1) / 4);
                    }
                } else if c < 3 {
                    let index = -(index + 1);
                    let index = (index % 143) as usize;
                    let mut temp_sample = Self::DELTA_PALETTE[(index + 1) >> 1][c] as i32;
                    if index & 1 == 0 {
                        temp_sample = -temp_sample;
                    }
                    if bit_depth > 8 {
                        temp_sample <<= bit_depth.min(24) - 8;
                    }
                    *sample = S::from_i32(temp_sample);
                } else {
                    *sample = S::default();
                }
            }
        }

        if need_delta.is_empty() {
            return;
        }

        let d_pred = self.d_pred;
        let wp_header = if d_pred == Predictor::SelfCorrecting {
            self.wp_header.as_ref()
        } else {
            None
        };
        let mut predictor = PredictorState::<S>::new();
        predictor.reset(width as u32, &[], wp_header);

        let mut idx = 0;
        for y in 0..height {
            for x in 0..width {
                let properties = predictor.properties();
                let sample = grid.get_mut(x, y);
                let mut sample_value = sample.to_i32();
                if need_delta[idx] == (x, y) {
                    let diff = d_pred.predict(&properties);
                    sample_value = sample_value.wrapping_add(diff);
                    *sample = S::from_i32(sample_value);
                    idx += 1;
                    if idx >= need_delta.len() {
                        return;
                    }
                }
                properties.record(sample_value);
            }
        }
    }

    fn inverse<S: Sample>(&self, grids: &mut Vec<TransformedGrid<'_, S>>, bit_depth: u32) {
        let begin_c = self.begin_c as usize;
        let num_c = self.num_c as usize;

        let palette_grid = grids.remove(0);
        let leader = &mut grids[begin_c];
        let mut members = leader.unmerge(num_c - 1);
        let index_grid = leader.grid();
        let palette_grid = palette_grid.grid();

        for (c, grid) in members.iter_mut().enumerate() {
            self.inverse_once(
                c + 1,
                palette_grid,
                Some(index_grid),
                grid.grid_mut(),
                bit_depth,
            );
        }
        self.inverse_once(0, palette_grid, None, leader.grid_mut(), bit_depth);

        for (i, grid) in members.into_iter().enumerate() {
            grids.insert(begin_c + 1 + i, grid);
        }
    }
}

impl Squeeze {
    fn set_default_params(&mut self, channels: &super::ModularChannels) {
        if !self.sp.is_empty() {
            return;
        }

        let first = channels.nb_meta_channels;
        let first_ch = &channels.info[first as usize];
        let mut w = first_ch.width;
        let mut h = first_ch.height;
        if channels.info.len() as u32 - first >= 3 {
            let next_ch = &channels.info[first as usize + 1];
            if next_ch.width == w && next_ch.height == h {
                let param_base = SqueezeParams {
                    begin_c: first + 1,
                    num_c: 2,
                    in_place: false,
                    horizontal: false,
                };
                self.sp.push(SqueezeParams {
                    horizontal: true,
                    ..param_base
                });
                self.sp.push(param_base);
            }
        }

        let param_base = SqueezeParams {
            begin_c: first,
            num_c: channels.info.len() as u32 - first,
            in_place: true,
            horizontal: false,
        };

        if h >= w && h > 8 {
            self.sp.push(SqueezeParams {
                horizontal: false,
                ..param_base
            });
            h = (h + 1) / 2;
        }
        while w > 8 || h > 8 {
            if w > 8 {
                self.sp.push(SqueezeParams {
                    horizontal: true,
                    ..param_base
                });
                w = (w + 1) / 2;
            }
            if h > 8 {
                self.sp.push(SqueezeParams {
                    horizontal: false,
                    ..param_base
                });
                h = (h + 1) / 2;
            }
        }
    }

    fn transform_channel_info<S: Sample>(
        &self,
        channels: &mut super::ModularChannels,
        mut grids: Option<&mut Vec<TransformedGrid<'_, S>>>,
    ) -> Result<()> {
        for sp in &self.sp {
            let SqueezeParams {
                horizontal,
                in_place,
                begin_c: begin,
                num_c,
            } = *sp;
            let end = begin + num_c;

            if end as usize > channels.info.len() {
                return Err(Error::InvalidSqueezeParams);
            }
            if begin < channels.nb_meta_channels {
                if !in_place || end > channels.nb_meta_channels {
                    return Err(Error::InvalidSqueezeParams);
                }
                channels.nb_meta_channels += num_c;
            }

            let mut residu_cap = num_c as usize;
            if in_place {
                residu_cap += channels.info.len() - end as usize;
            }
            let mut residu_channels = Vec::with_capacity(residu_cap);
            let mut residu_grids = grids.is_some().then(|| Vec::with_capacity(residu_cap));

            for (idx, ch) in channels.info[(begin as usize)..(end as usize)]
                .iter_mut()
                .enumerate()
            {
                let mut residu = ch.clone();
                let super::ModularChannelInfo {
                    width: w,
                    height: h,
                    hshift,
                    vshift,
                } = ch;
                if *w == 0 || *h == 0 {
                    return Err(Error::InvalidSqueezeParams);
                }

                let (target_len, target_shift, residu_len, residu_shift) = if horizontal {
                    (w, hshift, &mut residu.width, &mut residu.hshift)
                } else {
                    (h, vshift, &mut residu.height, &mut residu.vshift)
                };
                let len = *target_len;
                *target_len = (len + 1) / 2;
                *residu_len = len / 2;
                if *target_shift >= 0 {
                    *target_shift += 1;
                    *residu_shift += 1;
                }

                residu_channels.push(residu);

                if let Some(residu_grids) = &mut residu_grids {
                    let grids = grids.as_mut().unwrap();
                    let g = &mut grids[begin as usize + idx];
                    let g = g.grid_mut();
                    let residu_grid = if horizontal {
                        g.split_horizontal_in_place((g.width() + 1) / 2)
                    } else {
                        g.split_vertical_in_place((g.height() + 1) / 2)
                    };
                    residu_grids.push(TransformedGrid::from(residu_grid));
                }
            }

            if in_place {
                residu_channels.extend(channels.info.drain((end as usize)..));
            }
            channels.info.extend(residu_channels);

            if let Some(grids) = &mut grids {
                let mut residu_grids = residu_grids.unwrap();
                if in_place {
                    residu_grids.extend(grids.drain((end as usize)..));
                }
                grids.extend(residu_grids);
            }
        }
        Ok(())
    }

    fn inverse<S: Sample>(&self, grids: &mut Vec<TransformedGrid<'_, S>>, pool: &JxlThreadPool) {
        for sp in self.sp.iter().rev() {
            let begin = sp.begin_c as usize;
            let channel_count = sp.num_c as usize;
            let end = begin + channel_count;
            let residual_channels: Vec<_> = if sp.in_place {
                grids.drain(end..(end + channel_count)).collect()
            } else {
                grids.drain((grids.len() - channel_count)..).collect()
            };

            for (ch, residu) in grids[begin..end].iter_mut().zip(residual_channels) {
                sp.inverse(ch, residu, pool);
            }
        }
    }
}

impl SqueezeParams {
    fn inverse<'dest, S: Sample>(
        &self,
        i0: &mut TransformedGrid<'dest, S>,
        i1: TransformedGrid<'dest, S>,
        pool: &jxl_threadpool::JxlThreadPool,
    ) {
        let i0 = i0.grid_mut();
        let TransformedGrid::Single(i1) = i1 else {
            panic!("residual channel should be Single channel")
        };
        if self.horizontal {
            i0.merge_horizontal_in_place(i1);
            let width = i0.width();
            let height = i0.height();
            if height > 16 {
                let remaining = i0.split_vertical(0).1;
                pool.for_each_vec(remaining.into_groups(width, 16), |mut group| {
                    squeeze::inverse_h(&mut group)
                });
            } else {
                squeeze::inverse_h(i0);
            }
        } else {
            i0.merge_vertical_in_place(i1);
            let width = i0.width();
            let height = i0.height();
            if width > 16 {
                let remaining = i0.split_horizontal(0).1;
                pool.for_each_vec(remaining.into_groups(16, height), |mut group| {
                    squeeze::inverse_v(&mut group)
                });
            } else {
                squeeze::inverse_v(i0);
            }
        }
    }
}
