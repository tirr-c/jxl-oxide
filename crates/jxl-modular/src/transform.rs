use jxl_bitstream::{define_bundle, read_bits, Bitstream, Bundle};
use jxl_grid::Grid;

use crate::{Error, Result};
use super::{ModularChannelInfo, Image, predictor::{Predictor, PredictorState, WpHeader}};

#[derive(Debug)]
pub enum TransformInfo {
    Rct(Rct),
    Palette(Palette),
    Squeeze(Squeeze),
}

impl TransformInfo {
    pub(super) fn transform_channel_info(&self, channels: &mut super::ModularChannels) -> Result<()> {
        match self {
            Self::Rct(_) => Ok(()),
            Self::Palette(pal) => pal.transform_channel_info(channels),
            Self::Squeeze(sq) => sq.transform_channel_info(channels),
        }
    }

    pub(super) fn or_default(&mut self, channels: &mut super::ModularChannels) {
        match self {
            Self::Rct(_) => {},
            Self::Palette(_) => {},
            Self::Squeeze(sq) => sq.or_default(channels),
        }
    }

    pub(super) fn inverse(&self, image: &mut Image) {
        match self {
            Self::Rct(rct) => rct.inverse(image),
            Self::Palette(pal) => pal.inverse(image),
            Self::Squeeze(sq) => sq.inverse(image),
        }
    }
}

impl Bundle<&WpHeader> for TransformInfo {
    type Error = crate::Error;

    fn parse<R: std::io::Read>(bitstream: &mut Bitstream<R>, wp_header: &WpHeader) -> crate::Result<Self> {
        let tr = bitstream.read_bits(2)?;
        match tr {
            0 => read_bits!(bitstream, Bundle(Rct)).map(Self::Rct),
            1 => read_bits!(bitstream, Bundle(Palette), wp_header).map(Self::Palette),
            2 => read_bits!(bitstream, Bundle(Squeeze)).map(Self::Squeeze),
            value => Err(crate::Error::Bitstream(jxl_bitstream::Error::InvalidEnum {
                name: "TransformId",
                value,
            }))
        }
    }
}

define_bundle! {
    #[derive(Debug)]
    pub struct Rct error(crate::Error) {
        begin_c: ty(U32(u(3), 8 + u(6), 72 + u(10), 1096 + u(13))),
        rct_type: ty(U32(6, u(2), 2 + u(4), 10 + u(6))),
    }

    #[derive(Debug)]
    pub struct Squeeze error(crate::Error) {
        num_sq: ty(U32(0, 1 + u(4), 9 + u(6), 41 + u(8))),
        sp: ty(Vec[Bundle(SqueezeParams)]; num_sq),
    }

    #[derive(Debug)]
    struct SqueezeParams error(crate::Error) {
        horizontal: ty(Bool),
        in_place: ty(Bool),
        begin_c: ty(U32(u(3), 8 + u(6), 72 + u(10), 1096 + u(13))),
        num_c: ty(U32(1, 2, 3, 4 + u(4))),
    }
}

#[derive(Debug)]
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

    fn parse<R: std::io::Read>(bitstream: &mut Bitstream<R>, wp_header: &WpHeader) -> Result<Self> {
        let begin_c = read_bits!(bitstream, U32(u(3), 8 + u(6), 72 + u(10), 1096 + u(13)))?;
        let num_c = read_bits!(bitstream, U32(1, 3, 4, 1 + u(13)))?;
        let nb_colours = read_bits!(bitstream, U32(u(8), 256 + u(10), 1280 + u(12), 5376 + u(16)))?;
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
    fn inverse(&self, image: &mut super::Image) {
        let permutation = (self.rct_type / 7) as usize;
        let ty = self.rct_type % 7;

        let channel_data = image.channel_data_mut();
        let begin_c = self.begin_c as usize;
        let mut channels = channel_data
            .iter_mut()
            .skip(begin_c)
            .take(3)
            .collect::<Vec<_>>();

        jxl_grid::zip_iterate(&mut channels, |samples| {
            let [a, b, c] = samples else { unreachable!() };
            let a = **a;
            let b = **b;
            let c = **c;
            let d;
            let e;
            let f;
            if ty == 6 {
                let tmp = a - (c >> 1);
                e = c + tmp;
                f = tmp - (b >> 1);
                d = f + b;
            } else {
                d = a;
                f = if ty & 1 != 0 {
                    c + a
                } else {
                    c
                };
                e = if (ty >> 1) == 1 {
                    b + a
                } else if (ty >> 1) == 2 {
                    b + ((a + f) >> 1)
                } else {
                    b
                };
            }
            *samples[permutation % 3] = d;
            *samples[(permutation + 1 + (permutation / 3)) % 3] = e;
            *samples[(permutation + 2 - (permutation / 3)) % 3] = f;
        });
    }
}

impl Palette {
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

    fn transform_channel_info(&self, channels: &mut super::ModularChannels) -> Result<()> {
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

        channels.info.drain((begin_c as usize + 1)..(end_c as usize));
        channels.info.insert(0, ModularChannelInfo::new_shifted(self.nb_colours, self.num_c, -1, -1));
        Ok(())
    }

    fn inverse(&self, image: &mut Image) {
        let bit_depth = image.bit_depth();
        let channel_data = image.channel_data_mut();
        let begin_c = self.begin_c as usize;
        let num_c = self.num_c as usize;

        for i in 1..num_c {
            let grid = channel_data[begin_c + 1].clone();
            channel_data.insert(begin_c + 1 + i, grid);
        }

        let (palette_grid, left) = channel_data.split_first_mut().unwrap();
        let palette_grid = &*palette_grid;
        let output_grids = left.iter_mut().skip(begin_c).take(num_c).collect::<Vec<_>>();

        let nb_deltas = self.nb_deltas as i32;
        let nb_colors = self.nb_colours as i32;
        for (c, grid) in output_grids.into_iter().enumerate() {
            let c = c as i32;
            let mut need_delta = Vec::new();
            grid.iter_init_mut(|x, y, sample| {
                let index = *sample;
                if index < nb_deltas {
                    need_delta.push((x, y));
                }
                if (0..nb_colors).contains(&index) {
                    *sample = palette_grid[(index, c)];
                } else if index >= nb_colors {
                    let index = index - nb_colors;
                    if index < 64 {
                        let value = *sample;
                        *sample = ((value >> (2 * c)) % 4) * ((1i32 << bit_depth) - 1) / 4 +
                            (1i32 << bit_depth.saturating_sub(3));
                    } else {
                        let mut index = index - 64;
                        for _ in 0..c {
                            index /= 5;
                        }
                        *sample = (index % 5) * ((1i32 << bit_depth) - 1) / 4;
                    }
                } else if c < 3 {
                    let index = -(index + 1);
                    let index = (index % 143) as usize;
                    *sample = Self::DELTA_PALETTE[(index + 1) >> 1][c as usize] as i32;
                    if index & 1 == 0 {
                        *sample = -*sample;
                    }
                    if bit_depth > 8 {
                        *sample <<= bit_depth.min(24) - 8;
                    }
                } else {
                    *sample = 0;
                }
            });

            if need_delta.is_empty() {
                continue;
            }
            need_delta.sort_by(|(lx, ly), (rx, ry)| ly.cmp(ry).then(lx.cmp(rx)));

            let d_pred = self.d_pred;
            let wp_header = if d_pred == Predictor::SelfCorrecting {
                self.wp_header.clone()
            } else {
                None
            };
            let width = grid.width() as i32;
            let height = grid.height() as i32;
            let mut predictor = PredictorState::new(grid.width(), 0, 0, 0, wp_header);

            let mut idx = 0;
            'outer: for y in 0..height {
                for x in 0..width {
                    let properties = predictor.properties(&[]);
                    let diff = d_pred.predict(&properties);
                    let sample = &mut grid[(x, y)];
                    if need_delta[idx] == (x as u32, y as u32) {
                        *sample += diff;
                        idx += 1;
                        if idx >= need_delta.len() {
                            break 'outer;
                        }
                    }
                    properties.record(*sample);
                }
            }
        }

        channel_data.remove(0);
    }
}

impl Squeeze {
    fn or_default(&mut self, channels: &mut super::ModularChannels) {
        let first = channels.nb_meta_channels;

        let first_ch = &channels.info[first as usize];
        let mut w = first_ch.width;
        let mut h = first_ch.height;
        if let Some(next_ch) = channels.info.get(first as usize + 1) {
            if next_ch.width == w && next_ch.height == h {
                let param_base = SqueezeParams {
                    begin_c: first + 1,
                    num_c: 2,
                    in_place: false,
                    horizontal: false,
                };
                self.sp.push(SqueezeParams { horizontal: true, ..param_base });
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
            self.sp.push(SqueezeParams { horizontal: false, ..param_base });
            h = (h + 1) / 2;
        }
        while w > 8 || h > 8 {
            if w > 8 {
                self.sp.push(SqueezeParams { horizontal: true, ..param_base });
                w = (w + 1) / 2;
            }
            if h > 8 {
                self.sp.push(SqueezeParams { horizontal: false, ..param_base });
                h = (h + 1) / 2;
            }
        }
    }

    fn transform_channel_info(&self, channels: &mut super::ModularChannels) -> Result<()> {
        for sp in &self.sp {
            let SqueezeParams { horizontal, in_place, begin_c: begin, num_c } = *sp;
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

            for ch in &mut channels.info[(begin as usize)..(end as usize)] {
                let mut residu = ch.clone();
                let super::ModularChannelInfo { width: w, height: h, hshift, vshift } = ch;
                if *w == 0 || *h == 0 {
                    return Err(Error::InvalidSqueezeParams);
                }

                let (target_len, target_shift, residu_len) = if horizontal {
                    (w, hshift, &mut residu.width)
                } else {
                    (h, vshift, &mut residu.height)
                };
                *target_len = (*target_len + 1) / 2;
                if *target_shift >= 0 {
                    *target_shift += 1;
                }
                *residu_len = *target_len / 2;

                residu_channels.push(residu);
            }

            if in_place {
                residu_channels.extend(channels.info.drain((end as usize)..));
            }
            channels.info.extend(residu_channels);
        }
        Ok(())
    }

    fn inverse(&self, image: &mut Image) {
        let channel_data = image.channel_data_mut();
        for sp in self.sp.iter().rev() {
            let begin = sp.begin_c as usize;
            let channel_count = sp.num_c as usize;
            let end = begin + channel_count;
            let residual_channels: Vec<_> = if sp.in_place {
                channel_data.drain(end..(end + channel_count)).collect()
            } else {
                channel_data.drain((channel_data.len() - channel_count)..).collect()
            };

            for (ch, residu) in channel_data[begin..end].iter_mut().zip(residual_channels) {
                let output = sp.inverse(&*ch, &residu);
                *ch = output;
            }
        }
    }
}

impl SqueezeParams {
    fn inverse(&self, i0: &Grid<i32>, i1: &Grid<i32>) -> Grid<i32> {
        if self.horizontal {
            self.inverse_h(i0, i1)
        } else {
            self.inverse_v(i0, i1)
        }
    }

    fn inverse_h(&self, i0: &Grid<i32>, i1: &Grid<i32>) -> Grid<i32> {
        assert_eq!(i0.height(), i1.height());

        let height = i1.height() as i32;
        let width = i1.width() as i32;
        let w0 = i0.width() as i32;
        let mut output = Grid::new(i0.width() + i1.width(), i1.height(), i0.group_size());
        for y in 0..height {
            for x in 0..width {
                let avg = i0[(x, y)];
                let residu = i1[(x, y)];
                let next_avg = if x + 1 < w0 { i0[(x + 1, y)] } else { avg };
                let left = if x > 0 {
                    output[(2 * x - 1, y)]
                } else {
                    avg
                };
                let diff = residu + tendency(left, avg, next_avg);
                let first = avg + diff / 2;
                output[(2 * x, y)] = first;
                output[(2 * x + 1, y)] = first - diff;
            }
            if w0 > width {
                output[(2 * width, y)] = i0[(width, y)];
            }
        }
        output
    }

    fn inverse_v(&self, i0: &Grid<i32>, i1: &Grid<i32>) -> Grid<i32> {
        assert_eq!(i0.width(), i1.width());

        let width = i1.width() as i32;
        let height = i1.height() as i32;
        let h0 = i0.height() as i32;
        let mut output = Grid::new(i1.width(), i0.height() + i1.height(), i0.group_size());
        for y in 0..height {
            for x in 0..width {
                let avg = i0[(x, y)];
                let residu = i1[(x, y)];
                let next_avg = if y + 1 < h0 { i0[(x, y + 1)] } else { avg };
                let top = if y > 0 {
                    output[(x, 2 * y - 1)]
                } else {
                    avg
                };
                let diff = residu + tendency(top, avg, next_avg);
                let first = avg + diff / 2;
                output[(x, 2 * y)] = first;
                output[(x, 2 * y + 1)] = first - diff;
            }
        }
        if h0 > height {
            for x in 0..width {
                output[(x, 2 * height)] = i0[(x, height)];
            }
        }
        output
    }
}

fn tendency(a: i32, b: i32, c: i32) -> i32 {
    if a >= b && b >= c {
        let mut x = (4 * a - 3 * c - b + 6) / 12;
        if x - (x & 1) > 2 * (a - b) {
            x = 2 * (a - b) + 1;
        }
        if x + (x & 1) > 2 * (b - c) {
            x = 2 * (b - c);
        }
        x
    } else if a <= b && b <= c {
        let mut x = (4 * a - 3 * c - b - 6) / 12;
        if x + (x & 1) < 2 * (a - b) {
            x = 2 * (a - b) - 1;
        }
        if x - (x & 1) < 2 * (b - c) {
            x = 2 * (b - c);
        }
        x
    } else {
        0
    }
}
