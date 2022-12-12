use jxl_bitstream::{define_bundle, read_bits, Bitstream, Bundle};

use crate::{Error, Result};
use super::{ChannelShift, ModularChannelInfo};

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
}

impl<Ctx> Bundle<Ctx> for TransformInfo {
    type Error = crate::Error;

    fn parse<R: std::io::Read>(bitstream: &mut Bitstream<R>, _: Ctx) -> crate::Result<Self> {
        let tr = bitstream.read_bits(2)?;
        match tr {
            0 => read_bits!(bitstream, Bundle(Rct)).map(Self::Rct),
            1 => read_bits!(bitstream, Bundle(Palette)).map(Self::Palette),
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
    pub struct Palette error(crate::Error) {
        begin_c: ty(U32(u(3), 8 + u(6), 72 + u(10), 1096 + u(13))),
        num_c: ty(U32(1, 3, 4, 1 + u(13))),
        nb_colours: ty(U32(u(8), 256 + u(10), 1280 + u(12), 5376 + u(16))),
        nb_deltas: ty(U32(0, 1 + u(8), 257 + u(10), 1281 + u(16))),
        d_pred: ty(u(4)),
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

impl Palette {
    fn transform_channel_info(&self, channels: &mut super::ModularChannels) -> Result<()> {
        let begin_c = self.begin_c;
        let end_c = begin_c + self.num_c;
        if end_c as usize >= channels.info.len() {
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
}
