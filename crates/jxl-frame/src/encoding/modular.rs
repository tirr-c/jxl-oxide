use std::io::Read;

use jxl_bitstream::{define_bundle, read_bits, header::Headers, Bitstream, Bundle};

use crate::{FrameHeader, Result, frame_data::GlobalModular};

mod ma;
mod predictor;
mod transform;
pub use ma::{MaConfig, MaContext};

#[derive(Debug)]
pub struct Modular {
    header: ModularHeader,
    ma_ctx: ma::MaContext,
    channels: ModularChannels,
}

#[derive(Debug, Clone)]
pub struct ModularParams<'a> {
    pub width: u32,
    pub height: u32,
    pub channel_shifts: Vec<i32>,
    pub gmodular: &'a GlobalModular,
}

impl<'a> ModularParams<'a> {
    pub fn new(width: u32, height: u32, channel_shifts: Vec<i32>, gmodular: &'a GlobalModular) -> Self {
        Self { width, height, channel_shifts, gmodular }
    }
}

impl Bundle<(ModularParams<'_>, &Headers, &FrameHeader)> for Modular {
    type Error = crate::Error;

    fn parse<R: Read>(
        bitstream: &mut Bitstream<R>,
        (params, image_header, frame_header): (ModularParams<'_>, &Headers, &FrameHeader),
    ) -> std::result::Result<Self, Self::Error> {
        let mut header = read_bits!(bitstream, Bundle(ModularHeader))?;
        let ma_ctx = if header.use_global_tree {
            params.gmodular.make_context().ok_or(crate::Error::GlobalMaTreeNotAvailable)?
        } else {
            read_bits!(bitstream, Bundle(ma::MaConfig))?.into()
        };

        let mut channels = ModularChannels::from_params(&params);
        for transform in &mut header.transform {
            transform.or_default(&mut channels);
            transform.transform_channel_info(&mut channels)?;
        }

        Ok(Self {
            header,
            ma_ctx,
            channels,
        })
    }
}

define_bundle! {
    #[derive(Debug)]
    pub struct ModularHeader error(crate::Error) {
        use_global_tree: ty(Bool),
        wp_params: ty(Bundle(predictor::WpHeader)),
        nb_transforms: ty(U32(0, 1, 2 + u(4), 18 + u(8))),
        transform: ty(Vec[Bundle(transform::TransformInfo)]; nb_transforms),
    }
}

#[derive(Debug)]
struct ModularChannels {
    info: Vec<ModularChannelInfo>,
    nb_meta_channels: u32,
}

impl ModularChannels {
    fn from_params(params: &ModularParams<'_>) -> Self {
        let width = params.width;
        let height = params.height;
        let info = params.channel_shifts.iter()
            .map(|&shift| ModularChannelInfo::new(width, height, shift))
            .collect();
        Self {
            info,
            nb_meta_channels: 0,
        }
    }
}

#[derive(Debug, Clone)]
struct ModularChannelInfo {
    width: u32,
    height: u32,
    hshift: i32,
    vshift: i32,
}

impl ModularChannelInfo {
    fn new(width: u32, height: u32, shift: i32) -> Self {
        Self {
            width,
            height,
            hshift: shift,
            vshift: shift,
        }
    }
}
