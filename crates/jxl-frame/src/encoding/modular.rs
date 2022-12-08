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
    channels: Vec<ModularChannelInfo>,
}

#[derive(Debug, Copy, Clone)]
pub struct ModularParams<'a> {
    pub channels: u32,
    pub width: u32,
    pub height: u32,
    pub gmodular: &'a GlobalModular,
}

impl<'a> ModularParams<'a> {
    pub fn new(channels: u32, width: u32, height: u32, gmodular: &'a GlobalModular) -> Self {
        Self { channels, width, height, gmodular }
    }
}

impl Bundle<(ModularParams<'_>, &Headers, &FrameHeader)> for Modular {
    type Error = crate::Error;

    fn parse<R: Read>(
        bitstream: &mut Bitstream<R>,
        (params, image_header, frame_header): (ModularParams<'_>, &Headers, &FrameHeader),
    ) -> std::result::Result<Self, Self::Error> {
        let header = read_bits!(bitstream, Bundle(ModularHeader))?;
        let ma_ctx = if header.use_global_tree {
            params.gmodular.make_context().ok_or(crate::Error::GlobalMaTreeNotAvailable)?
        } else {
            read_bits!(bitstream, Bundle(ma::MaConfig))?.into()
        };
        // TODO
        let channels = Vec::new();

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
struct ModularChannelInfo {
}
