use jxl_bitstream::{define_bundle, read_bits, Bitstream, Bundle};
use jxl_image::ImageHeader;
use jxl_modular::{ChannelShift, Modular, ModularParams, MaConfig, ModularChannelParams};
use jxl_vardct::{
    LfChannelDequantization,
    LfChannelCorrelation,
    HfBlockContext,
    Quantizer,
};

use crate::{
    FrameHeader,
    Result,
    header::Encoding,
};

use super::{
    Patches,
    Splines,
    NoiseParameters,
};

#[derive(Debug)]
pub struct LfGlobal {
    pub patches: Option<Patches>,
    pub splines: Option<Splines>,
    pub noise: Option<NoiseParameters>,
    pub lf_dequant: LfChannelDequantization,
    pub vardct: Option<LfGlobalVarDct>,
    pub gmodular: GlobalModular,
}

impl Bundle<(&ImageHeader, &FrameHeader)> for LfGlobal {
    type Error = crate::Error;

    fn parse<R: std::io::Read>(bitstream: &mut Bitstream<R>, (image_header, header): (&ImageHeader, &FrameHeader)) -> Result<Self> {
        let patches = header.flags.patches().then(|| {
            Patches::parse(bitstream, image_header)
        }).transpose()?;
        let splines = header.flags.splines().then(|| {
            Splines::parse(bitstream, header)
        }).transpose()?;
        let noise = header.flags.noise().then(|| {
            NoiseParameters::parse(bitstream, ())
        }).transpose()?;
        let lf_dequant = read_bits!(bitstream, Bundle(LfChannelDequantization))?;
        let vardct = (header.encoding == crate::header::Encoding::VarDct).then(|| {
            read_bits!(bitstream, Bundle(LfGlobalVarDct))
        }).transpose()?;
        let gmodular = read_bits!(bitstream, Bundle(GlobalModular), (image_header, header))?;

        Ok(Self {
            patches,
            splines,
            noise,
            lf_dequant,
            vardct,
            gmodular,
        })
    }
}

impl LfGlobal {
    pub(crate) fn apply_modular_inverse_transform(&mut self) {
        self.gmodular.modular.inverse_transform();
    }
}

define_bundle! {
    #[derive(Debug)]
    pub struct LfGlobalVarDct error(crate::Error) {
        pub quantizer: ty(Bundle(Quantizer)),
        pub hf_block_ctx: ty(Bundle(HfBlockContext)),
        pub lf_chan_corr: ty(Bundle(LfChannelCorrelation)),
    }
}

#[derive(Debug)]
pub struct GlobalModular {
    pub ma_config: Option<MaConfig>,
    pub modular: Modular,
    extra_channel_from: usize,
}

impl GlobalModular {
    pub fn ma_config(&self) -> Option<&MaConfig> {
        self.ma_config.as_ref()
    }

    pub fn extra_channel_from(&self) -> usize {
        self.extra_channel_from
    }
}

impl Bundle<(&ImageHeader, &FrameHeader)> for GlobalModular {
    type Error = crate::Error;

    fn parse<R: std::io::Read>(bitstream: &mut Bitstream<R>, (image_header, header): (&ImageHeader, &FrameHeader)) -> Result<Self> {
        let ma_config = bitstream.read_bool()?
            .then(|| read_bits!(bitstream, Bundle(MaConfig)))
            .transpose()?;
        let mut shifts = Vec::new();
        if header.encoding == Encoding::Modular {
            let width = header.color_sample_width();
            let height = header.color_sample_height();
            if header.do_ycbcr {
                // Cb, Y, Cr
                shifts.push(ModularChannelParams::jpeg(width, height, header.jpeg_upsampling, 0));
                shifts.push(ModularChannelParams::jpeg(width, height, header.jpeg_upsampling, 1));
                shifts.push(ModularChannelParams::jpeg(width, height, header.jpeg_upsampling, 2));
            } else {
                let channel_param = ModularChannelParams::new(width, height);
                let channels = image_header.metadata.encoded_color_channels();
                shifts.extend(std::iter::repeat(channel_param).take(channels));
            }
        }

        let extra_channel_from = shifts.len();

        for (&ec_upsampling, ec_info) in header.ec_upsampling.iter().zip(image_header.metadata.ec_info.iter()) {
            let width = header.sample_width(ec_upsampling);
            let height = header.sample_height(ec_upsampling);
            let shift = ChannelShift::from_shift(ec_info.dim_shift);
            shifts.push(ModularChannelParams::with_shift(width, height, shift));
        }

        let group_dim = header.group_dim();
        let modular_params = ModularParams::with_channels(
            group_dim,
            image_header.metadata.bit_depth.bits_per_sample(),
            shifts,
            ma_config.as_ref(),
        );
        let mut modular = read_bits!(bitstream, Bundle(Modular), modular_params)?;
        modular.decode_image_gmodular(bitstream)?;

        Ok(Self {
            ma_config,
            modular,
            extra_channel_from,
        })
    }
}
