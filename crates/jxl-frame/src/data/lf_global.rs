use jxl_bitstream::{define_bundle, read_bits, Bitstream, Bundle};
use jxl_image::Headers;
use jxl_modular::{ChannelShift, Modular, ModularParams, MaConfig};
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

impl Bundle<(&Headers, &FrameHeader)> for LfGlobal {
    type Error = crate::Error;

    fn parse<R: std::io::Read>(bitstream: &mut Bitstream<R>, (image_header, header): (&Headers, &FrameHeader)) -> Result<Self> {
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

impl Bundle<(&Headers, &FrameHeader)> for GlobalModular {
    type Error = crate::Error;

    fn parse<R: std::io::Read>(bitstream: &mut Bitstream<R>, (image_header, header): (&Headers, &FrameHeader)) -> Result<Self> {
        let ma_config = bitstream.read_bool()?
            .then(|| read_bits!(bitstream, Bundle(MaConfig)))
            .transpose()?;
        let mut shifts = Vec::new();
        if header.encoding == Encoding::Modular {
            if header.do_ycbcr {
                // Cb, Y, Cr
                shifts.push(ChannelShift::from_jpeg_upsampling(header.jpeg_upsampling, 0));
                shifts.push(ChannelShift::from_jpeg_upsampling(header.jpeg_upsampling, 1));
                shifts.push(ChannelShift::from_jpeg_upsampling(header.jpeg_upsampling, 2));
            } else {
                let shift = ChannelShift::from_shift(0);
                let channels = header.gmodular_extra_channel_from as usize;
                shifts.extend(std::iter::repeat(shift).take(channels));
            }
        }

        let extra_channel_from = shifts.len();

        for (&ec_upsampling, ec_info) in header.ec_upsampling.iter().zip(image_header.metadata.ec_info.iter()) {
            let dim_shift = ec_info.dim_shift;
            let shift = ChannelShift::from_upsampling_factor_and_shift(ec_upsampling, dim_shift);
            shifts.push(shift);
        }

        let group_dim = header.group_dim();
        let modular_params = ModularParams::new(
            header.sample_width(),
            header.sample_height(),
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
