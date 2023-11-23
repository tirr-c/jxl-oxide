use jxl_bitstream::{define_bundle, read_bits, Bitstream, Bundle};
use jxl_image::ImageHeader;
use jxl_modular::{ChannelShift, MaConfig, Modular, ModularChannelParams, ModularParams};
use jxl_vardct::{HfBlockContext, LfChannelCorrelation, LfChannelDequantization, Quantizer};

use crate::{header::Encoding, FrameHeader, Result};

use super::{NoiseParameters, Patches, Splines};

#[derive(Debug)]
pub struct LfGlobal {
    pub patches: Option<Patches>,
    pub splines: Option<Splines>,
    pub noise: Option<NoiseParameters>,
    pub lf_dequant: LfChannelDequantization,
    pub vardct: Option<LfGlobalVarDct>,
    pub gmodular: GlobalModular,
}

#[derive(Debug, Clone, Copy)]
pub struct LfGlobalParams<'a> {
    pub image_header: &'a ImageHeader,
    pub frame_header: &'a FrameHeader,
    pub allow_partial: bool,
}

impl<'a> LfGlobalParams<'a> {
    pub fn new(
        image_header: &'a ImageHeader,
        frame_header: &'a FrameHeader,
        allow_partial: bool,
    ) -> Self {
        Self {
            image_header,
            frame_header,
            allow_partial,
        }
    }
}

impl Bundle<LfGlobalParams<'_>> for LfGlobal {
    type Error = crate::Error;

    fn parse(bitstream: &mut Bitstream, params: LfGlobalParams<'_>) -> Result<Self> {
        let LfGlobalParams {
            image_header,
            frame_header: header,
            ..
        } = params;
        let patches = header
            .flags
            .patches()
            .then(|| -> Result<_> {
                let span = tracing::span!(tracing::Level::TRACE, "Decode Patches");
                let _guard = span.enter();

                let patches = Patches::parse(bitstream, (image_header, header))?;
                let it = patches
                    .patches
                    .iter()
                    .flat_map(|patch| &patch.patch_targets)
                    .flat_map(|target| &target.blending);
                for blending_info in it {
                    if blending_info.mode.use_alpha()
                        && blending_info.alpha_channel as usize
                            >= image_header.metadata.ec_info.len()
                    {
                        return Err(jxl_bitstream::Error::ValidationFailed(
                            "blending_info.alpha_channel out of range",
                        )
                        .into());
                    }
                }
                Ok(patches)
            })
            .transpose()?;
        let splines = header
            .flags
            .splines()
            .then(|| {
                let span = tracing::span!(tracing::Level::TRACE, "Decode Splines");
                let _guard = span.enter();

                Splines::parse(bitstream, header)
            })
            .transpose()?;
        let noise = header
            .flags
            .noise()
            .then(|| {
                let span = tracing::span!(tracing::Level::TRACE, "Decode Noise");
                let _guard = span.enter();

                NoiseParameters::parse(bitstream, ())
            })
            .transpose()?;
        let lf_dequant = read_bits!(bitstream, Bundle(LfChannelDequantization))?;
        let vardct = (header.encoding == crate::header::Encoding::VarDct)
            .then(|| read_bits!(bitstream, Bundle(LfGlobalVarDct)))
            .transpose()?;
        let gmodular = read_bits!(bitstream, Bundle(GlobalModular), params)?;

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
    pub fn try_clone(&self) -> Result<Self> {
        Ok(Self {
            ma_config: self.ma_config.clone(),
            modular: self.modular.try_clone()?,
            extra_channel_from: self.extra_channel_from,
        })
    }

    pub fn ma_config(&self) -> Option<&MaConfig> {
        self.ma_config.as_ref()
    }

    pub fn extra_channel_from(&self) -> usize {
        self.extra_channel_from
    }
}

impl Bundle<LfGlobalParams<'_>> for GlobalModular {
    type Error = crate::Error;

    fn parse(bitstream: &mut Bitstream, params: LfGlobalParams<'_>) -> Result<Self> {
        let LfGlobalParams {
            image_header,
            frame_header: header,
            allow_partial,
        } = params;
        let span = tracing::span!(tracing::Level::TRACE, "Decode GlobalModular");
        let _guard = span.enter();

        let ma_config = bitstream
            .read_bool()?
            .then(|| read_bits!(bitstream, Bundle(MaConfig)))
            .transpose()?;

        let mut shifts = Vec::new();
        if header.encoding == Encoding::Modular {
            let width = header.color_sample_width();
            let height = header.color_sample_height();
            if header.do_ycbcr {
                // Cb, Y, Cr
                shifts.push(ModularChannelParams::jpeg(
                    width,
                    height,
                    header.jpeg_upsampling,
                    0,
                ));
                shifts.push(ModularChannelParams::jpeg(
                    width,
                    height,
                    header.jpeg_upsampling,
                    1,
                ));
                shifts.push(ModularChannelParams::jpeg(
                    width,
                    height,
                    header.jpeg_upsampling,
                    2,
                ));
            } else {
                let channel_param = ModularChannelParams::new(width, height);
                let channels = image_header.metadata.encoded_color_channels();
                shifts.extend(std::iter::repeat(channel_param).take(channels));
            }
        }

        let extra_channel_from = shifts.len();

        for (&ec_upsampling, ec_info) in header
            .ec_upsampling
            .iter()
            .zip(image_header.metadata.ec_info.iter())
        {
            let width = header.sample_width(ec_upsampling);
            let height = header.sample_height(ec_upsampling);
            let shift = ChannelShift::from_shift(ec_info.dim_shift);
            shifts.push(ModularChannelParams::with_shift(width, height, shift));
        }

        if let Some(ma_config) = &ma_config {
            let num_channels = (image_header.metadata.encoded_color_channels()
                + image_header.metadata.ec_info.len()) as u64;
            let max_global_ma_nodes =
                1024 + header.width as u64 * header.height as u64 * num_channels / 16;
            let max_global_ma_nodes = (1 << 22).min(max_global_ma_nodes) as usize;
            let global_ma_nodes = ma_config.num_tree_nodes();
            if global_ma_nodes > max_global_ma_nodes {
                tracing::error!(
                    global_ma_nodes,
                    max_global_ma_nodes,
                    "Too many global MA tree nodes"
                );
                return Err(jxl_bitstream::Error::ProfileConformance(
                    "too many global MA tree nodes",
                )
                .into());
            }
        }

        let group_dim = header.group_dim();
        let modular_params = ModularParams::with_channels(
            group_dim,
            image_header.metadata.bit_depth.bits_per_sample(),
            shifts,
            ma_config.as_ref(),
        );
        let mut modular = read_bits!(bitstream, Bundle(Modular), modular_params)?;
        if let Some(image) = modular.image_mut() {
            let mut gmodular = image.prepare_gmodular()?;
            gmodular.decode(bitstream, 0, allow_partial)?;
        }

        Ok(Self {
            ma_config,
            modular,
            extra_channel_from,
        })
    }
}
