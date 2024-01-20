use jxl_bitstream::{define_bundle, read_bits, Bitstream, Bundle};
use jxl_grid::AllocTracker;
use jxl_image::ImageHeader;
use jxl_modular::{
    ChannelShift, MaConfig, MaConfigParams, Modular, ModularChannelParams, ModularParams, Sample,
};
use jxl_vardct::{HfBlockContext, LfChannelCorrelation, LfChannelDequantization, Quantizer};

use crate::{header::Encoding, FrameHeader, Result};

use super::{NoiseParameters, Patches, Splines};

#[derive(Debug)]
pub struct LfGlobal<S: Sample> {
    pub patches: Option<Patches>,
    pub splines: Option<Splines>,
    pub noise: Option<NoiseParameters>,
    pub lf_dequant: LfChannelDequantization,
    pub vardct: Option<LfGlobalVarDct>,
    pub gmodular: GlobalModular<S>,
}

#[derive(Debug, Clone, Copy)]
pub struct LfGlobalParams<'a, 'b> {
    pub image_header: &'a ImageHeader,
    pub frame_header: &'a FrameHeader,
    pub tracker: Option<&'b AllocTracker>,
    pub allow_partial: bool,
}

impl<'a, 'b> LfGlobalParams<'a, 'b> {
    pub fn new(
        image_header: &'a ImageHeader,
        frame_header: &'a FrameHeader,
        tracker: Option<&'b AllocTracker>,
        allow_partial: bool,
    ) -> Self {
        Self {
            image_header,
            frame_header,
            tracker,
            allow_partial,
        }
    }
}

impl<S: Sample> Bundle<LfGlobalParams<'_, '_>> for LfGlobal<S> {
    type Error = crate::Error;

    fn parse(bitstream: &mut Bitstream, params: LfGlobalParams) -> Result<Self> {
        let LfGlobalParams {
            image_header,
            frame_header: header,
            ..
        } = params;
        let image_size = (header.width * header.height) as u64;

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

        if let Some(splines) = &splines {
            let base_correlation_xb = vardct.as_ref().map(|vardct| {
                let lf_chan_corr = &vardct.lf_chan_corr;
                (
                    lf_chan_corr.base_correlation_x,
                    lf_chan_corr.base_correlation_b,
                )
            });
            let estimated_area = splines.estimate_area(base_correlation_xb);

            // Maximum total_estimated_area_reached for Level 10
            let max_estimated_area = (1u64 << 42).min(1024 * image_size + (1u64 << 32));
            if estimated_area > max_estimated_area {
                tracing::error!(
                    estimated_area,
                    max_estimated_area,
                    "Too large estimated area for splines"
                );
                return Err(jxl_bitstream::Error::ProfileConformance(
                    "too large estimated area for splines",
                )
                .into());
            }
            // Maximum total_estimated_area_reached for Level 5
            if estimated_area > (1u64 << 30).min(8 * image_size + (1u64 << 25)) {
                tracing::warn!(
                    "Large estimated_area of splines, expect slower decoding: {}",
                    estimated_area
                );
            }
        }

        let gmodular = read_bits!(bitstream, Bundle(GlobalModular::<S>), params)?;

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
pub struct GlobalModular<S: Sample> {
    pub ma_config: Option<MaConfig>,
    pub modular: Modular<S>,
    extra_channel_from: usize,
}

impl<S: Sample> GlobalModular<S> {
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

impl<S: Sample> Bundle<LfGlobalParams<'_, '_>> for GlobalModular<S> {
    type Error = crate::Error;

    fn parse(bitstream: &mut Bitstream, params: LfGlobalParams) -> Result<Self> {
        let LfGlobalParams {
            image_header,
            frame_header: header,
            tracker,
            allow_partial,
        } = params;
        let span = tracing::span!(tracing::Level::TRACE, "Decode GlobalModular");
        let _guard = span.enter();

        let num_channels = (image_header.metadata.encoded_color_channels()
            + image_header.metadata.ec_info.len()) as u64;
        let max_global_ma_nodes =
            1024 + header.width as u64 * header.height as u64 * num_channels / 16;
        let max_global_ma_nodes = (1 << 22).min(max_global_ma_nodes) as usize;
        let ma_config_params = MaConfigParams {
            tracker: params.tracker,
            node_limit: max_global_ma_nodes,
        };
        let ma_config = bitstream
            .read_bool()?
            .then(|| bitstream.read_bundle_with_ctx::<MaConfig, _>(ma_config_params))
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

        let group_dim = header.group_dim();
        let modular_params = ModularParams::with_channels(
            group_dim,
            image_header.metadata.bit_depth.bits_per_sample(),
            shifts,
            ma_config.as_ref(),
            tracker,
        );
        let mut modular = read_bits!(bitstream, Bundle(Modular::<S>), modular_params)?;
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
