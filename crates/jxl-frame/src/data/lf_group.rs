use jxl_bitstream::{Bitstream, Bundle};
use jxl_grid::AllocTracker;
use jxl_modular::{image::TransformedModularSubimage, MaConfig};
use jxl_vardct::{HfMetadata, HfMetadataParams, LfCoeff, LfCoeffParams, Quantizer};

use crate::{filter::EdgePreservingFilter, header::Encoding, FrameHeader, Result};

#[derive(Debug)]
pub struct LfGroupParams<'a, 'dest, 'tracker> {
    pub frame_header: &'a FrameHeader,
    pub quantizer: Option<&'a Quantizer>,
    pub global_ma_config: Option<&'a MaConfig>,
    pub mlf_group: Option<TransformedModularSubimage<'dest>>,
    pub lf_group_idx: u32,
    pub allow_partial: bool,
    pub tracker: Option<&'tracker AllocTracker>,
    pub pool: &'a jxl_threadpool::JxlThreadPool,
}

#[derive(Debug)]
pub struct LfGroup {
    pub lf_coeff: Option<LfCoeff>,
    pub hf_meta: Option<HfMetadata>,
    pub partial: bool,
}

impl Bundle<LfGroupParams<'_, '_, '_>> for LfGroup {
    type Error = crate::Error;

    fn parse(bitstream: &mut Bitstream, params: LfGroupParams) -> Result<Self> {
        let LfGroupParams {
            frame_header,
            global_ma_config,
            mlf_group,
            lf_group_idx,
            allow_partial,
            tracker,
            pool,
            ..
        } = params;
        let (lf_width, lf_height) = frame_header.lf_group_size_for(lf_group_idx);

        let lf_coeff = (frame_header.encoding == Encoding::VarDct
            && !frame_header.flags.use_lf_frame())
        .then(|| {
            let lf_coeff_params = LfCoeffParams {
                lf_group_idx,
                lf_width,
                lf_height,
                jpeg_upsampling: frame_header.jpeg_upsampling,
                bits_per_sample: frame_header.bit_depth.bits_per_sample(),
                global_ma_config,
                allow_partial,
                tracker,
                pool,
            };
            LfCoeff::parse(bitstream, lf_coeff_params)
        })
        .transpose()?;

        if let Some(lf_coeff_inner) = &lf_coeff {
            if lf_coeff_inner.partial {
                return Ok(Self {
                    lf_coeff,
                    hf_meta: None,
                    partial: true,
                });
            }
        }

        let mut is_mlf_complete = true;
        if let Some(image) = mlf_group {
            let mut subimage = image.recursive(bitstream, global_ma_config, tracker)?;
            let mut subimage = subimage.prepare_subimage()?;
            subimage.decode(
                bitstream,
                1 + frame_header.num_lf_groups() + lf_group_idx,
                allow_partial,
            )?;
            is_mlf_complete = subimage.finish(pool);
        }

        let hf_meta = (frame_header.encoding == Encoding::VarDct && is_mlf_complete)
            .then(|| {
                let hf_meta_params = HfMetadataParams {
                    num_lf_groups: frame_header.num_lf_groups(),
                    lf_group_idx,
                    lf_width,
                    lf_height,
                    jpeg_upsampling: frame_header.jpeg_upsampling,
                    bits_per_sample: frame_header.bit_depth.bits_per_sample(),
                    global_ma_config,
                    epf: match &frame_header.restoration_filter.epf {
                        EdgePreservingFilter::Disabled => None,
                        EdgePreservingFilter::Enabled {
                            sharp_lut, sigma, ..
                        } => Some((sigma.quant_mul, *sharp_lut)),
                    },
                    quantizer_global_scale: params.quantizer.unwrap().global_scale,
                    tracker,
                    pool,
                };
                HfMetadata::parse(bitstream, hf_meta_params)
            })
            .transpose();
        match hf_meta {
            Err(e) if e.unexpected_eof() && allow_partial => {
                tracing::debug!("Decoded partial HfMeta");
                Ok(Self {
                    lf_coeff,
                    hf_meta: None,
                    partial: true,
                })
            }
            Err(e) => Err(e.into()),
            Ok(hf_meta) => Ok(Self {
                lf_coeff,
                hf_meta,
                partial: !is_mlf_complete,
            }),
        }
    }
}
