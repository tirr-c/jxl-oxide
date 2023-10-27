use jxl_bitstream::{read_bits, Bitstream, Bundle};
use jxl_modular::Modular;
use jxl_vardct::{HfMetadata, HfMetadataParams, Quantizer, LfCoeff, LfCoeffParams};

use crate::{
    FrameHeader,
    GlobalModular,
    LfGlobal,
    Result,
    filter::EdgePreservingFilter,
    header::Encoding,
};

#[derive(Debug, Clone, Copy)]
pub struct LfGroupParams<'a> {
    frame_header: &'a FrameHeader,
    quantizer: Option<&'a Quantizer>,
    gmodular: &'a GlobalModular,
    lf_group_idx: u32,
    allow_partial: bool,
}

impl<'a> LfGroupParams<'a> {
    pub fn new(frame_header: &'a FrameHeader, lf_global: &'a LfGlobal, lf_group_idx: u32, allow_partial: bool) -> Self {
        Self {
            frame_header,
            quantizer: lf_global.vardct.as_ref().map(|vardct| &vardct.quantizer),
            gmodular: &lf_global.gmodular,
            lf_group_idx,
            allow_partial,
        }
    }
}

#[derive(Debug)]
pub struct LfGroup {
    pub lf_coeff: Option<LfCoeff>,
    pub mlf_group: Modular,
    pub hf_meta: Option<HfMetadata>,
}

impl Bundle<LfGroupParams<'_>> for LfGroup {
    type Error = crate::Error;

    fn parse(bitstream: &mut Bitstream, params: LfGroupParams<'_>) -> Result<Self> {
        let LfGroupParams { frame_header, gmodular, lf_group_idx, allow_partial, .. } = params;
        let (lf_width, lf_height) = frame_header.lf_group_size_for(lf_group_idx);

        let lf_coeff = (frame_header.encoding == Encoding::VarDct && !frame_header.flags.use_lf_frame())
            .then(|| {
                let lf_coeff_params = LfCoeffParams {
                    lf_group_idx,
                    lf_width,
                    lf_height,
                    jpeg_upsampling: frame_header.jpeg_upsampling,
                    bits_per_sample: frame_header.bit_depth.bits_per_sample(),
                    global_ma_config: gmodular.ma_config(),
                    allow_partial,
                };
                LfCoeff::parse(bitstream, lf_coeff_params)
            })
            .transpose()?;

        if let Some(lf_coeff_inner) = &lf_coeff {
            if lf_coeff_inner.lf_quant.is_partial() {
                return Ok(Self { lf_coeff, mlf_group: Modular::empty(), hf_meta: None });
            }
        }

        let mlf_group_params = gmodular.modular
            .make_subimage_params_lf_group(gmodular.ma_config.as_ref(), lf_group_idx);
        let mut mlf_group = read_bits!(bitstream, Bundle(Modular), mlf_group_params.clone())?;
        mlf_group.decode_image(bitstream, 1 + frame_header.num_lf_groups() + lf_group_idx, allow_partial)?;
        mlf_group.inverse_transform();

        let hf_meta = (frame_header.encoding == Encoding::VarDct && !mlf_group.is_partial())
            .then(|| {
                let hf_meta_params = HfMetadataParams {
                    num_lf_groups: frame_header.num_lf_groups(),
                    lf_group_idx,
                    lf_width,
                    lf_height,
                    jpeg_upsampling: frame_header.jpeg_upsampling,
                    bits_per_sample: frame_header.bit_depth.bits_per_sample(),
                    global_ma_config: gmodular.ma_config(),
                    epf: match &frame_header.restoration_filter.epf {
                        EdgePreservingFilter::Disabled => None,
                        EdgePreservingFilter::Enabled { sharp_lut, sigma, .. } => {
                            Some((sigma.quant_mul, *sharp_lut))
                        },
                    },
                    quantizer_global_scale: params.quantizer.unwrap().global_scale,
                };
                HfMetadata::parse(bitstream, hf_meta_params)
            })
            .transpose();
        match hf_meta {
            Err(e) if e.unexpected_eof() && allow_partial => {
                tracing::debug!("Decoded partial HfMeta");
                Ok(Self { lf_coeff, mlf_group, hf_meta: None })
            },
            Err(e) => Err(e.into()),
            Ok(hf_meta) => Ok(Self { lf_coeff, mlf_group, hf_meta }),
        }
    }
}
