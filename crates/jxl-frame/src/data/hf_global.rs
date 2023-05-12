use jxl_bitstream::{Bitstream, Bundle};
use jxl_image::ImageMetadata;
use jxl_modular::MaConfig;
use jxl_vardct::{
    DequantMatrixSet,
    DequantMatrixSetParams,
    HfBlockContext,
    HfPass,
    HfPassParams,
};

use crate::{FrameHeader, Result};
use super::LfGlobal;

#[derive(Debug, Copy, Clone)]
pub struct HfGlobalParams<'a> {
    metadata: &'a ImageMetadata,
    frame_header: &'a FrameHeader,
    ma_config: Option<&'a MaConfig>,
    hf_block_ctx: &'a HfBlockContext,
}

impl<'a> HfGlobalParams<'a> {
    pub fn new(metadata: &'a ImageMetadata, frame_header: &'a FrameHeader, lf_global: &'a LfGlobal) -> Self {
        let Some(lf_vardct) = &lf_global.vardct else { panic!("VarDCT not initialized") };
        Self {
            metadata,
            frame_header,
            ma_config: lf_global.gmodular.ma_config.as_ref(),
            hf_block_ctx: &lf_vardct.hf_block_ctx,
        }
    }
}

#[derive(Debug)]
pub struct HfGlobal {
    pub dequant_matrices: DequantMatrixSet,
    pub num_hf_presets: u32,
    pub hf_passes: Vec<HfPass>,
}

impl Bundle<HfGlobalParams<'_>> for HfGlobal {
    type Error = crate::Error;

    fn parse<R: std::io::Read>(bitstream: &mut Bitstream<R>, params: HfGlobalParams<'_>) -> Result<Self> {
        let HfGlobalParams { metadata, frame_header, ma_config, hf_block_ctx } = params;
        let dequant_matrix_params = DequantMatrixSetParams::new(
            metadata.bit_depth.bits_per_sample(),
            frame_header.num_lf_groups(),
            ma_config,
        );
        let dequant_matrices = DequantMatrixSet::parse(bitstream, dequant_matrix_params)?;

        let num_groups = frame_header.num_groups();
        let num_hf_presets = bitstream.read_bits(num_groups.next_power_of_two().trailing_zeros())? + 1;

        let hf_pass_params = HfPassParams::new(hf_block_ctx, num_hf_presets);
        let hf_passes = std::iter::repeat_with(|| HfPass::parse(bitstream, hf_pass_params))
            .take(frame_header.passes.num_passes as usize)
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(Self {
            dequant_matrices,
            num_hf_presets,
            hf_passes,
        })
    }
}
