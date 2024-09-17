use jxl_bitstream::Bitstream;
use jxl_grid::AllocTracker;
use jxl_image::ImageMetadata;
use jxl_modular::{MaConfig, Sample};
use jxl_oxide_common::Bundle;
use jxl_threadpool::JxlThreadPool;
use jxl_vardct::{DequantMatrixSet, DequantMatrixSetParams, HfBlockContext, HfPass, HfPassParams};

use super::LfGlobal;
use crate::{FrameHeader, Result};

#[derive(Debug, Copy, Clone)]
pub struct HfGlobalParams<'a, 'b> {
    metadata: &'a ImageMetadata,
    frame_header: &'a FrameHeader,
    ma_config: Option<&'a MaConfig>,
    hf_block_ctx: &'a HfBlockContext,
    tracker: Option<&'b AllocTracker>,
    pool: &'a JxlThreadPool,
}

impl<'a, 'b> HfGlobalParams<'a, 'b> {
    pub fn new<S: Sample>(
        metadata: &'a ImageMetadata,
        frame_header: &'a FrameHeader,
        lf_global: &'a LfGlobal<S>,
        tracker: Option<&'b AllocTracker>,
        pool: &'a JxlThreadPool,
    ) -> Self {
        let Some(lf_vardct) = &lf_global.vardct else {
            panic!("VarDCT not initialized")
        };
        Self {
            metadata,
            frame_header,
            ma_config: lf_global.gmodular.ma_config.as_ref(),
            hf_block_ctx: &lf_vardct.hf_block_ctx,
            tracker,
            pool,
        }
    }
}

#[derive(Debug)]
pub struct HfGlobal {
    pub dequant_matrices: DequantMatrixSet,
    pub num_hf_presets: u32,
    pub hf_passes: Vec<HfPass>,
}

impl Bundle<HfGlobalParams<'_, '_>> for HfGlobal {
    type Error = crate::Error;

    fn parse(bitstream: &mut Bitstream, params: HfGlobalParams) -> Result<Self> {
        let HfGlobalParams {
            metadata,
            frame_header,
            ma_config,
            hf_block_ctx,
            tracker,
            pool,
        } = params;
        let dequant_matrix_params = DequantMatrixSetParams::new(
            metadata.bit_depth.bits_per_sample(),
            frame_header.num_lf_groups(),
            ma_config,
            tracker,
            pool,
        );
        let dequant_matrices = DequantMatrixSet::parse(bitstream, dequant_matrix_params)?;

        let num_groups = frame_header.num_groups();
        let num_hf_presets =
            bitstream.read_bits(num_groups.next_power_of_two().trailing_zeros() as usize)? + 1;

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
