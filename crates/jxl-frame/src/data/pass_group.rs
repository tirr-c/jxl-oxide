use jxl_bitstream::{Bitstream, Bundle};
use jxl_grid::CutGrid;
use jxl_modular::{ChannelShift, Modular};
use jxl_vardct::{HfCoeffParams, write_hf_coeff};

use crate::{FrameHeader, Result};
use super::{
    GlobalModular,
    LfGlobalVarDct,
    LfGroup,
    HfGlobal,
};

pub fn decode_pass_group<R: std::io::Read>(
    bitstream: &mut Bitstream<R>,
    frame_header: &FrameHeader,
    lf_vardct: Option<&LfGlobalVarDct>,
    lf_group: &LfGroup,
    hf_global: Option<&HfGlobal>,
    pass_idx: u32,
    group_idx: u32,
    shift: Option<(i32, i32)>,
    gmodular: &mut GlobalModular,
    hf_coeff_output: Option<&mut [CutGrid<'_, f32>; 3]>,
) -> Result<()> {
    if let (Some(lf_vardct), Some(hf_meta), Some(hf_global), Some(hf_coeff_output)) = (lf_vardct, &lf_group.hf_meta, hf_global, hf_coeff_output) {
        let hf_pass = &hf_global.hf_passes[pass_idx as usize];
        let coeff_shift = frame_header.passes.shift.get(pass_idx as usize)
            .copied()
            .unwrap_or(0);

        let group_col = group_idx % frame_header.groups_per_row();
        let group_row = group_idx / frame_header.groups_per_row();
        let lf_col = (group_col % 8) as usize;
        let lf_row = (group_row % 8) as usize;
        let group_dim_blocks = (frame_header.group_dim() / 8) as usize;

        let block_info = &hf_meta.block_info;

        let block_left = lf_col * group_dim_blocks;
        let block_top = lf_row * group_dim_blocks;
        let block_width = (block_info.width() - block_left).min(group_dim_blocks);
        let block_height = (block_info.height() - block_top).min(group_dim_blocks);

        let jpeg_upsampling = frame_header.jpeg_upsampling;
        let block_info = block_info.subgrid(block_left, block_top, block_width, block_height);
        let lf_quant: Option<[_; 3]> = lf_group.lf_coeff.as_ref().map(|lf_coeff| {
            let lf_quant_channels = lf_coeff.lf_quant.image().channel_data();
            std::array::from_fn(|idx| {
                let lf_quant = &lf_quant_channels[[1, 0, 2][idx]];
                let shift = ChannelShift::from_jpeg_upsampling(jpeg_upsampling, idx);

                let block_left = block_left >> shift.hshift();
                let block_top = block_top >> shift.vshift();
                let (block_width, block_height) = shift.shift_size((block_width as u32, block_height as u32));
                lf_quant.subgrid(block_left, block_top, block_width as usize, block_height as usize)
            })
        });

        let params = HfCoeffParams {
            num_hf_presets: hf_global.num_hf_presets,
            hf_block_ctx: &lf_vardct.hf_block_ctx,
            block_info,
            jpeg_upsampling,
            lf_quant,
            hf_pass,
            coeff_shift,
        };

        write_hf_coeff(bitstream, params, hf_coeff_output)?;
    }

    if let Some((minshift, maxshift)) = shift {
        let modular_params = gmodular.modular.make_subimage_params_pass_group(gmodular.ma_config.as_ref(), group_idx, minshift, maxshift);
        let mut modular = Modular::parse(bitstream, modular_params)?;
        modular.decode_image(bitstream, 1 + 3 * frame_header.num_lf_groups() + 17 + pass_idx * frame_header.num_groups() + group_idx)?;
        modular.inverse_transform();
        gmodular.modular.copy_from_modular(modular);
    }

    Ok(())
}
