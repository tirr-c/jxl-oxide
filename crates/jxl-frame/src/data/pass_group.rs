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

#[derive(Debug)]
pub struct PassGroupParams<'frame, 'buf, 'g> {
    pub frame_header: &'frame FrameHeader,
    pub lf_group: &'frame LfGroup,
    pub pass_idx: u32,
    pub group_idx: u32,
    pub shift: Option<(i32, i32)>,
    pub gmodular: &'g mut GlobalModular,
    pub vardct: Option<PassGroupParamsVardct<'frame, 'buf, 'g>>,
    pub allow_partial: bool,
}

#[derive(Debug)]
pub struct PassGroupParamsVardct<'frame, 'buf, 'g> {
    pub lf_vardct: &'frame LfGlobalVarDct,
    pub hf_global: &'frame HfGlobal,
    pub hf_coeff_output: &'buf mut [CutGrid<'g, f32>; 3],
}

pub fn decode_pass_group(
    bitstream: &mut Bitstream,
    params: PassGroupParams,
) -> Result<()> {
    let PassGroupParams {
        frame_header,
        lf_group,
        pass_idx,
        group_idx,
        shift,
        gmodular,
        vardct,
        allow_partial,
    } = params;

    if let (Some(PassGroupParamsVardct { lf_vardct, hf_global, hf_coeff_output }), Some(hf_meta)) = (vardct, &lf_group.hf_meta) {
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
        let block_info = block_info.subgrid(block_left..(block_left + block_width), block_top..(block_top + block_height));
        let lf_quant: Option<[_; 3]> = lf_group.lf_coeff.as_ref().map(|lf_coeff| {
            let lf_quant_channels = lf_coeff.lf_quant.image().channel_data();
            std::array::from_fn(|idx| {
                let lf_quant = lf_quant_channels[[1, 0, 2][idx]].as_simple().unwrap();
                let shift = ChannelShift::from_jpeg_upsampling(jpeg_upsampling, idx);

                let block_left = block_left >> shift.hshift();
                let block_top = block_top >> shift.vshift();
                let (block_width, block_height) = shift.shift_size((block_width as u32, block_height as u32));
                lf_quant.subgrid(block_left..(block_left + block_width as usize), block_top..(block_top + block_height as usize))
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

        match write_hf_coeff(bitstream, params, hf_coeff_output) {
            Err(e) if e.unexpected_eof() && allow_partial => {
                tracing::debug!("Partially decoded HfCoeff");
                return Ok(());
            },
            Err(e) => return Err(e.into()),
            Ok(_) => {},
        };
    }

    if let Some((minshift, maxshift)) = shift {
        let modular_params = gmodular.modular.make_subimage_params_pass_group(gmodular.ma_config.as_ref(), group_idx, minshift, maxshift);
        let mut modular = Modular::parse(bitstream, modular_params)?;
        modular.decode_image(bitstream, 1 + 3 * frame_header.num_lf_groups() + 17 + pass_idx * frame_header.num_groups()? + group_idx, allow_partial)?;
        modular.inverse_transform();
        gmodular.modular.copy_from_modular(modular);
    }

    Ok(())
}
