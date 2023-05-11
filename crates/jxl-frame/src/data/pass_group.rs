use jxl_bitstream::{read_bits, Bitstream, Bundle};
use jxl_modular::{ChannelShift, Modular};
use jxl_vardct::{HfCoeff, HfCoeffParams};

use crate::{FrameHeader, Result};
use super::{
    GlobalModular,
    LfGlobal,
    LfGlobalVarDct,
    LfGroup,
    HfGlobal,
};

#[derive(Debug, Clone, Copy)]
pub struct PassGroupParams<'a> {
    frame_header: &'a FrameHeader,
    gmodular: &'a GlobalModular,
    lf_vardct: Option<&'a LfGlobalVarDct>,
    lf_group: &'a LfGroup,
    hf_global: Option<&'a HfGlobal>,
    pass_idx: u32,
    group_idx: u32,
    shift: Option<(i32, i32)>,
}

impl<'a> PassGroupParams<'a> {
    pub fn new(
        frame_header: &'a FrameHeader,
        lf_global: &'a LfGlobal,
        lf_group: &'a LfGroup,
        hf_global: Option<&'a HfGlobal>,
        pass_idx: u32,
        group_idx: u32,
        shift: Option<(i32, i32)>,
    ) -> Self {
        Self {
            frame_header,
            gmodular: &lf_global.gmodular,
            lf_vardct: lf_global.vardct.as_ref(),
            lf_group,
            hf_global,
            pass_idx,
            group_idx,
            shift,
        }
    }
}

#[derive(Debug)]
pub struct PassGroup {
    pub hf_coeff: Option<HfCoeff>,
    pub modular: Modular,
}

impl Bundle<PassGroupParams<'_>> for PassGroup {
    type Error = crate::Error;

    fn parse<R: std::io::Read>(bitstream: &mut Bitstream<R>, params: PassGroupParams<'_>) -> Result<Self> {
        let PassGroupParams {
            frame_header,
            gmodular,
            lf_vardct,
            lf_group,
            hf_global,
            pass_idx,
            group_idx,
            shift,
        } = params;

        let hf_coeff = lf_vardct
            .zip(lf_group.hf_meta.as_ref())
            .zip(hf_global)
            .map(|((lf_vardct, hf_meta), hf_global)| {
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
                HfCoeff::parse(bitstream, params)
            })
            .transpose()?;

        let modular = if let Some((minshift, maxshift)) = shift {
            let modular_params = gmodular.modular.make_subimage_params_pass_group(gmodular.ma_config.as_ref(), group_idx, minshift, maxshift);
            let mut modular = read_bits!(bitstream, Bundle(Modular), modular_params)?;
            modular.decode_image(bitstream, 1 + 3 * frame_header.num_lf_groups() + 17 + pass_idx * frame_header.num_groups() + group_idx)?;
            modular.inverse_transform();
            modular
        } else {
            Modular::empty()
        };

        Ok(Self {
            hf_coeff,
            modular,
        })
    }
}
