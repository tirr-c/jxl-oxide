use std::collections::HashMap;

use jxl_bitstream::{read_bits, Bitstream, Bundle};
use jxl_grid::{Grid, Subgrid, SimpleGrid};
use jxl_modular::{ChannelShift, Modular};
use jxl_vardct::{HfBlockContext, HfPass, TransformType};

use crate::{FrameHeader, Result};
use super::{
    BlockInfo,
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

#[derive(Debug)]
struct HfCoeffParams<'a> {
    num_hf_presets: u32,
    hf_block_ctx: &'a HfBlockContext,
    block_info: Subgrid<'a, BlockInfo>,
    jpeg_upsampling: [u32; 3],
    lf_quant: Option<[Subgrid<'a, i32>; 3]>,
    hf_pass: &'a HfPass,
    coeff_shift: u32,
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
            .map(|((lf_vardct, hf_meta), hf_global)| -> Result<HfCoeff> {
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

#[derive(Debug, Clone)]
pub struct HfCoeff {
    pub data: HashMap<(usize, usize), CoeffData>,
}

impl HfCoeff {
    pub fn empty() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    pub fn merge(&mut self, other: &HfCoeff) {
        for (coord, other_data) in &other.data {
            if let Some(target_data) = self.data.get_mut(coord) {
                assert_eq!(target_data.dct_select, other_data.dct_select);
                for (target, v) in target_data.coeff.iter_mut().zip(other_data.coeff.iter()) {
                    assert_eq!(target.width(), v.width());
                    assert_eq!(target.height(), v.height());
                    for (target, v) in target.buf_mut().iter_mut().zip(v.buf()) {
                        *target += *v;
                    }
                }
            } else {
                self.data.insert(*coord, other_data.clone());
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct CoeffData {
    pub dct_select: TransformType,
    pub hf_mul: i32,
    pub coeff: [SimpleGrid<i32>; 3], // x, y, b
}

impl Bundle<HfCoeffParams<'_>> for HfCoeff {
    type Error = crate::Error;

    fn parse<R: std::io::Read>(bitstream: &mut Bitstream<R>, params: HfCoeffParams<'_>) -> Result<Self> {
        const COEFF_FREQ_CONTEXT: [u32; 64] = [
            0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
            15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22,
            23, 23, 23, 23, 24, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 26,
            27, 27, 27, 27, 28, 28, 28, 28, 29, 29, 29, 29, 30, 30, 30, 30,
        ];
        const COEFF_NUM_NONZERO_CONTEXT: [u32; 64] = [
            0,     0,  31,  62,  62,  93,  93,  93,  93, 123, 123, 123, 123,
            152, 152, 152, 152, 152, 152, 152, 152, 180, 180, 180, 180, 180,
            180, 180, 180, 180, 180, 180, 180, 206, 206, 206, 206, 206, 206,
            206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206,
            206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206,
        ];

        let mut data = HashMap::new();

        let HfCoeffParams {
            num_hf_presets,
            hf_block_ctx,
            block_info,
            jpeg_upsampling,
            lf_quant,
            hf_pass,
            coeff_shift,
        } = params;
        let mut dist = hf_pass.clone_decoder();
        let span = tracing::span!(tracing::Level::TRACE, "HfCoeff::parse");
        let _guard = span.enter();

        let HfBlockContext {
            qf_thresholds,
            lf_thresholds,
            block_ctx_map,
            num_block_clusters,
        } = hf_block_ctx;
        let upsampling_shifts: [_; 3] = std::array::from_fn(|idx| ChannelShift::from_jpeg_upsampling(jpeg_upsampling, idx));

        let hfp_bits = num_hf_presets.next_power_of_two().trailing_zeros();
        let hfp = bitstream.read_bits(hfp_bits)?;
        let ctx_offset = 495 * *num_block_clusters * hfp;

        dist.begin(bitstream)?;

        let width = block_info.width();
        let height = block_info.height();
        let mut non_zeros_grid = upsampling_shifts.map(|shift| {
            let (width, height) = shift.shift_size((width as u32, height as u32));
            Grid::new(width, height, width, height)
        });
        let predict_non_zeros = |grid: &Grid<u32>, x: usize, y: usize| {
            if x == 0 && y == 0 {
                32u32
            } else if x == 0 {
                *grid.get(x, y - 1).unwrap()
            } else if y == 0 {
                *grid.get(x - 1, y).unwrap()
            } else {
                (
                    *grid.get(x, y - 1).unwrap() +
                    *grid.get(x - 1, y).unwrap() +
                    1
                ) >> 1
            }
        };

        for y in 0..height {
            for x in 0..width {
                let BlockInfo::Data { dct_select, hf_mul: qf } = *block_info.get(x, y).unwrap() else {
                    continue;
                };
                let (w8, h8) = dct_select.dct_select_size();
                let coeff_size = dct_select.dequant_matrix_size();
                let num_blocks = w8 * h8;
                let order_id = dct_select.order_id();
                let qdc: Option<[_; 3]> = lf_quant.as_ref().map(|lf_quant| {
                    std::array::from_fn(|idx| {
                        let shift = upsampling_shifts[idx];
                        let x = x >> shift.hshift();
                        let y = y >> shift.vshift();
                        *lf_quant[idx].get(x, y).unwrap()
                    })
                });

                let hf_idx = {
                    let mut idx = 0usize;
                    for &threshold in qf_thresholds {
                        if qf > threshold as i32 {
                            idx += 1;
                        }
                    }
                    idx
                };
                let lf_idx = if let Some(qdc) = qdc {
                    let mut idx = 0usize;
                    for c in [0, 2, 1] {
                        let lf_thresholds = &lf_thresholds[c];
                        idx *= lf_thresholds.len() + 1;

                        let q = qdc[c];
                        for &threshold in lf_thresholds {
                            if q > threshold {
                                idx += 1;
                            }
                        }
                    }
                    idx
                } else {
                    0
                };
                let lf_idx_mul = (lf_thresholds[0].len() + 1) * (lf_thresholds[1].len() + 1) * (lf_thresholds[2].len() + 1);

                let mut coeff = [
                    SimpleGrid::new(coeff_size.0 as usize, coeff_size.1 as usize),
                    SimpleGrid::new(coeff_size.0 as usize, coeff_size.1 as usize),
                    SimpleGrid::new(coeff_size.0 as usize, coeff_size.1 as usize),
                ];
                for c in [1, 0, 2] { // y, x, b
                    let shift = upsampling_shifts[c];
                    let sx = x >> shift.hshift();
                    let sy = y >> shift.vshift();
                    if sx << shift.hshift() != x || sy << shift.vshift() != y {
                        continue;
                    }

                    let ch_idx = [1, 0, 2][c] * 13 + order_id as usize;
                    let idx = (ch_idx * (qf_thresholds.len() + 1) + hf_idx) * lf_idx_mul + lf_idx;
                    let block_ctx = block_ctx_map[idx] as u32;
                    let non_zeros_ctx = {
                        let predicted = predict_non_zeros(&non_zeros_grid[c], sx, sy).min(64);
                        let idx = if predicted >= 8 {
                            4 + predicted / 2
                        } else {
                            predicted
                        };
                        block_ctx + idx * num_block_clusters
                    };

                    let mut non_zeros = dist.read_varint(bitstream, ctx_offset + non_zeros_ctx)?;
                    let non_zeros_val = (non_zeros + num_blocks - 1) / num_blocks;
                    let non_zeros_grid = &mut non_zeros_grid[c];
                    for dy in 0..h8 as usize {
                        for dx in 0..w8 as usize {
                            non_zeros_grid.set(sx + dx, sy + dy, non_zeros_val);
                        }
                    }

                    let size = (w8 * 8) * (h8 * 8);
                    let coeff_grid = &mut coeff[c];
                    let mut prev_coeff = (non_zeros <= size / 16) as i32;
                    let order_it = hf_pass.order(order_id as usize, c);
                    for (idx, coeff_coord) in order_it.enumerate().skip(num_blocks as usize) {
                        if non_zeros == 0 {
                            break;
                        }

                        let idx = idx as u32;
                        let coeff_ctx = {
                            let prev = (prev_coeff != 0) as u32;
                            let non_zeros = (non_zeros + num_blocks - 1) / num_blocks;
                            let idx = idx / num_blocks;
                            (COEFF_NUM_NONZERO_CONTEXT[non_zeros as usize] + COEFF_FREQ_CONTEXT[idx as usize]) * 2 +
                                prev + block_ctx * 458 + 37 * num_block_clusters
                        };
                        let ucoeff = dist.read_varint(bitstream, ctx_offset + coeff_ctx)?;
                        let coeff = jxl_bitstream::unpack_signed(ucoeff) << coeff_shift;
                        let (x, y) = coeff_coord;
                        *coeff_grid.get_mut(x as usize, y as usize).unwrap() = coeff;
                        prev_coeff = coeff;

                        if coeff != 0 {
                            non_zeros -= 1;
                        }
                    }
                }

                data.insert((x, y), CoeffData {
                    dct_select,
                    hf_mul: qf,
                    coeff,
                });
            }
        }

        dist.finalize()?;

        Ok(Self { data })
    }
}
