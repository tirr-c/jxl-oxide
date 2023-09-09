use jxl_bitstream::Bitstream;
use jxl_grid::{Subgrid, CutGrid, SimpleGrid};
use jxl_modular::ChannelShift;

use crate::{
    BlockInfo,
    HfBlockContext,
    HfPass,
    Result,
};

/// Parameters for decoding `HfCoeff`.
#[derive(Debug)]
pub struct HfCoeffParams<'a> {
    pub num_hf_presets: u32,
    pub hf_block_ctx: &'a HfBlockContext,
    pub block_info: Subgrid<'a, BlockInfo>,
    pub jpeg_upsampling: [u32; 3],
    pub lf_quant: Option<[Subgrid<'a, i32>; 3]>,
    pub hf_pass: &'a HfPass,
    pub coeff_shift: u32,
}

pub fn write_hf_coeff<R: std::io::Read>(
    bitstream: &mut Bitstream<R>,
    params: HfCoeffParams,
    hf_coeff_output: &mut [CutGrid<'_, f32>; 3],
) -> Result<()> {
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
        SimpleGrid::new(width as usize, height as usize)
    });
    let predict_non_zeros = |grid: &SimpleGrid<u32>, x: usize, y: usize| {
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
                        *non_zeros_grid.get_mut(sx + dx, sy + dy).unwrap() = non_zeros_val;
                    }
                }

                let size = (w8 * 8) * (h8 * 8);
                let coeff_grid = &mut hf_coeff_output[c];
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
                    let (x, y) = if dct_select.need_transpose() {
                        (sx * 8 + coeff_coord.1 as usize, sy * 8 + coeff_coord.0 as usize)
                    } else {
                        (sx * 8 + coeff_coord.0 as usize, sy * 8 + coeff_coord.1 as usize)
                    };
                    *coeff_grid.get_mut(x, y) += coeff as f32;
                    prev_coeff = coeff;

                    if coeff != 0 {
                        non_zeros -= 1;
                    }
                }
            }
        }
    }

    dist.finalize()?;

    Ok(())
}
