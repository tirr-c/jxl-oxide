use jxl_bitstream::Bitstream;
use jxl_grid::{AllocTracker, MutableSubgrid, SharedSubgrid};
use jxl_modular::{ChannelShift, Sample};

use crate::{BlockInfo, HfBlockContext, HfPass, Result};

/// Parameters for decoding `HfCoeff`.
#[derive(Debug)]
pub struct HfCoeffParams<'a, 'b, S: Sample> {
    pub num_hf_presets: u32,
    pub hf_block_ctx: &'a HfBlockContext,
    pub block_info: SharedSubgrid<'a, BlockInfo>,
    pub jpeg_upsampling: [u32; 3],
    pub lf_quant: Option<[SharedSubgrid<'a, S>; 3]>,
    pub hf_pass: &'a HfPass,
    pub coeff_shift: u32,
    pub tracker: Option<&'b AllocTracker>,
}

/// Decode and write HF coefficients from the bitstream.
pub fn write_hf_coeff<S: Sample>(
    bitstream: &mut Bitstream,
    params: HfCoeffParams<S>,
    hf_coeff_output: &mut [MutableSubgrid<i32>; 3],
) -> Result<()> {
    const COEFF_FREQ_CONTEXT: [u32; 63] = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19,
        20, 20, 21, 21, 22, 22, 23, 23, 23, 23, 24, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 26, 27,
        27, 27, 27, 28, 28, 28, 28, 29, 29, 29, 29, 30, 30, 30, 30,
    ];
    const COEFF_NUM_NONZERO_CONTEXT: [u32; 63] = [
        0, 31, 62, 62, 93, 93, 93, 93, 123, 123, 123, 123, 152, 152, 152, 152, 152, 152, 152, 152,
        180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 206, 206, 206, 206, 206, 206,
        206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206,
        206, 206, 206, 206, 206, 206, 206,
    ];

    let HfCoeffParams {
        num_hf_presets,
        hf_block_ctx,
        block_info,
        jpeg_upsampling,
        lf_quant,
        hf_pass,
        coeff_shift,
        tracker,
    } = params;
    let mut dist = hf_pass.clone_decoder();

    let HfBlockContext {
        qf_thresholds,
        lf_thresholds,
        block_ctx_map,
        num_block_clusters,
    } = hf_block_ctx;
    let lf_idx_mul =
        (lf_thresholds[0].len() + 1) * (lf_thresholds[1].len() + 1) * (lf_thresholds[2].len() + 1);
    let hf_idx_mul = qf_thresholds.len() + 1;
    let upsampling_shifts: [_; 3] =
        std::array::from_fn(|idx| ChannelShift::from_jpeg_upsampling(jpeg_upsampling, idx));
    let hshifts = upsampling_shifts.map(|shift| shift.hshift());
    let vshifts = upsampling_shifts.map(|shift| shift.vshift());

    let hfp_bits = num_hf_presets.next_power_of_two().trailing_zeros();
    let hfp = bitstream.read_bits(hfp_bits as usize)?;
    if hfp >= num_hf_presets {
        tracing::error!(hfp, num_hf_presets, "selected HF preset out of bounds");
        return Err(
            jxl_bitstream::Error::ValidationFailed("selected HF preset out of bounds").into(),
        );
    }

    let ctx_size = 495 * *num_block_clusters;
    let cluster_map = dist.cluster_map()[(ctx_size * hfp) as usize..][..ctx_size as usize].to_vec();

    dist.begin(bitstream)?;

    let width = block_info.width();
    let height = block_info.height();
    let non_zeros_grid_lengths =
        upsampling_shifts.map(|shift| shift.shift_size((width as u32, height as u32)).0 as usize);

    let _non_zeros_grid_handle = tracker
        .map(|tracker| {
            let len =
                non_zeros_grid_lengths[0] + non_zeros_grid_lengths[1] + non_zeros_grid_lengths[2];
            tracker.alloc::<u32>(len)
        })
        .transpose()?;
    let mut non_zeros_grid_row = [
        vec![0u32; non_zeros_grid_lengths[0]],
        vec![0u32; non_zeros_grid_lengths[1]],
        vec![0u32; non_zeros_grid_lengths[2]],
    ];

    for y in 0..height {
        for x in 0..width {
            let BlockInfo::Data {
                dct_select,
                hf_mul: qf,
            } = block_info.get(x, y)
            else {
                continue;
            };
            let (w8, h8) = dct_select.dct_select_size();
            let num_blocks = w8 * h8; // power of 2
            let num_blocks_log = num_blocks.trailing_zeros();
            let order_id = dct_select.order_id();

            let lf_idx = if let Some(lf_quant) = &lf_quant {
                let mut idx = 0usize;
                for c in [0, 2, 1] {
                    let lf_thresholds = &lf_thresholds[c];
                    idx *= lf_thresholds.len() + 1;

                    let x = x >> hshifts[c];
                    let y = y >> vshifts[c];
                    let q = lf_quant[c].get(x, y);
                    for &threshold in lf_thresholds {
                        if q.to_i32() > threshold {
                            idx += 1;
                        }
                    }
                }
                idx
            } else {
                0
            };

            let hf_idx = {
                let mut idx = 0usize;
                for &threshold in qf_thresholds {
                    if qf > threshold as i32 {
                        idx += 1;
                    }
                }
                idx
            };

            for c in 0..3 {
                let ch_idx = c * 13 + order_id as usize;
                let c = [1, 0, 2][c]; // y, x, b

                let hshift = hshifts[c];
                let vshift = vshifts[c];
                let sx = x >> hshift;
                let sy = y >> vshift;
                if hshift != 0 || vshift != 0 {
                    if sx << hshift != x || sy << vshift != y {
                        continue;
                    }
                    if !matches!(block_info.get(sx, sy), BlockInfo::Data { .. }) {
                        continue;
                    }
                }

                let idx = (ch_idx * hf_idx_mul + hf_idx) * lf_idx_mul + lf_idx;
                let block_ctx = block_ctx_map[idx] as u32;
                let non_zeros_ctx = {
                    let predicted = if sy == 0 {
                        if sx == 0 {
                            32
                        } else {
                            non_zeros_grid_row[c][sx - 1]
                        }
                    } else if sx == 0 {
                        non_zeros_grid_row[c][sx]
                    } else {
                        (non_zeros_grid_row[c][sx] + non_zeros_grid_row[c][sx - 1] + 1) >> 1
                    };
                    debug_assert!(predicted < 64);

                    let idx = if predicted >= 8 {
                        4 + predicted / 2
                    } else {
                        predicted
                    };
                    block_ctx + idx * num_block_clusters
                };

                let mut non_zeros = dist.read_varint_with_multiplier_clustered(
                    bitstream,
                    cluster_map[non_zeros_ctx as usize],
                    0,
                )?;
                if non_zeros > (63 << num_blocks_log) {
                    tracing::error!(non_zeros, num_blocks, "non_zeros too large");
                    return Err(
                        jxl_bitstream::Error::ValidationFailed("non_zeros too large").into(),
                    );
                }

                let non_zeros_val = (non_zeros + num_blocks - 1) >> num_blocks_log;
                for dx in 0..w8 as usize {
                    non_zeros_grid_row[c][sx + dx] = non_zeros_val;
                }
                if non_zeros == 0 {
                    continue;
                }

                let coeff_grid = &mut hf_coeff_output[c];
                let mut is_prev_coeff_nonzero = (non_zeros <= num_blocks * 4) as u32;
                let order = hf_pass.order(order_id as usize, c);

                let coeff_ctx_base = block_ctx * 458 + 37 * num_block_clusters;
                let cluster_map = &cluster_map[coeff_ctx_base as usize..][..458];
                for (idx, &coeff_coord) in order[num_blocks as usize..].iter().enumerate() {
                    let coeff_ctx = {
                        let non_zeros = (non_zeros - 1) >> num_blocks_log;
                        let idx = idx >> num_blocks_log;
                        (COEFF_NUM_NONZERO_CONTEXT[non_zeros as usize] + COEFF_FREQ_CONTEXT[idx])
                            * 2
                            + is_prev_coeff_nonzero
                    };
                    let cluster = *cluster_map.get(coeff_ctx as usize).ok_or_else(|| {
                        tracing::error!("too many zeros in varblock HF coefficient");
                        jxl_bitstream::Error::ValidationFailed(
                            "too many zeros in varblock HF coefficient",
                        )
                    })?;
                    let ucoeff =
                        dist.read_varint_with_multiplier_clustered(bitstream, cluster, 0)?;
                    if ucoeff == 0 {
                        is_prev_coeff_nonzero = 0;
                        continue;
                    }

                    let coeff = jxl_bitstream::unpack_signed(ucoeff) << coeff_shift;
                    let (mut dx, mut dy) = coeff_coord;
                    if dct_select.need_transpose() {
                        std::mem::swap(&mut dx, &mut dy);
                    }
                    let x = sx * 8 + dx as usize;
                    let y = sy * 8 + dy as usize;

                    *coeff_grid.get_mut(x, y) += coeff;

                    is_prev_coeff_nonzero = 1;
                    non_zeros -= 1;

                    if non_zeros == 0 {
                        break;
                    }
                }
            }
        }
    }

    dist.finalize()?;

    Ok(())
}
