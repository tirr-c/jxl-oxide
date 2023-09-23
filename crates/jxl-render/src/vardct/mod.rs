use std::collections::HashMap;

use jxl_frame::{data::{LfGroup, LfGlobal, HfGlobal}, FrameHeader};
use jxl_grid::{CutGrid, SimpleGrid, Grid};
use jxl_image::ImageHeader;
use jxl_modular::ChannelShift;
use jxl_vardct::{
    LfChannelCorrelation,
    LfChannelDequantization,
    Quantizer,
    TransformType,
    BlockInfo,
};

use crate::{dct, region::ImageWithRegion, Region};

mod transform;
pub use transform::transform;

#[cfg(target_arch = "x86_64")]
mod x86_64;
#[cfg(target_arch = "x86_64")]
use x86_64 as impls;

mod generic;
#[cfg(not(target_arch = "x86_64"))]
use generic as impls;

pub fn copy_lf_dequant(
    out: &mut SimpleGrid<f32>,
    left: usize,
    top: usize,
    quantizer: &Quantizer,
    m_lf: f32,
    channel_data: &Grid<i32>,
    extra_precision: u8,
) {
    debug_assert!(extra_precision < 4);
    let precision_scale = 1i32 << (9 - extra_precision);
    let scale_inv = quantizer.global_scale * quantizer.quant_lf;
    let scale = m_lf * precision_scale as f32 / scale_inv as f32;

    let width = channel_data.width();
    let height = channel_data.height();
    let stride = out.width();
    let buf = &mut out.buf_mut()[top * stride + left..];
    let mut grid = CutGrid::from_buf(buf, width, height, stride);
    if let Some(channel_data) = channel_data.as_simple() {
        let buf = channel_data.buf();
        for y in 0..height {
            let row = grid.get_row_mut(y);
            let quant = &buf[y * width..][..width];
            for (out, &q) in row.iter_mut().zip(quant) {
                *out = q as f32 * scale;
            }
        }
    } else {
        for y in 0..height {
            let row = grid.get_row_mut(y);
            for (x, out) in row.iter_mut().enumerate() {
                let q = *channel_data.get(x, y).unwrap();
                *out = q as f32 * scale;
            }
        }
    }
}

pub fn adaptive_lf_smoothing(
    lf_image: &mut [SimpleGrid<f32>],
    lf_dequant: &LfChannelDequantization,
    quantizer: &Quantizer,
) {
    let scale_inv = quantizer.global_scale * quantizer.quant_lf;
    let lf_x = 512.0 * lf_dequant.m_x_lf / scale_inv as f32;
    let lf_y = 512.0 * lf_dequant.m_y_lf / scale_inv as f32;
    let lf_b = 512.0 * lf_dequant.m_b_lf / scale_inv as f32;

    let [in_x, in_y, in_b] = lf_image else { panic!("lf_image should be three-channel image") };
    let width = in_x.width();
    let height = in_x.height();

    let in_x = in_x.buf_mut();
    let in_y = in_y.buf_mut();
    let in_b = in_b.buf_mut();

    impls::adaptive_lf_smoothing_impl(
        width,
        height,
        [in_x, in_y, in_b],
        [lf_x, lf_y, lf_b],
    );
}

pub fn dequant_hf_varblock(
    out: &mut ImageWithRegion,
    image_header: &ImageHeader,
    frame_header: &FrameHeader,
    lf_global: &LfGlobal,
    lf_groups: &HashMap<u32, LfGroup>,
    hf_global: &HfGlobal,
) {
    let region = out.region();
    let out = out.buffer_mut();
    let shifts_cbycr: [_; 3] = std::array::from_fn(|idx| {
        ChannelShift::from_jpeg_upsampling(frame_header.jpeg_upsampling, idx)
    });
    let oim = &image_header.metadata.opsin_inverse_matrix;
    let quantizer = &lf_global.vardct.as_ref().unwrap().quantizer;
    let dequant_matrices = &hf_global.dequant_matrices;

    let qm_scale = [
        0.8f32.powi(frame_header.x_qm_scale as i32 - 2),
        1.0f32,
        0.8f32.powi(frame_header.b_qm_scale as i32 - 2),
    ];

    let quant_bias_numerator = oim.quant_bias_numerator;

    for lf_group_idx in 0..frame_header.num_lf_groups() {
        let Some(lf_group) = lf_groups.get(&lf_group_idx) else { continue; };
        let hf_meta = lf_group.hf_meta.as_ref().unwrap();

        let lf_left = (lf_group_idx % frame_header.lf_groups_per_row()) * frame_header.lf_group_dim();
        let lf_top = (lf_group_idx / frame_header.lf_groups_per_row()) * frame_header.lf_group_dim();

        let block_info = &hf_meta.block_info;
        let w8 = block_info.width();
        let h8 = block_info.height();

        for (channel, coeff) in out.iter_mut().enumerate() {
            let shift = shifts_cbycr[channel];
            let vshift = shift.vshift();
            let hshift = shift.hshift();

            let quant_bias = oim.quant_bias[channel];
            let stride = coeff.width();
            for by in 0..h8 {
                for bx in 0..w8 {
                    let &BlockInfo::Data { dct_select, hf_mul } = block_info.get(bx, by).unwrap() else { continue; };
                    if ((bx >> hshift) << hshift) != bx || ((by >> vshift) << vshift) != by {
                        continue;
                    }

                    let left = lf_left as usize + bx * 8;
                    let top = lf_top as usize + by * 8;
                    let (bw, bh) = dct_select.dct_select_size();
                    let width = bw * 8;
                    let height = bh * 8;
                    let block_region = Region {
                        left: left as i32,
                        top: top as i32,
                        width,
                        height,
                    };
                    if region.intersection(block_region).is_empty() {
                        continue;
                    }

                    let left = left >> hshift;
                    let top = top >> vshift;
                    let offset = (top - (region.top as usize >> vshift)) * stride + (left - (region.left as usize >> hshift));
                    let coeff_buf = coeff.buf_mut();

                    let need_transpose = dct_select.need_transpose();
                    let mul = 65536.0 / (quantizer.global_scale as i32 * hf_mul) as f32 * qm_scale[channel];

                    let matrix = if need_transpose {
                        dequant_matrices.get_transposed(channel, dct_select)
                    } else {
                        dequant_matrices.get(channel, dct_select)
                    };

                    let mut coeff = CutGrid::from_buf(
                        &mut coeff_buf[offset..],
                        width as usize,
                        height as usize,
                        stride,
                    );
                    for (y, matrix_row) in matrix.chunks_exact(width as usize).enumerate() {
                        let row = coeff.get_row_mut(y);
                        for (q, &m) in row.iter_mut().zip(matrix_row) {
                            if q.abs() <= 1.0f32 {
                                *q *= quant_bias;
                            } else {
                                *q -= quant_bias_numerator / *q;
                            }
                            *q *= m;
                            *q *= mul;
                        }
                    }
                }
            }
        }
    }
}

pub fn chroma_from_luma_lf(
    coeff_xyb: &mut [SimpleGrid<f32>],
    lf_chan_corr: &LfChannelCorrelation,
) {
    let LfChannelCorrelation {
        colour_factor,
        base_correlation_x,
        base_correlation_b,
        x_factor_lf,
        b_factor_lf,
        ..
    } = *lf_chan_corr;

    let x_factor = x_factor_lf as i32 - 128;
    let b_factor = b_factor_lf as i32 - 128;
    let kx = base_correlation_x + (x_factor as f32 / colour_factor as f32);
    let kb = base_correlation_b + (b_factor as f32 / colour_factor as f32);

    let [x, y, b] = coeff_xyb else { panic!("coeff_xyb should be three-channel image") };
    for ((x, y), b) in x.buf_mut().iter_mut().zip(y.buf()).zip(b.buf_mut()) {
        let y = *y;
        *x += kx * y;
        *b += kb * y;
    }
}

pub fn chroma_from_luma_hf(
    coeff_xyb: &mut ImageWithRegion,
    cfl_grid: &ImageWithRegion,
) {
    let region = coeff_xyb.region();
    let cfl_grid_region = cfl_grid.region();
    let coeff_xyb = coeff_xyb.buffer_mut();
    let cfl_grid = cfl_grid.buffer();

    let [coeff_x, coeff_y, coeff_b] = coeff_xyb else { panic!() };
    let [x_from_y, b_from_y] = cfl_grid else { panic!() };
    let stride = cfl_grid_region.width as usize;

    for y in 0..region.height {
        let cy = (y as i32 + region.top) / 64 - cfl_grid_region.top;
        debug_assert!(cy >= 0);
        let cy = cy as usize;
        let y = y as usize;

        let x_from_y = &x_from_y.buf()[cy * stride..][..stride];
        let b_from_y = &b_from_y.buf()[cy * stride..][..stride];

        let mut x = cfl_grid_region.left * 64 - region.left - 1;
        'outer: for (kx, kb) in x_from_y.iter().zip(b_from_y) {
            for _ in 0..64 {
                x += 1;
                if x < 0 {
                    continue;
                }
                if x >= region.width as i32 {
                    break 'outer;
                }
                let x = x as usize;

                let coeff_y = *coeff_y.get(x, y).unwrap();
                *coeff_x.get_mut(x, y).unwrap() += kx * coeff_y;
                *coeff_b.get_mut(x, y).unwrap() += kb * coeff_y;
            }
        }
    }
}

pub fn transform_with_lf(
    lf: &ImageWithRegion,
    coeff_out: &mut ImageWithRegion,
    frame_header: &FrameHeader,
    lf_groups: &HashMap<u32, LfGroup>,
) {
    use TransformType::*;

    let lf_region = lf.region();
    let coeff_region = coeff_out.region();
    let lf = lf.buffer();
    let coeff_out = coeff_out.buffer_mut();
    let shifts_cbycr: [_; 3] = std::array::from_fn(|idx| {
        ChannelShift::from_jpeg_upsampling(frame_header.jpeg_upsampling, idx)
    });

    for lf_group_idx in 0..frame_header.num_lf_groups() {
        let Some(lf_group) = lf_groups.get(&lf_group_idx) else { continue; };
        let hf_meta = lf_group.hf_meta.as_ref().unwrap();

        let lf_left = (lf_group_idx % frame_header.lf_groups_per_row()) * frame_header.lf_group_dim();
        let lf_top = (lf_group_idx / frame_header.lf_groups_per_row()) * frame_header.lf_group_dim();

        let block_info = &hf_meta.block_info;
        let w8 = block_info.width();
        let h8 = block_info.height();

        for (channel, (coeff, lf)) in coeff_out.iter_mut().zip(lf).enumerate() {
            let shift = shifts_cbycr[channel];
            let vshift = shift.vshift();
            let hshift = shift.hshift();

            let stride = coeff.width();
            for by in 0..h8 {
                for bx in 0..w8 {
                    let &BlockInfo::Data { dct_select, .. } = block_info.get(bx, by).unwrap() else { continue; };
                    if ((bx >> hshift) << hshift) != bx || ((by >> vshift) << vshift) != by {
                        continue;
                    }

                    let left = lf_left as usize + bx * 8;
                    let top = lf_top as usize + by * 8;
                    let (bw, bh) = dct_select.dct_select_size();
                    let width = bw * 8;
                    let height = bh * 8;
                    let block_region = Region {
                        left: left as i32,
                        top: top as i32,
                        width,
                        height,
                    };
                    if coeff_region.intersection(block_region).is_empty() {
                        continue;
                    }

                    let left = left >> hshift;
                    let top = top >> vshift;
                    let offset = (top - (coeff_region.top as usize >> vshift)) * stride + (left - (coeff_region.left as usize >> hshift));
                    let coeff_buf = coeff.buf_mut();

                    let bw = bw as usize;
                    let bh = bh as usize;
                    let logbw = bw.trailing_zeros() as usize;
                    let logbh = bh.trailing_zeros() as usize;

                    let lf_x = left / 8 - (lf_region.left as usize >> hshift);
                    let lf_y = top / 8 - (lf_region.top as usize >> vshift);
                    if matches!(dct_select, Hornuss | Dct2 | Dct4 | Dct8x4 | Dct4x8 | Dct8 | Afv0 | Afv1 | Afv2 | Afv3) {
                        debug_assert_eq!(bw * bh, 1);
                        coeff_buf[offset] = *lf.get(lf_x, lf_y).unwrap();
                    } else {
                        let mut out = CutGrid::from_buf(
                            &mut coeff_buf[offset..],
                            bw,
                            bh,
                            stride,
                        );

                        for y in 0..bh {
                            for x in 0..bw {
                                *out.get_mut(x, y) = *lf.get(lf_x + x, lf_y + y).unwrap();
                            }
                        }
                        dct::dct_2d(&mut out, dct::DctDirection::Forward);
                        for y in 0..bh {
                            for x in 0..bw {
                                *out.get_mut(x, y) /= scale_f(y, 5 - logbh) * scale_f(x, 5 - logbw);
                            }
                        }
                    }

                    let mut block = CutGrid::from_buf(
                        &mut coeff_buf[offset..],
                        width as usize,
                        height as usize,
                        stride,
                    );
                    transform(&mut block, dct_select);
                }
            }
        }
    }
}

fn scale_f(c: usize, logb: usize) -> f32 {
    // Precomputed for c = 0..32, b = 256
    #[allow(clippy::excessive_precision)]
    const SCALE_F: [f32; 32] = [
        1.0000000000000000, 0.9996047255830407,
        0.9984194528776054, 0.9964458326264695,
        0.9936866130906366, 0.9901456355893141,
        0.9858278282666936, 0.9807391980963174,
        0.9748868211368796, 0.9682788310563117,
        0.9609244059440204, 0.9528337534340876,
        0.9440180941651672, 0.9344896436056892,
        0.9242615922757944, 0.9133480844001980,
        0.9017641950288744, 0.8895259056651056,
        0.8766500784429904, 0.8631544288990163,
        0.8490574973847023, 0.8343786191696513,
        0.8191378932865928, 0.8033561501721485,
        0.7870549181591013, 0.7702563888779096,
        0.7529833816270532, 0.7352593067735488,
        0.7171081282466044, 0.6985543251889097,
        0.6796228528314652, 0.6603391026591464,
    ];
    SCALE_F[c << logb]
}
