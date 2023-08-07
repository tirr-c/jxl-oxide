use std::collections::HashMap;

use jxl_frame::{data::{LfGroup, LfGlobal, HfGlobal}, FrameHeader};
use jxl_grid::{CutGrid, SimpleGrid};
use jxl_image::ImageHeader;
use jxl_modular::ChannelShift;
use jxl_vardct::{
    LfChannelCorrelation,
    LfChannelDequantization,
    Quantizer,
    TransformType,
    BlockInfo,
};

use crate::dct;

mod transform;
pub use transform::transform;

#[cfg(target_arch = "x86_64")]
mod x86_64;
#[cfg(target_arch = "x86_64")]
use x86_64 as impls;

mod generic;
#[cfg(not(target_arch = "x86_64"))]
use generic as impls;

pub fn dequant_lf(
    in_out_xyb: &mut [CutGrid<'_, f32>; 3],
    lf_dequant: &LfChannelDequantization,
    quantizer: &Quantizer,
    extra_precision: u8,
) {
    debug_assert!(extra_precision < 4);
    let precision_scale = (1 << (9 - extra_precision)) as f32;
    let scale_inv = quantizer.global_scale * quantizer.quant_lf;
    let lf_x = lf_dequant.m_x_lf * precision_scale / scale_inv as f32;
    let lf_y = lf_dequant.m_y_lf * precision_scale / scale_inv as f32;
    let lf_b = lf_dequant.m_b_lf * precision_scale / scale_inv as f32;
    let lf_scaled = [lf_x, lf_y, lf_b];

    for (out, lf_scaled) in in_out_xyb.iter_mut().zip(lf_scaled) {
        let height = out.height();
        for y in 0..height {
            let row = out.get_row_mut(y);
            for out in row {
                *out *= lf_scaled;
            }
        }
    }
}

pub fn adaptive_lf_smoothing(
    lf_image: &mut [SimpleGrid<f32>; 3],
    lf_dequant: &LfChannelDequantization,
    quantizer: &Quantizer,
) {
    let scale_inv = quantizer.global_scale * quantizer.quant_lf;
    let lf_x = 512.0 * lf_dequant.m_x_lf / scale_inv as f32;
    let lf_y = 512.0 * lf_dequant.m_y_lf / scale_inv as f32;
    let lf_b = 512.0 * lf_dequant.m_b_lf / scale_inv as f32;

    let [in_x, in_y, in_b] = lf_image;
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
    out: &mut [SimpleGrid<f32>; 3],
    image_header: &ImageHeader,
    frame_header: &FrameHeader,
    lf_global: &LfGlobal,
    lf_groups: &HashMap<u32, LfGroup>,
    hf_global: &HfGlobal,
) {
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

                    let (bw, bh) = dct_select.dct_select_size();
                    let width = bw * 8;
                    let height = bh * 8;
                    let need_transpose = dct_select.need_transpose();
                    let mul = 65536.0 / (quantizer.global_scale as i32 * hf_mul) as f32 * qm_scale[channel];

                    let mut new_matrix;
                    let mut matrix = dequant_matrices.get(channel, dct_select);
                    if need_transpose {
                        new_matrix = vec![0f32; matrix.len()];
                        for (idx, val) in new_matrix.iter_mut().enumerate() {
                            let mat_x = idx % width as usize;
                            let mat_y = idx / width as usize;
                            *val = matrix[mat_x * height as usize + mat_y];
                        }
                        matrix = &new_matrix;
                    }

                    let left = lf_left as usize + bx * 8;
                    let top = lf_top as usize + by * 8;
                    let left = left >> hshift;
                    let top = top >> vshift;

                    let mut coeff = CutGrid::from_buf(
                        &mut coeff.buf_mut()[top * stride + left..],
                        width as usize,
                        height as usize,
                        stride,
                    );
                    for y in 0..height {
                        let row = coeff.get_row_mut(y as usize);
                        let matrix_row = &matrix[(y * width) as usize..][..width as usize];
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
    coeff_xyb: &mut [CutGrid<'_, f32>; 3],
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

    let [x, y, b] = coeff_xyb;
    let height = x.height();
    for row in 0..height {
        let x = x.get_row_mut(row);
        let y = y.get_row(row);
        let b = b.get_row_mut(row);
        for ((x, y), b) in x.iter_mut().zip(y).zip(b) {
            let y = *y;
            *x += kx * y;
            *b += kb * y;
        }
    }
}

pub fn chroma_from_luma_hf(
    coeff_xyb: &mut [SimpleGrid<f32>; 3],
    frame_header: &FrameHeader,
    lf_global: &LfGlobal,
    lf_groups: &HashMap<u32, LfGroup>,
) {
    let LfChannelCorrelation {
        colour_factor,
        base_correlation_x,
        base_correlation_b,
        ..
    } = lf_global.vardct.as_ref().unwrap().lf_chan_corr;

    let [coeff_x, coeff_y, coeff_b] = coeff_xyb;
    let width = coeff_x.width();
    let height = coeff_x.height();
    let lf_group_dim = frame_header.lf_group_dim() as usize;

    for lf_group_idx in 0..frame_header.num_lf_groups() {
        let Some(lf_group) = lf_groups.get(&lf_group_idx) else { continue; };
        let hf_meta = lf_group.hf_meta.as_ref().unwrap();
        let x_from_y = &hf_meta.x_from_y;
        let b_from_y = &hf_meta.b_from_y;

        let lf_left = ((lf_group_idx % frame_header.lf_groups_per_row()) * frame_header.lf_group_dim()) as usize;
        let lf_top = ((lf_group_idx / frame_header.lf_groups_per_row()) * frame_header.lf_group_dim()) as usize;
        let lf_group_width = lf_group_dim.min(width - lf_left);
        let lf_group_height = lf_group_dim.min(height - lf_top);

        for cy in 0..lf_group_height {
            for cx in 0..lf_group_width {
                let x = lf_left + cx;
                let y = lf_top + cy;
                let cfactor_x = cx / 64;
                let cfactor_y = cy / 64;

                let x_factor = *x_from_y.get(cfactor_x, cfactor_y).unwrap();
                let b_factor = *b_from_y.get(cfactor_x, cfactor_y).unwrap();
                let kx = base_correlation_x + (x_factor as f32 / colour_factor as f32);
                let kb = base_correlation_b + (b_factor as f32 / colour_factor as f32);

                let coeff_y = *coeff_y.get(x, y).unwrap();
                *coeff_x.get_mut(x, y).unwrap() += kx * coeff_y;
                *coeff_b.get_mut(x, y).unwrap() += kb * coeff_y;
            }
        }
    }
}

pub fn transform_with_lf(
    lf: &[SimpleGrid<f32>],
    coeff_out: &mut [SimpleGrid<f32>; 3],
    frame_header: &FrameHeader,
    lf_groups: &HashMap<u32, LfGroup>,
) {
    use TransformType::*;

    fn scale_f(c: usize, b: usize) -> f32 {
        let cb = c as f32 / b as f32;
        let recip = (cb * std::f32::consts::FRAC_PI_2).cos() *
            (cb * std::f32::consts::PI).cos() *
            (cb * 2.0 * std::f32::consts::PI).cos();
        recip.recip()
    }

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

                    let (bw, bh) = dct_select.dct_select_size();
                    let bw = bw as usize;
                    let bh = bh as usize;

                    let left = lf_left as usize + bx * 8;
                    let top = lf_top as usize + by * 8;
                    let left = left >> hshift;
                    let top = top >> vshift;

                    if matches!(dct_select, Hornuss | Dct2 | Dct4 | Dct8x4 | Dct4x8 | Dct8 | Afv0 | Afv1 | Afv2 | Afv3) {
                        debug_assert_eq!(bw * bh, 1);
                        *coeff.get_mut(left, top).unwrap() = *lf.get(left / 8, top / 8).unwrap();
                    } else {
                        let mut out = CutGrid::from_buf(
                            &mut coeff.buf_mut()[top * stride + left..],
                            bw,
                            bh,
                            stride,
                        );

                        for y in 0..bh {
                            for x in 0..bw {
                                *out.get_mut(x, y) = *lf.get(left / 8 + x, top / 8 + y).unwrap();
                            }
                        }
                        dct::dct_2d(&mut out, dct::DctDirection::Forward);
                        for y in 0..bh {
                            for x in 0..bw {
                                *out.get_mut(x, y) *= scale_f(y, bh * 8) * scale_f(x, bw * 8);
                            }
                        }
                    }

                    let mut block = CutGrid::from_buf(
                        &mut coeff.buf_mut()[top * stride + left..],
                        bw * 8,
                        bh * 8,
                        stride,
                    );
                    transform(&mut block, dct_select);
                }
            }
        }
    }
}
