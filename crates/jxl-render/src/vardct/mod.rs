use jxl_color::OpsinInverseMatrix;
use jxl_frame::data::CoeffData;
use jxl_grid::{CutGrid, Grid, SimpleGrid};
use jxl_vardct::{
    LfChannelDequantization,
    Quantizer,
    DequantMatrixSet,
    LfChannelCorrelation, TransformType,
};

use crate::dct::dct_2d;

mod transform;
pub use transform::transform;

pub fn dequant_lf(
    in_out_xyb: &mut [CutGrid<'_, f32>; 3],
    lf_dequant: &LfChannelDequantization,
    quantizer: &Quantizer,
    extra_precision: u8,
) {
    debug_assert!(extra_precision < 4);
    let precision_scale = ((9 - extra_precision) as u32) << 23;
    let scale_inv = quantizer.global_scale * quantizer.quant_lf;
    let lf_x = lf_dequant.m_x_lf.to_bits() + precision_scale;
    let lf_y = lf_dequant.m_y_lf.to_bits() + precision_scale;
    let lf_b = lf_dequant.m_b_lf.to_bits() + precision_scale;
    let lf_x = f32::from_bits(lf_x) / scale_inv as f32;
    let lf_y = f32::from_bits(lf_y) / scale_inv as f32;
    let lf_b = f32::from_bits(lf_b) / scale_inv as f32;
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
    lf_image: &[SimpleGrid<f32>; 3],
    out: &mut [SimpleGrid<f32>; 3],
    lf_dequant: &LfChannelDequantization,
    quantizer: &Quantizer,
) {
    // smoothing
    const SCALE_SELF: f32 = 0.052262735;
    const SCALE_SIDE: f32 = 0.2034514;
    const SCALE_DIAG: f32 = 0.03348292;

    let scale_inv = quantizer.global_scale * quantizer.quant_lf;
    let lf_x = 512.0 * lf_dequant.m_x_lf / scale_inv as f32;
    let lf_y = 512.0 * lf_dequant.m_y_lf / scale_inv as f32;
    let lf_b = 512.0 * lf_dequant.m_b_lf / scale_inv as f32;

    let [in_x, in_y, in_b] = lf_image;
    let [out_x, out_y, out_b] = out;
    let width = out_x.width();
    let height = out_x.height();

    let in_x = in_x.buf();
    let in_y = in_y.buf();
    let in_b = in_b.buf();
    let out_x = out_x.buf_mut();
    let out_y = out_y.buf_mut();
    let out_b = out_b.buf_mut();

    let compute_self_side_diag = |g: &[f32], base_idx, width| {
        let celf = g[base_idx];
        let side = g[base_idx - width] + g[base_idx - 1] + g[base_idx + 1] + g[base_idx + width];
        let diag = g[base_idx - width - 1] + g[base_idx - width + 1] + g[base_idx + width - 1] + g[base_idx + width + 1];
        (celf, side, diag)
    };

    out_x[..width].copy_from_slice(&in_x[..width]);
    out_y[..width].copy_from_slice(&in_y[..width]);
    out_b[..width].copy_from_slice(&in_b[..width]);
    for y in 1..(height - 1) {
        out_x[y * width] = in_x[y * width];
        out_y[y * width] = in_y[y * width];
        out_b[y * width] = in_b[y * width];
        for x in 1..(width - 1) {
            let base_idx = y * width + x;

            let (x_self, x_side, x_diag) = compute_self_side_diag(in_x, base_idx, width);
            let (y_self, y_side, y_diag) = compute_self_side_diag(in_y, base_idx, width);
            let (b_self, b_side, b_diag) = compute_self_side_diag(in_b, base_idx, width);
            let x_wa = x_self * SCALE_SELF + x_side * SCALE_SIDE + x_diag * SCALE_DIAG;
            let y_wa = y_self * SCALE_SELF + y_side * SCALE_SIDE + y_diag * SCALE_DIAG;
            let b_wa = b_self * SCALE_SELF + b_side * SCALE_SIDE + b_diag * SCALE_DIAG;
            let x_gap_t = (x_wa - x_self).abs() / lf_x;
            let y_gap_t = (y_wa - y_self).abs() / lf_y;
            let b_gap_t = (b_wa - b_self).abs() / lf_b;

            let gap = 0.5f32.max(x_gap_t).max(y_gap_t).max(b_gap_t);
            let gap_scale = (3.0 - 4.0 * gap).max(0.0);

            out_x[base_idx] = (x_wa - x_self) * gap_scale + x_self;
            out_y[base_idx] = (y_wa - y_self) * gap_scale + y_self;
            out_b[base_idx] = (b_wa - b_self) * gap_scale + b_self;
        }
        out_x[(y + 1) * width - 1] = in_x[(y + 1) * width - 1];
        out_y[(y + 1) * width - 1] = in_y[(y + 1) * width - 1];
        out_b[(y + 1) * width - 1] = in_b[(y + 1) * width - 1];
    }
    out_x[(height - 1) * width..][..width].copy_from_slice(&in_x[(height - 1) * width..][..width]);
    out_y[(height - 1) * width..][..width].copy_from_slice(&in_y[(height - 1) * width..][..width]);
    out_b[(height - 1) * width..][..width].copy_from_slice(&in_b[(height - 1) * width..][..width]);
}

pub fn dequant_hf_varblock(
    coeff_data: &CoeffData,
    out: &mut CutGrid<'_>,
    channel: usize,
    oim: &OpsinInverseMatrix,
    quantizer: &Quantizer,
    dequant_matrices: &DequantMatrixSet,
    qm_scale: Option<u32>,
) {
    let CoeffData { dct_select, hf_mul, ref coeff } = *coeff_data;
    let need_transpose = dct_select.need_transpose();

    let mut mul = 65536.0 / (quantizer.global_scale as i32 * hf_mul) as f32;
    if let Some(qm_scale) = qm_scale {
        let scale = 0.8f32.powi(qm_scale as i32 - 2);
        mul *= scale;
    }
    let quant_bias = oim.quant_bias[channel];
    let quant_bias_numerator = oim.quant_bias_numerator;

    let coeff = &coeff[channel];
    let mut buf = vec![0f32; coeff.width() * coeff.height()];

    for (&quant, out) in coeff.buf().iter().zip(&mut buf) {
        *out = match quant {
            -1 => -quant_bias,
            0 => 0.0,
            1 => quant_bias,
            quant => {
                let q = quant as f32;
                q - (quant_bias_numerator / q)
            },
        };
    }

    let matrix = dequant_matrices.get(channel, dct_select);
    for (out, &mat) in buf.iter_mut().zip(matrix) {
        let val = *out * mat;
        *out = val * mul;
    }

    if need_transpose {
        for y in 0..coeff.height() {
            for x in 0..coeff.width() {
                *out.get_mut(y, x) = buf[y * coeff.width() + x];
            }
        }
    } else {
        for y in 0..coeff.height() {
            let row = out.get_row_mut(y);
            row.copy_from_slice(&buf[y * coeff.width()..][..coeff.width()]);
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
    coeff_xyb: &mut [&mut CutGrid<'_>; 3],
    lf_left: usize,
    lf_top: usize,
    x_from_y: &Grid<i32>,
    b_from_y: &Grid<i32>,
    lf_chan_corr: &LfChannelCorrelation,
) {
    let LfChannelCorrelation {
        colour_factor,
        base_correlation_x,
        base_correlation_b,
        ..
    } = *lf_chan_corr;

    let [coeff_x, coeff_y, coeff_b] = coeff_xyb;
    let width = coeff_x.width();
    let height = coeff_x.height();

    for cy in 0..height {
        for cx in 0..width {
            let (x, y) = (lf_left + cx, lf_top + cy);
            let cfactor_x = x / 64;
            let cfactor_y = y / 64;

            let x_factor = *x_from_y.get(cfactor_x, cfactor_y).unwrap();
            let b_factor = *b_from_y.get(cfactor_x, cfactor_y).unwrap();
            let kx = base_correlation_x + (x_factor as f32 / colour_factor as f32);
            let kb = base_correlation_b + (b_factor as f32 / colour_factor as f32);

            let coeff_y = coeff_y.get(cx, cy);
            *coeff_x.get_mut(cx, cy) += kx * coeff_y;
            *coeff_b.get_mut(cx, cy) += kb * coeff_y;
        }
    }
}

pub fn llf_from_lf(
    lf: &SimpleGrid<f32>,
    left: usize,
    top: usize,
    dct_select: TransformType,
) -> SimpleGrid<f32> {
    use TransformType::*;

    fn scale_f(c: usize, b: usize) -> f32 {
        let cb = c as f32 / b as f32;
        let recip = (cb * std::f32::consts::FRAC_PI_2).cos() *
            (cb * std::f32::consts::PI).cos() *
            (cb * 2.0 * std::f32::consts::PI).cos();
        recip.recip()
    }

    let (bw, bh) = dct_select.dct_select_size();
    let bw = bw as usize;
    let bh = bh as usize;

    if matches!(dct_select, Hornuss | Dct2 | Dct4 | Dct8x4 | Dct4x8 | Dct8 | Afv0 | Afv1 | Afv2 | Afv3) {
        debug_assert_eq!(bw * bh, 1);
        let mut out = SimpleGrid::new(1, 1);
        out.buf_mut()[0] = *lf.get(left, top).unwrap();
        out
    } else {
        let mut out = SimpleGrid::new(bw, bh);
        for y in 0..bh {
            for x in 0..bw {
                out.buf_mut()[y * bw + x] = *lf.get(left + x, top + y).unwrap();
            }
        }
        dct_2d(&mut out);

        for y in 0..bh {
            for x in 0..bw {
                out.buf_mut()[y * bw + x] *= scale_f(y, bh * 8) * scale_f(x, bw * 8);
            }
        }

        out
    }
}
