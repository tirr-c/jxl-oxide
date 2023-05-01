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
    lf_image: &[SimpleGrid<f32>; 3],
    out: &mut [SimpleGrid<f32>; 3],
    lf_dequant: &LfChannelDequantization,
    quantizer: &Quantizer,
) {
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

    impls::adaptive_lf_smoothing_impl(
        width,
        height,
        [in_x, in_y, in_b],
        [out_x, out_y, out_b],
        [lf_x, lf_y, lf_b],
    );
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
            let (x, y) = if width == height {
                (lf_left + cy, lf_top + cx)
            } else {
                (lf_left + cx, lf_top + cy)
            };
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
