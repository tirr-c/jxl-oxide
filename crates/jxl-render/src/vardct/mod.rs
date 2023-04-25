use jxl_color::OpsinInverseMatrix;
use jxl_frame::data::{LfCoeff, CoeffData};
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
    lf_dequant: &LfChannelDequantization,
    quantizer: &Quantizer,
    lf_coeff: &LfCoeff,
) -> [Grid<f32>; 3] { // [x, y, b]
    let lf_x = 512.0 * lf_dequant.m_x_lf / quantizer.global_scale as f32 / quantizer.quant_lf as f32;
    let lf_y = 512.0 * lf_dequant.m_y_lf / quantizer.global_scale as f32 / quantizer.quant_lf as f32;
    let lf_b = 512.0 * lf_dequant.m_b_lf / quantizer.global_scale as f32 / quantizer.quant_lf as f32;
    let lf = [lf_x, lf_y, lf_b];

    let precision_scale = (-(lf_coeff.extra_precision as f32)).exp2();
    let channel_data = lf_coeff.lf_quant.image().channel_data();
    tracing::trace!(val0 = channel_data[1].get(203, 229), val1 = channel_data[0].get(203, 229), val2 = channel_data[2].get(203, 229));

    // the first two channels are flipped (YXB)
    let mut it = [1, 0, 2].into_iter().zip(lf)
        .map(|(c, lf)| {
            let g = &channel_data[c];
            let width = g.width();
            let height = g.height();
            let mut out = Grid::new_similar(g);
            for y in 0..height {
                for x in 0..width {
                    let s = *g.get(x, y).unwrap() as f32;
                    out.set(x, y, s * lf * precision_scale);
                }
            }
            out
        });

    let dq_channels = [
        it.next().unwrap(),
        it.next().unwrap(),
        it.next().unwrap(),
    ];

    tracing::trace!(val0 = dq_channels[0].get(203, 229), val1 = dq_channels[1].get(203, 229), val2 = dq_channels[2].get(203, 229));
    dq_channels
}

pub fn adaptive_lf_smoothing(
    lf_image: &[Grid<f32>; 3],
    out: &mut [SimpleGrid<f32>; 3],
    lf_dequant: &LfChannelDequantization,
    quantizer: &Quantizer,
) {
    // smoothing
    const SCALE_SELF: f32 = 0.052262735;
    const SCALE_SIDE: f32 = 0.2034514;
    const SCALE_DIAG: f32 = 0.03348292;

    let lf_x = 512.0 * lf_dequant.m_x_lf / quantizer.global_scale as f32 / quantizer.quant_lf as f32;
    let lf_y = 512.0 * lf_dequant.m_y_lf / quantizer.global_scale as f32 / quantizer.quant_lf as f32;
    let lf_b = 512.0 * lf_dequant.m_b_lf / quantizer.global_scale as f32 / quantizer.quant_lf as f32;

    let width = out[0].width();
    let height = out[0].height();

    for y in 0..height {
        for x in 0..width {
            if x == 0 || y == 0 || x == width - 1 || y == height - 1 {
                out[0].buf_mut()[y * width + x] = lf_image[0].get(x, y).copied().unwrap_or(0.0);
                out[1].buf_mut()[y * width + x] = lf_image[1].get(x, y).copied().unwrap_or(0.0);
                out[2].buf_mut()[y * width + x] = lf_image[2].get(x, y).copied().unwrap_or(0.0);
                continue;
            }

            let mut s_self = [0.0f32; 3];
            let mut s_side = [0.0f32; 3];
            let mut s_diag = [0.0f32; 3];
            for (idx, g) in lf_image.iter().enumerate() {
                s_self[idx] = g.get(x, y).copied().unwrap_or(0.0);
                s_side[idx] = g.get(x, y - 1).copied().unwrap_or(0.0)
                    + g.get(x - 1, y).copied().unwrap_or(0.0)
                    + g.get(x + 1, y).copied().unwrap_or(0.0)
                    + g.get(x, y + 1).copied().unwrap_or(0.0);
                s_diag[idx] = g.get(x - 1, y - 1).copied().unwrap_or(0.0)
                    + g.get(x + 1, y - 1).copied().unwrap_or(0.0)
                    + g.get(x - 1, y + 1).copied().unwrap_or(0.0)
                    + g.get(x + 1, y + 1).copied().unwrap_or(0.0);
            }
            let wa = [
                s_self[0] * SCALE_SELF + s_side[0] * SCALE_SIDE + s_diag[0] * SCALE_DIAG,
                s_self[1] * SCALE_SELF + s_side[1] * SCALE_SIDE + s_diag[1] * SCALE_DIAG,
                s_self[2] * SCALE_SELF + s_side[2] * SCALE_SIDE + s_diag[2] * SCALE_DIAG,
            ];
            let gap_t = [
                (wa[0] - s_self[0]).abs() / lf_x,
                (wa[1] - s_self[1]).abs() / lf_y,
                (wa[2] - s_self[2]).abs() / lf_b,
            ];
            let gap = gap_t.into_iter().fold(0.5f32, |acc, v| acc.max(v));
            let gap_scale = (3.0 - 4.0 * gap).max(0.0);
            for ((g, wa), s) in out.iter_mut().zip(wa).zip(s_self) {
                g.buf_mut()[y * width + x] = (wa - s) * gap_scale + s;
            }
        }
    }
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
    coeff_xyb: &mut [Grid<f32>; 3],
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

    let mut it = coeff_xyb.iter_mut();
    let x = it.next().unwrap();
    let y = it.next().unwrap();
    let b = it.next().unwrap();
    x.zip3_mut(y, b, |x, y, b| {
        let y = *y;
        *x += kx * y;
        *b += kb * y;
    });
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
