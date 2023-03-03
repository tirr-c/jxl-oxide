use jxl_frame::{
    data::{LfCoeff, CoeffData},
    FrameHeader,
};
use jxl_grid::{Grid, Subgrid};
use jxl_image::OpsinInverseMatrix;
use jxl_vardct::{
    LfChannelDequantization,
    Quantizer,
    DequantMatrixSet,
    LfChannelCorrelation, TransformType,
};

use crate::dct::dct_2d_in_place;

mod transform;
pub use transform::transform;

pub fn dequant_lf(
    frame_header: &FrameHeader,
    lf_dequant: &LfChannelDequantization,
    quantizer: &Quantizer,
    lf_coeff: &LfCoeff,
) -> [Grid<f32>; 3] { // [y, x, b]
    let subsampled = frame_header.jpeg_upsampling.into_iter().any(|x| x != 0);
    let do_smoothing = !frame_header.flags.skip_adaptive_lf_smoothing();

    let lf_y = 512.0 * lf_dequant.m_y_lf / quantizer.global_scale as f32 / quantizer.quant_lf as f32;
    let lf_x = 512.0 * lf_dequant.m_x_lf / quantizer.global_scale as f32 / quantizer.quant_lf as f32;
    let lf_b = 512.0 * lf_dequant.m_b_lf / quantizer.global_scale as f32 / quantizer.quant_lf as f32;
    let lf = [lf_y, lf_x, lf_b];

    let precision_scale = (-(lf_coeff.extra_precision as f32)).exp2();
    let channel_data = lf_coeff.lf_quant.image().channel_data();

    let mut it = channel_data.iter().zip(lf)
        .map(|(g, lf)| {
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

    // [y, x, b]
    let dq_channels = [
        it.next().unwrap(),
        it.next().unwrap(),
        it.next().unwrap(),
    ];

    if !do_smoothing {
        return dq_channels;
    }
    if subsampled {
        panic!();
    }

    // smoothing
    const SCALE_SELF: f32 = 0.052262735;
    const SCALE_SIDE: f32 = 0.2034514;
    const SCALE_DIAG: f32 = 0.03348292;

    let mut out = dq_channels.clone();
    let width = out[0].width();
    let height = out[0].height();

    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            let mut s_self = [0.0f32; 3];
            let mut s_side = [0.0f32; 3];
            let mut s_diag = [0.0f32; 3];
            for (idx, g) in dq_channels.iter().enumerate() {
                let g = g.as_simple().unwrap();
                let stride = g.width();
                let g = g.buf();
                let base_idx = y * stride + x;
                s_self[idx] = g[base_idx];
                s_side[idx] = g[base_idx - 1] + g[base_idx - stride] + g[base_idx + 1] + g[base_idx + stride];
                s_diag[idx] = g[base_idx - stride - 1] + g[base_idx - stride + 1] + g[base_idx + stride - 1] + g[base_idx + stride + 1];
            }
            let wa = [
                s_self[0] * SCALE_SELF + s_side[0] * SCALE_SIDE + s_diag[0] * SCALE_DIAG,
                s_self[1] * SCALE_SELF + s_side[1] * SCALE_SIDE + s_diag[1] * SCALE_DIAG,
                s_self[2] * SCALE_SELF + s_side[2] * SCALE_SIDE + s_diag[2] * SCALE_DIAG,
            ];
            let gap_t = [
                (wa[0] - s_self[0]).abs() / lf_y,
                (wa[1] - s_self[1]).abs() / lf_x,
                (wa[2] - s_self[2]).abs() / lf_b,
            ];
            let gap = gap_t.into_iter().fold(0.5f32, |acc, v| acc.max(v));
            let gap_scale = (3.0 - 4.0 * gap).max(0.0);
            for ((g, wa), s) in out.iter_mut().zip(wa).zip(s_self) {
                g.set(x, y, (wa - s) * gap_scale + s);
            }
        }
    }

    out
}

pub fn dequant_hf_varblock(
    coeff_data: &CoeffData,
    channel: usize,
    oim: &OpsinInverseMatrix,
    quantizer: &Quantizer,
    dequant_matrices: &DequantMatrixSet,
    qm_scale: Option<u32>,
) -> Grid<f32> {
    let CoeffData { dct_select, hf_mul, ref coeff } = *coeff_data;
    let coeff = &coeff[channel];
    let mut out = Grid::new_similar(coeff);

    let quant_bias = oim.quant_bias[channel];
    let quant_bias_numerator = oim.quant_bias_numerator;
    let matrix = dequant_matrices.get(channel, dct_select);
    for (y, mat_row) in matrix.iter().enumerate() {
        for (x, &mat) in mat_row.iter().enumerate() {
            let quant = *coeff.get(x, y).unwrap();
            let quant = if (-1..=1).contains(&quant) {
                quant as f32 * quant_bias
            } else {
                let q = quant as f32;
                q - (quant_bias_numerator / q)
            };

            let mul = 65536.0 / quantizer.global_scale as f32 / hf_mul as f32;
            let mut quant = quant * mul;

            if let Some(qm_scale) = qm_scale {
                let scale = 0.8f32.powi(qm_scale as i32 - 2);
                quant *= scale;
            }

            out.set(x, y, quant * mat);
        }
    }

    out
}

pub fn chroma_from_luma_lf(
    coeff_yxb: &mut [Grid<f32>; 3],
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

    let mut it = coeff_yxb.iter_mut();
    let y = it.next().unwrap();
    let x = it.next().unwrap();
    let b = it.next().unwrap();
    y.zip3_mut(x, b, |y, x, b| {
        let y = *y;
        *x += kx * y;
        *b += kb * y;
    });
}

pub fn chroma_from_luma_hf(
    coeff_yxb: &mut [Grid<f32>; 3],
    lf_left: usize,
    lf_top: usize,
    x_from_y: &Grid<i32>,
    b_from_y: &Grid<i32>,
    lf_chan_corr: &LfChannelCorrelation,
    transposed: bool,
) {
    let LfChannelCorrelation {
        colour_factor,
        base_correlation_x,
        base_correlation_b,
        ..
    } = *lf_chan_corr;

    let [coeff_y, coeff_x, coeff_b] = coeff_yxb;
    let width = coeff_y.width();
    let height = coeff_y.height();

    for cy in 0..height {
        for cx in 0..width {
            let (x, y) = if transposed {
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

            let coeff_y = *coeff_y.get(cx, cy).unwrap();
            *coeff_x.get_mut(cx, cy).unwrap() += kx * coeff_y;
            *coeff_b.get_mut(cx, cy).unwrap() += kb * coeff_y;
        }
    }
}

pub fn llf_from_lf(
    lf: Subgrid<'_, f32>,
    dct_select: TransformType,
) -> Grid<f32> {
    use TransformType::*;

    fn scale_f(c: usize, b: usize) -> f32 {
        let cb = c as f32 / b as f32;
        let recip = (cb * std::f32::consts::FRAC_PI_2).cos() *
            (cb * std::f32::consts::PI).cos() *
            (cb * 2.0 * std::f32::consts::PI).cos();
        recip.recip()
    }

    let (bw, bh) = dct_select.dct_select_size();
    debug_assert_eq!(bw as usize, lf.width());
    debug_assert_eq!(bh as usize, lf.height());

    if matches!(dct_select, Hornuss | Dct2 | Dct4 | Dct8x4 | Dct4x8 | Afv0 | Afv1 | Afv2 | Afv3) {
        debug_assert_eq!(bw * bh, 1);
        let mut out = Grid::new_usize(1, 1, 1, 1);
        out.set(0, 0, *lf.get(0, 0).unwrap());
        out
    } else {
        let mut llf = vec![0.0f32; bw as usize * bh as usize];
        for y in 0..bh as usize {
            for x in 0..bw as usize {
                llf[y * bw as usize + x] = *lf.get(x, y).unwrap();
            }
        }
        dct_2d_in_place(&mut llf, bw as usize, bh as usize);

        let cx = bw.max(bh) as usize;
        let cy = bw.min(bh) as usize;
        let mut out = Grid::new_usize(cx, cy, cx, cy);
        for y in 0..cy {
            for x in 0..cx {
                out.set(x, y, llf[y * cx + x] * scale_f(y, cy * 8) * scale_f(x, cx * 8));
            }
        }

        out
    }
}
