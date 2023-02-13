use jxl_bitstream::header::OpsinInverseMatrix;
use jxl_frame::{
    data::{LfCoeff, CoeffData},
    FrameHeader,
};
use jxl_grid::{Grid, Subgrid};
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
            let mut out = Grid::new(width, height, g.group_size());
            for y in 0..height {
                for x in 0..width {
                    let s = g[(x, y)] as f32;
                    out[(x, y)] = s * lf * precision_scale;
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
                s_self[idx] = g[(x, y)];
                s_side[idx] = g[(x - 1, y)] + g[(x, y - 1)] + g[(x + 1, y)] + g[(x, y + 1)];
                s_diag[idx] = g[(x - 1, y - 1)] + g[(x - 1, y + 1)] + g[(x + 1, y - 1)] + g[(x + 1, y + 1)];
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
                g[(x, y)] = (wa - s) * gap_scale + s;
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

    let width = coeff.width();
    let height = coeff.height();
    let mut out = Grid::new(width, height, coeff.group_size());

    let quant_bias = oim.quant_bias[channel];
    let quant_bias_numerator = oim.quant_bias_numerator;
    let matrix = dequant_matrices.get(channel, dct_select);
    for y in 0..height {
        for x in 0..width {
            let quant = coeff[(x, y)];
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

            out[(x, y)] = quant * matrix[y as usize][x as usize];
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
    let mut arr = [
        it.next().unwrap(),
        it.next().unwrap(),
        it.next().unwrap(),
    ];
    jxl_grid::zip_iterate(&mut arr, |samples| {
        let [y, x, b] = samples else { unreachable!() };
        let y = **y;
        **x += kx * y;
        **b += kb * y;
    });
}

pub fn chroma_from_luma_hf(
    coeff_yxb: &mut [Grid<f32>; 3],
    lf_left: u32,
    lf_top: u32,
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

            let x_factor = x_from_y[(cfactor_x, cfactor_y)];
            let b_factor = b_from_y[(cfactor_x, cfactor_y)];
            let kx = base_correlation_x + (x_factor as f32 / colour_factor as f32);
            let kb = base_correlation_b + (b_factor as f32 / colour_factor as f32);

            let coeff_y = coeff_y[(cx, cy)];
            coeff_x[(cx, cy)] += kx * coeff_y;
            coeff_b[(cx, cy)] += kb * coeff_y;
        }
    }
}

pub fn llf_from_lf(
    lf: Subgrid<'_, f32>,
    dct_select: TransformType,
) -> Grid<f32> {
    use TransformType::*;

    fn scale_f(c: u32, b: u32) -> f32 {
        let cb = c as f32 / b as f32;
        let recip = (cb * std::f32::consts::FRAC_PI_2).cos() *
            (cb * std::f32::consts::PI).cos() *
            (cb * 2.0 * std::f32::consts::PI).cos();
        recip.recip()
    }

    let (bw, bh) = dct_select.dct_select_size();
    debug_assert_eq!(bw, lf.width());
    debug_assert_eq!(bh, lf.height());

    if matches!(dct_select, Hornuss | Dct2 | Dct4 | Dct8x4 | Dct4x8 | Afv0 | Afv1 | Afv2 | Afv3) {
        debug_assert_eq!(bw * bh, 1);
        let mut out = Grid::new(1, 1, (1, 1));
        out[(0, 0)] = lf[(0, 0)];
        out
    } else {
        let mut llf = vec![0.0f32; bw as usize * bh as usize];
        for y in 0..bh {
            for x in 0..bw {
                llf[(y * bw + x) as usize] = lf[(x, y)];
            }
        }
        dct_2d_in_place(&mut llf, bw as usize, bh as usize);

        let cx = bw.max(bh);
        let cy = bw.min(bh);
        let mut out = Grid::new(cx, cy, (cx, cy));
        for y in 0..cy {
            for x in 0..cx {
                out[(x, y)] = llf[(y * cx + x) as usize] * scale_f(y, cy * 8) * scale_f(x, cx * 8);
            }
        }

        out
    }
}
