use jxl_bitstream::header::OpsinInverseMatrix;
use jxl_frame::{data::{LfCoeff, CoeffData}, FrameHeader};
use jxl_grid::Grid;
use jxl_vardct::{LfChannelDequantization, Quantizer, TransformType, DequantMatrixSet, LfChannelCorrelation};

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
    let mut dq_channels = [
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

    let width = dq_channels[0].width();
    let height = dq_channels[0].height();

    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            let mut s_self = [&mut 0.0f32, &mut 0.0f32, &mut 0.0f32];
            let mut s_side = [0.0f32; 3];
            let mut s_diag = [0.0f32; 3];
            for (idx, g) in dq_channels.iter_mut().enumerate() {
                s_side[idx] = g[(x - 1, y)] + g[(x, y - 1)] + g[(x + 1, y)] + g[(x, y + 1)];
                s_diag[idx] = g[(x - 1, y - 1)] + g[(x - 1, y + 1)] + g[(x + 1, y - 1)] + g[(x + 1, y + 1)];
                s_self[idx] = &mut g[(x, y)];
            }
            let wa = [
                *s_self[0] * SCALE_SELF + s_side[0] * SCALE_SIDE + s_diag[0] * SCALE_DIAG,
                *s_self[1] * SCALE_SELF + s_side[1] * SCALE_SIDE + s_diag[1] * SCALE_DIAG,
                *s_self[2] * SCALE_SELF + s_side[2] * SCALE_SIDE + s_diag[2] * SCALE_DIAG,
            ];
            let gap_t = [
                (wa[0] - *s_self[0]).abs() / lf_y,
                (wa[1] - *s_self[1]).abs() / lf_y,
                (wa[2] - *s_self[2]).abs() / lf_y,
            ];
            let gap = gap_t.into_iter().fold(0.5f32, |acc, v| acc.max(v));
            let gap_scale = (3.0 - 4.0 * gap).max(0.0);
            for (s, wa) in s_self.into_iter().zip(wa) {
                *s = (wa - *s) * gap_scale + *s;
            }
        }
    }

    dq_channels
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

            let mul = 65536.0 * quantizer.global_scale as f32 / hf_mul as f32;
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
