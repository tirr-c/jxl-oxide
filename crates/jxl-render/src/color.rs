use jxl_bitstream::header::ImageMetadata;
use jxl_grid::Grid;
use jxl_vardct::LfChannelDequantization;

pub fn perform_inverse_xyb_modular(yxb: [&Grid<i32>; 3], lf_dequant: &LfChannelDequantization, metadata: &ImageMetadata) -> [Grid<f32>; 3] {
    let mut yxb_unscaled = yxb.map(|g| Grid::new(g.width(), g.height(), g.group_size()));
    let m_lf_unscaled = [
        lf_dequant.m_y_lf / 128.0,
        lf_dequant.m_x_lf / 128.0,
        lf_dequant.m_b_lf / 128.0,
    ];

    for ((target, grid), m_lf) in yxb_unscaled.iter_mut().zip(yxb).zip(m_lf_unscaled) {
        let width = grid.width();
        let height = grid.height();
        for y in 0..height {
            for x in 0..width {
                target[(x, y)] = grid[(x, y)] as f32 * m_lf;
            }
        }
    }

    let [y, x, b] = &mut yxb_unscaled;
    perform_inverse_xyb([y, x, b], metadata);
    yxb_unscaled
}

pub fn perform_inverse_xyb(mut yxb: [&mut Grid<f32>; 3], metadata: &ImageMetadata) {
    let itscale = 255.0 / metadata.tone_mapping.intensity_target;
    let oim = &metadata.opsin_inverse_matrix;

    let ob = oim.opsin_bias;
    let cbrt_ob = ob.map(|v| v.cbrt());

    let inv_mat = oim.inv_mat;

    jxl_grid::zip_iterate(&mut yxb, |yxb| {
        let [y, x, b] = yxb else { unreachable!() };
        let g_l = **y + **x;
        let g_m = **y - **x;
        let g_s = **b;

        let mix_l = ((g_l - cbrt_ob[0]).powi(3) + ob[0]) * itscale;
        let mix_m = ((g_m - cbrt_ob[1]).powi(3) + ob[1]) * itscale;
        let mix_s = ((g_s - cbrt_ob[2]).powi(3) + ob[2]) * itscale;

        **y = inv_mat[0][0] * mix_l + inv_mat[0][1] * mix_m + inv_mat[0][2] * mix_s;
        **x = inv_mat[1][0] * mix_l + inv_mat[1][1] * mix_m + inv_mat[1][2] * mix_s;
        **b = inv_mat[2][0] * mix_l + inv_mat[2][1] * mix_m + inv_mat[2][2] * mix_s;
    });
}
