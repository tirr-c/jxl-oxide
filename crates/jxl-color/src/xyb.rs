use jxl_image::ImageMetadata;
use jxl_grid::SimpleGrid;

pub fn perform_inverse_xyb(fb_xyb: [&mut SimpleGrid<f32>; 3], metadata: &ImageMetadata) {
    let itscale = 255.0 / metadata.tone_mapping.intensity_target;
    let oim = &metadata.opsin_inverse_matrix;

    let ob = oim.opsin_bias;

    let inv_mat = oim.inv_mat;

    let [x, y, b] = fb_xyb;
    let x = x.buf_mut();
    let y = y.buf_mut();
    let b = b.buf_mut();
    if x.len() != y.len() || y.len() != b.len() {
        panic!("Grid size mismatch");
    }
    xyb_impl::run([x, y, b], ob, inv_mat, itscale);
}

#[cfg(
    not(
        all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "fma"
        )
    )
)]
mod xyb_impl {
    pub(super) fn run(
        xyb: [&mut [f32]; 3],
        ob: [f32; 3],
        inv_mat: [[f32; 3]; 3],
        itscale: f32,
    ) {
        let cbrt_ob = ob.map(|v| v.cbrt());
        let [x, y, b] = xyb;

        for ((x, y), b) in x.iter_mut().zip(y).zip(b) {
            let g_l = *y + *x;
            let g_m = *y - *x;
            let g_s = *b;

            let mix_l = ((g_l - cbrt_ob[0]).powi(3) + ob[0]) * itscale;
            let mix_m = ((g_m - cbrt_ob[1]).powi(3) + ob[1]) * itscale;
            let mix_s = ((g_s - cbrt_ob[2]).powi(3) + ob[2]) * itscale;

            *x = inv_mat[0][0] * mix_l + inv_mat[0][1] * mix_m + inv_mat[0][2] * mix_s;
            *y = inv_mat[1][0] * mix_l + inv_mat[1][1] * mix_m + inv_mat[1][2] * mix_s;
            *b = inv_mat[2][0] * mix_l + inv_mat[2][1] * mix_m + inv_mat[2][2] * mix_s;
        }
    }
}

#[cfg(
    all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "fma"
    )
)]
mod xyb_impl {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;

    pub(super) fn run(
        xyb: [&mut [f32]; 3],
        ob: [f32; 3],
        inv_mat: [[f32; 3]; 3],
        itscale: f32,
    ) {
        const WIDTH: usize = 4;

        let cbrt_ob = ob.map(|v| v.cbrt());
        let [x, y, b] = xyb;

        let mut it_x = x.chunks_exact_mut(WIDTH);
        let mut it_y = y.chunks_exact_mut(WIDTH);
        let mut it_b = b.chunks_exact_mut(WIDTH);
        let it = (&mut it_x).zip(&mut it_y).zip(&mut it_b);

        unsafe {
            let ob_lane = ob.map(|v| _mm_set1_ps(v));
            let cbrt_ob_lane = cbrt_ob.map(|v| _mm_set1_ps(v));
            let inv_mat_lane = inv_mat.map(|row| row.map(|v| _mm_set1_ps(v)));
            let itscale_lane = _mm_set1_ps(itscale);

            for ((xlane, ylane), blane) in it {
                let x = _mm_loadu_ps(xlane.as_ptr());
                let y = _mm_loadu_ps(ylane.as_ptr());
                let b = _mm_loadu_ps(blane.as_ptr());
                let g_l = _mm_add_ps(y, x);
                let g_m = _mm_sub_ps(y, x);
                let g_s = b;

                let mix_l = {
                    let interm = _mm_sub_ps(g_l, cbrt_ob_lane[0]);
                    let sq = _mm_mul_ps(interm, interm);
                    let mix = _mm_fmadd_ps(sq, interm, ob_lane[0]);
                    _mm_mul_ps(mix, itscale_lane)
                };
                let mix_m = {
                    let interm = _mm_sub_ps(g_m, cbrt_ob_lane[1]);
                    let sq = _mm_mul_ps(interm, interm);
                    let mix = _mm_fmadd_ps(sq, interm, ob_lane[1]);
                    _mm_mul_ps(mix, itscale_lane)
                };
                let mix_s = {
                    let interm = _mm_sub_ps(g_s, cbrt_ob_lane[2]);
                    let sq = _mm_mul_ps(interm, interm);
                    let mix = _mm_fmadd_ps(sq, interm, ob_lane[2]);
                    _mm_mul_ps(mix, itscale_lane)
                };

                let x = _mm_mul_ps(mix_l, inv_mat_lane[0][0]);
                let x = _mm_fmadd_ps(mix_m, inv_mat_lane[0][1], x);
                let x = _mm_fmadd_ps(mix_s, inv_mat_lane[0][2], x);
                _mm_store_ps(xlane.as_mut_ptr(), x);
                let y = _mm_mul_ps(mix_l, inv_mat_lane[1][0]);
                let y = _mm_fmadd_ps(mix_m, inv_mat_lane[1][1], y);
                let y = _mm_fmadd_ps(mix_s, inv_mat_lane[1][2], y);
                _mm_store_ps(ylane.as_mut_ptr(), y);
                let b = _mm_mul_ps(mix_l, inv_mat_lane[2][0]);
                let b = _mm_fmadd_ps(mix_m, inv_mat_lane[2][1], b);
                let b = _mm_fmadd_ps(mix_s, inv_mat_lane[2][2], b);
                _mm_store_ps(blane.as_mut_ptr(), b);
            }
        }

        let x = it_x.into_remainder();
        let y = it_y.into_remainder();
        let b = it_b.into_remainder();
        for ((x, y), b) in x.iter_mut().zip(y).zip(b) {
            let g_l = *y + *x;
            let g_m = *y - *x;
            let g_s = *b;

            let mix_l = ((g_l - cbrt_ob[0]).powi(3) + ob[0]) * itscale;
            let mix_m = ((g_m - cbrt_ob[1]).powi(3) + ob[1]) * itscale;
            let mix_s = ((g_s - cbrt_ob[2]).powi(3) + ob[2]) * itscale;

            *x = inv_mat[0][0] * mix_l + inv_mat[0][1] * mix_m + inv_mat[0][2] * mix_s;
            *y = inv_mat[1][0] * mix_l + inv_mat[1][1] * mix_m + inv_mat[1][2] * mix_s;
            *b = inv_mat[2][0] * mix_l + inv_mat[2][1] * mix_m + inv_mat[2][2] * mix_s;
        }
    }
}
