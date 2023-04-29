use jxl_grid::SimpleGrid;

pub fn perform_inverse_xyb(
    fb_xyb: [&mut SimpleGrid<f32>; 3],
    oim: &crate::OpsinInverseMatrix,
    intensity_target: f32,
) {
    let itscale = 255.0 / intensity_target;
    let ob = oim.opsin_bias;
    let inv_mat = oim.inv_mat;

    let [x, y, b] = fb_xyb;
    let x = x.buf_mut();
    let y = y.buf_mut();
    let b = b.buf_mut();
    let xyb = [x, y, b];

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma") {
            // SAFETY: Feature set is checked above.
            return unsafe { run_x86_64_avx2(xyb, ob, inv_mat, itscale) };
        }
    }

    run_generic(xyb, ob, inv_mat, itscale)
}

#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
unsafe fn run_x86_64_avx2(
    xyb: [&mut [f32]; 3],
    ob: [f32; 3],
    inv_mat: [[f32; 3]; 3],
    itscale: f32,
) {
    run_generic(xyb, ob, inv_mat, itscale)
}

#[inline(always)]
fn run_generic(
    xyb: [&mut [f32]; 3],
    ob: [f32; 3],
    inv_mat: [[f32; 3]; 3],
    itscale: f32,
) {
    let [x, y, b] = xyb;
    if x.len() != y.len() || y.len() != b.len() {
        panic!("Grid size mismatch");
    }
    let cbrt_ob = ob.map(|v| v.cbrt());

    for ((x, y), b) in x.iter_mut().zip(y).zip(b) {
        let g_l = *y + *x;
        let g_m = *y - *x;
        let g_s = *b;

        let g_l = g_l - cbrt_ob[0];
        let g_m = g_m - cbrt_ob[1];
        let g_s = g_s - cbrt_ob[2];

        let mix_l = (g_l * g_l * g_l + ob[0]) * itscale;
        let mix_m = (g_m * g_m * g_m + ob[1]) * itscale;
        let mix_s = (g_s * g_s * g_s + ob[2]) * itscale;

        *x = inv_mat[0][0] * mix_l + inv_mat[0][1] * mix_m + inv_mat[0][2] * mix_s;
        *y = inv_mat[1][0] * mix_l + inv_mat[1][1] * mix_m + inv_mat[1][2] * mix_s;
        *b = inv_mat[2][0] * mix_l + inv_mat[2][1] * mix_m + inv_mat[2][2] * mix_s;
    }
}
