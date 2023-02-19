use jxl_bitstream::header::ImageMetadata;

use crate::FrameBuffer;

pub fn perform_inverse_xyb(fb_yxb: &mut FrameBuffer, metadata: &ImageMetadata) {
    let itscale = 255.0 / metadata.tone_mapping.intensity_target;
    let oim = &metadata.opsin_inverse_matrix;

    let ob = oim.opsin_bias;
    let cbrt_ob = ob.map(|v| v.cbrt());

    let inv_mat = oim.inv_mat;

    let width = fb_yxb.width() as usize;
    let height = fb_yxb.height() as usize;
    let stride = fb_yxb.stride() as usize;

    let mut channels = fb_yxb.channel_buffers_mut().into_iter();
    let y = channels.next().unwrap();
    let x = channels.next().unwrap();
    let b = channels.next().unwrap();
    for r in 0..height {
        for c in 0..width {
            let idx = r * stride + c;
            let y = &mut y[idx];
            let x = &mut x[idx];
            let b = &mut b[idx];
            let g_l = *y + *x;
            let g_m = *y - *x;
            let g_s = *b;

            let mix_l = ((g_l - cbrt_ob[0]).powi(3) + ob[0]) * itscale;
            let mix_m = ((g_m - cbrt_ob[1]).powi(3) + ob[1]) * itscale;
            let mix_s = ((g_s - cbrt_ob[2]).powi(3) + ob[2]) * itscale;

            *y = inv_mat[0][0] * mix_l + inv_mat[0][1] * mix_m + inv_mat[0][2] * mix_s;
            *x = inv_mat[1][0] * mix_l + inv_mat[1][1] * mix_m + inv_mat[1][2] * mix_s;
            *b = inv_mat[2][0] * mix_l + inv_mat[2][1] * mix_m + inv_mat[2][2] * mix_s;
        }
    }
}

pub fn perform_inverse_ycbcr(fb_ycbcr: &mut FrameBuffer) {
    let width = fb_ycbcr.width() as usize;
    let height = fb_ycbcr.height() as usize;
    let stride = fb_ycbcr.stride() as usize;

    let mut channels = fb_ycbcr.channel_buffers_mut().into_iter();
    let y = channels.next().unwrap();
    let cb = channels.next().unwrap();
    let cr = channels.next().unwrap();
    for r in 0..height {
        for c in 0..width {
            let idx = r * stride + c;
            let r = &mut y[idx];
            let g = &mut cb[idx];
            let b = &mut cr[idx];
            let y = *r + 0.5;
            let cb = *g;
            let cr = *b;

            *r = y + 1.402 * cr;
            *g = y - 0.344016 * cb - 0.714136 * cr;
            *b = y + 1.772 * cb;
        }
    }
}
