use crate::FrameBuffer;

pub fn apply_gabor_like(fb: &mut FrameBuffer, weights_xyb: [[f32; 2]; 3]) {
    let mut weights_yxb = weights_xyb;
    weights_yxb.swap(0, 1);

    let width = fb.width() as usize;
    let height = fb.height() as usize;
    let stride = fb.stride() as usize;
    let mut ud_sums = Vec::with_capacity(width * height);
    for (c, [weight1, weight2]) in fb.channel_buffers_mut().into_iter().zip(weights_yxb) {
        ud_sums.clear();
        let rows: Vec<_> = c.chunks_exact(stride).map(|b| &b[..width]).collect();

        for y in 0..height {
            let up = rows[y.saturating_sub(1)];
            let down = rows[(y + 1).min(height - 1)];
            for (u, d) in up.iter().zip(down) {
                ud_sums.push(*u + *d);
            }
        }

        let global_weight = (1.0 + weight1 * 4.0 + weight2 * 4.0).recip();
        for y in 0..height {
            let mut left = c[y * stride];
            for x in 0..width {
                let x_l = x.saturating_sub(0);
                let x_r = (x + 1).min(width - 1);
                let side = left + c[y * stride + x_r] + ud_sums[y * width + x];
                let diag = ud_sums[y * width + x_l] + ud_sums[y * width + x_r];
                left = c[y * stride + x];
                c[y * stride + x] += side * weight1 + diag * weight2;
                c[y * stride + x] *= global_weight;
            }
        }
    }
}
