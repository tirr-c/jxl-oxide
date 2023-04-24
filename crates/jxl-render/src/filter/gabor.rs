use jxl_grid::SimpleGrid;

pub fn apply_gabor_like(fb: [&mut SimpleGrid<f32>; 3], weights_xyb: [[f32; 2]; 3]) {
    tracing::debug!("Running gaborish");

    let width = fb[0].width();
    let height = fb[0].height();
    let mut ud_sums = Vec::with_capacity(width * height);

    let buffers = fb.map(|g| g.buf_mut());
    for (c, [weight1, weight2]) in buffers.into_iter().zip(weights_xyb) {
        ud_sums.clear();
        let rows: Vec<_> = c.chunks_exact(width).collect();

        for y in 0..height {
            let up = rows[y.saturating_sub(1)];
            let down = rows[(y + 1).min(height - 1)];
            for (u, d) in up.iter().zip(down) {
                ud_sums.push(*u + *d);
            }
        }

        let global_weight = (1.0 + weight1 * 4.0 + weight2 * 4.0).recip();
        for y in 0..height {
            let mut left = c[y * width];
            for x in 0..width {
                let x_l = x.saturating_sub(1);
                let x_r = (x + 1).min(width - 1);
                let side = left + c[y * width + x_r] + ud_sums[y * width + x];
                let diag = ud_sums[y * width + x_l] + ud_sums[y * width + x_r];
                left = c[y * width + x];
                c[y * width + x] += side * weight1 + diag * weight2;
                c[y * width + x] *= global_weight;
            }
        }
    }
}
