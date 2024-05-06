use jxl_grid::AlignedGrid;

pub fn apply_jpeg_upsampling(grids_cbycr: [&mut AlignedGrid<f32>; 3], jpeg_upsampling: [u32; 3]) {
    fn interpolate(left: f32, center: f32, right: f32) -> (f32, f32) {
        (0.25 * left + 0.75 * center, 0.75 * center + 0.25 * right)
    }

    let shifts_cbycr =
        [0, 1, 2].map(|idx| jxl_modular::ChannelShift::from_jpeg_upsampling(jpeg_upsampling, idx));

    for (buf, shift) in grids_cbycr.into_iter().zip(shifts_cbycr) {
        let width = buf.width();
        let height = buf.height();
        let buf = buf.buf_mut();

        let h_upsampled = shift.hshift() == 0;
        let v_upsampled = shift.vshift() == 0;

        if !h_upsampled {
            let orig_width = width;
            let width = (width + 1) / 2;
            let height = if v_upsampled {
                height
            } else {
                (height + 1) / 2
            };

            for y in 0..height {
                let idx_base = y * orig_width;
                let mut prev_sample = buf[idx_base + width - 1];
                for x in (0..width).rev() {
                    let curr_sample = buf[idx_base + x];
                    let left_x = x.saturating_sub(1);

                    // We're interpolating right-to-left.
                    let (right, left) =
                        interpolate(prev_sample, curr_sample, buf[idx_base + left_x]);

                    buf[idx_base + x * 2] = left;
                    if x * 2 + 1 < orig_width {
                        buf[idx_base + x * 2 + 1] = right;
                    }

                    prev_sample = curr_sample;
                }
            }
        }

        // image is horizontally upsampled here
        if !v_upsampled {
            let orig_height = height;
            let height = (height + 1) / 2;

            let mut prev_row = buf[(height - 1) * width..][..width].to_vec();
            for y in (0..height).rev() {
                let idx_base = y * width;
                let top_base = idx_base.saturating_sub(width);
                for x in 0..width {
                    let curr_sample = buf[idx_base + x];

                    // We're interpolating bottom-to-top.
                    let (bottom, top) = interpolate(prev_row[x], curr_sample, buf[top_base + x]);
                    buf[idx_base * 2 + x] = top;
                    if y * 2 + 1 < orig_height {
                        buf[idx_base * 2 + width + x] = bottom;
                    }

                    prev_row[x] = curr_sample;
                }
            }
        }
    }
}
