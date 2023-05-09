use jxl_frame::FrameHeader;
use jxl_grid::{SimpleGrid, PaddedGrid};

pub fn upsample(
    grid: &mut SimpleGrid<f32>,
    image_header: &jxl_image::Headers,
    frame_header: &FrameHeader,
    channel_idx: usize,
) {
    let metadata = &image_header.metadata;
    let frame_width = frame_header.width;
    let frame_height = frame_header.height;

    let factor;
    if let Some(ec_idx) = channel_idx.checked_sub(3) {
        let dim_shift = image_header.metadata.ec_info[ec_idx].dim_shift;
        if dim_shift > 0 {
            tracing::debug!(channel_idx, dim_shift, "Applying non-separable upsampling for extra channel");

            let up8 = dim_shift / 3;
            let last_up = dim_shift % 3;

            let mut width = grid.width() as u32;
            let mut height = grid.height() as u32;
            for _ in 0..up8 {
                width = (width << 3).min(frame_width);
                height = (height << 3).min(frame_height);
                upsample_inner::<8, 210>(grid, width, height, &metadata.up8_weight);
            }
            width = (width << last_up).min(frame_width);
            height = (height << last_up).min(frame_height);
            match last_up {
                1 => upsample_inner::<2, 15>(grid, width, height, &metadata.up2_weight),
                2 => upsample_inner::<4, 55>(grid, width, height, &metadata.up4_weight),
                _ => {},
            }
        }

        factor = frame_header.ec_upsampling[ec_idx];
    } else {
        factor = frame_header.upsampling;
    };

    if factor != 1 {
        tracing::debug!(channel_idx, factor, "Applying non-separable upsampling");
    }

    match factor {
        1 => {},
        2 => upsample_inner::<2, 15>(grid, frame_width, frame_height, &metadata.up2_weight),
        4 => upsample_inner::<4, 55>(grid, frame_width, frame_height, &metadata.up4_weight),
        8 => upsample_inner::<8, 210>(grid, frame_width, frame_height, &metadata.up8_weight),
        _ => panic!("invalid upsampling factor {}", factor),
    }
}

fn upsample_inner<const K: usize, const NW: usize>(
    grid: &mut SimpleGrid<f32>,
    frame_width: u32,
    frame_height: u32,
    weights: &[f32; NW],
) {
    assert!(
        (K == 2 && NW == 15) ||
        (K == 4 && NW == 55) ||
        (K == 8 && NW == 210)
    );
    let frame_width = frame_width as usize;
    let frame_height = frame_height as usize;
    let grid_width = grid.width();
    let grid_height = grid.height();

    // 5x5 kernel
    const PADDING: usize = 2;
    let mut padded = PaddedGrid::new(grid_width, grid_height, PADDING);
    let padded_width = grid_width + PADDING * 2;

    let grid_buf = grid.buf();
    let padded_buf = padded.buf_padded_mut();
    for y in 0..grid.height() {
        padded_buf[(y + PADDING) * padded_width + PADDING..][..grid_width]
            .copy_from_slice(&grid_buf[y * grid_width..][..grid_width]);
    }
    padded.mirror_edges_padding();

    // For K = 8:
    // 0  1  2  3  3h 2h 1h 0h
    // 4  5  6  7  7h 6h 5h 4h
    // 8  9  a  b  bh ah 9h 8h
    // c  d  e  f  fh eh dh ch
    // cv dv ev fv f' e' d' c'
    // 8v 9v av bv b' a' 9' 8'
    // 4v 5v 6v 7v 7' 6' 5' 4'
    // 0v 1v 2v 3v 3' 2' 1' 0'
    let mut weights_quarter = vec![[0.0f32; 25]; K * K / 4];
    let mut weight_idx = 0usize;
    let mat_n = K / 2;
    for y in 0..5 * mat_n {
        let mat_y = y / 5;
        let ky = y % 5;
        for x in y..5 * mat_n {
            let mat_x = x / 5;
            let kx = x % 5;
            let w = weights[weight_idx];
            weight_idx += 1;

            weights_quarter[mat_y * mat_n + mat_x][ky * 5 + kx] = w;
            weights_quarter[mat_x * mat_n + mat_y][kx * 5 + ky] = w;
        }
    }

    *grid = SimpleGrid::new(frame_width, frame_height);
    let padded_buf = padded.buf_padded();
    let grid_buf = grid.buf_mut();
    for y in 0..frame_height {
        let ref_y = y / K;
        let mat_y = (y % K).min(K - y % K - 1);
        let flip_v = y % K >= mat_n;
        for x in 0..frame_width {
            let ref_x = x / K;
            let mat_x = (x % K).min(K - x % K - 1);
            let flip_h = x % K >= mat_n;

            let kernel = &weights_quarter[mat_y * mat_n + mat_x];
            let mut sum = 0.0f32;
            let mut min = f32::INFINITY;
            let mut max = -f32::INFINITY;
            for iy in 0..5 {
                let ky = if flip_v { 4 - iy } else { iy };
                for ix in 0..5 {
                    let kx = if flip_h { 4 - ix } else { ix };
                    let sample = padded_buf[(ref_y + iy) * padded_width + (ref_x + ix)];
                    sum += kernel[ky * 5 + kx] * sample;
                    min = min.min(sample);
                    max = max.max(sample);
                }
            }
            grid_buf[y * frame_width + x] = sum.clamp(min, max);
        }
    }
}
