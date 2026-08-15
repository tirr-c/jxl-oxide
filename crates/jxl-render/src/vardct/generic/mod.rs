use jxl_grid::{AlignedGrid, AllocTracker};
use jxl_modular::ChannelShift;

mod dct;
mod transform;
#[allow(unused)]
pub use dct::dct_2d;
#[allow(unused)]
pub use transform::*;

const SCALE_SELF: f32 = 0.052262735;
const SCALE_SIDE: f32 = 0.2034514;
const SCALE_DIAG: f32 = 0.03348292;

#[inline(always)]
pub fn adaptive_lf_smoothing_impl(
    width: usize,
    height: usize,
    shifts: [ChannelShift; 3],
    [in_x, in_y, in_b]: [&mut [f32]; 3],
    [lf_x, lf_y, lf_b]: [f32; 3],
    tracker: Option<&AllocTracker>,
) -> crate::Result<()> {
    if width <= 2 || height <= 2 {
        // Nothing to do
        return Ok(());
    }

    if shifts.map(|x| x.hshift() + x.vshift()) != [0; 3] {
        return adaptive_lf_smoothing_subsampled_impl(
            width,
            height,
            shifts,
            [in_x, in_y, in_b],
            [lf_x, lf_y, lf_b],
            tracker,
        );
    }

    assert_eq!(in_x.len(), in_y.len());
    assert_eq!(in_y.len(), in_b.len());
    assert_eq!(in_x.len(), width * height);

    let mut udsum_x = AlignedGrid::with_alloc_tracker(width, height - 2, tracker)?;
    let mut udsum_y = AlignedGrid::with_alloc_tracker(width, height - 2, tracker)?;
    let mut udsum_b = AlignedGrid::with_alloc_tracker(width, height - 2, tracker)?;

    for (g, out) in [
        (&mut *in_x, udsum_x.buf_mut()),
        (&mut *in_y, udsum_y.buf_mut()),
        (&mut *in_b, udsum_b.buf_mut()),
    ] {
        let up = g.chunks_exact(width);
        let down = g[width * 2..].chunks_exact(width);
        let out = out.chunks_exact_mut(width);
        for ((up, down), out) in up.zip(down).zip(out) {
            for ((&u, &d), out) in up.iter().zip(down).zip(out) {
                *out = u + d;
            }
        }
    }

    let mut in_x_row = in_x.chunks_exact_mut(width).skip(1);
    let mut in_y_row = in_y.chunks_exact_mut(width).skip(1);
    let mut in_b_row = in_b.chunks_exact_mut(width).skip(1);

    let udsum_x_row = udsum_x.buf_mut().chunks_exact(width);
    let mut udsum_y_row = udsum_y.buf_mut().chunks_exact(width);
    let mut udsum_b_row = udsum_b.buf_mut().chunks_exact(width);

    for udsum_x in udsum_x_row {
        let udsum_y = udsum_y_row.next().unwrap();
        let udsum_b = udsum_b_row.next().unwrap();
        let in_x = in_x_row.next().unwrap();
        let in_y = in_y_row.next().unwrap();
        let in_b = in_b_row.next().unwrap();

        let mut in_x_prev = in_x[0];
        let mut in_y_prev = in_y[0];
        let mut in_b_prev = in_b[0];
        for x in 1..(width - 1) {
            let x_self = in_x[x];
            let x_side = in_x_prev + in_x[x + 1] + udsum_x[x];
            let x_diag = udsum_x[x - 1] + udsum_x[x + 1];
            let x_wa = x_self * SCALE_SELF + x_side * SCALE_SIDE + x_diag * SCALE_DIAG;
            let x_gap_t = (x_wa - x_self).abs() / lf_x;

            let y_self = in_y[x];
            let y_side = in_y_prev + in_y[x + 1] + udsum_y[x];
            let y_diag = udsum_y[x - 1] + udsum_y[x + 1];
            let y_wa = y_self * SCALE_SELF + y_side * SCALE_SIDE + y_diag * SCALE_DIAG;
            let y_gap_t = (y_wa - y_self).abs() / lf_y;

            let b_self = in_b[x];
            let b_side = in_b_prev + in_b[x + 1] + udsum_b[x];
            let b_diag = udsum_b[x - 1] + udsum_b[x + 1];
            let b_wa = b_self * SCALE_SELF + b_side * SCALE_SIDE + b_diag * SCALE_DIAG;
            let b_gap_t = (b_wa - b_self).abs() / lf_b;

            let gap = 0.5f32.max(x_gap_t).max(y_gap_t).max(b_gap_t);
            let gap_scale = (3.0 - 4.0 * gap).max(0.0);

            in_x[x] = (x_wa - x_self) * gap_scale + x_self;
            in_y[x] = (y_wa - y_self) * gap_scale + y_self;
            in_b[x] = (b_wa - b_self) * gap_scale + b_self;
            in_x_prev = x_self;
            in_y_prev = y_self;
            in_b_prev = b_self;
        }
    }

    Ok(())
}

fn adaptive_lf_smoothing_subsampled_impl(
    width: usize,
    height: usize,
    shifts: [ChannelShift; 3],
    in_xyb: [&mut [f32]; 3],
    [lf_x, lf_y, lf_b]: [f32; 3],
    tracker: Option<&AllocTracker>,
) -> crate::Result<()> {
    let dims = shifts.map(|shift| {
        let (w, h) = shift.shift_size((width as u32, height as u32));
        (w as usize, h as usize)
    });
    for (w, h) in dims {
        if w <= 2 || h <= 2 {
            return Ok(());
        }
    }

    for ((w, h), in_lf) in std::iter::zip(dims, &in_xyb) {
        assert_eq!(in_lf.len(), w * h);
    }

    let [in_x, in_y, in_b] = in_xyb;

    let mut wa_x = AlignedGrid::with_alloc_tracker(dims[0].0 - 2, dims[0].1 - 2, tracker)?;
    let mut wa_y = AlignedGrid::with_alloc_tracker(dims[1].0 - 2, dims[1].1 - 2, tracker)?;
    let mut wa_b = AlignedGrid::with_alloc_tracker(dims[2].0 - 2, dims[2].1 - 2, tracker)?;

    for (width, g, out) in [
        (dims[0].0, &mut *in_x, wa_x.buf_mut()),
        (dims[1].0, &mut *in_y, wa_y.buf_mut()),
        (dims[2].0, &mut *in_b, wa_b.buf_mut()),
    ] {
        let up = g.chunks_exact(width);
        let center = g[width..].chunks_exact(width);
        let down = g[width * 2..].chunks_exact(width);
        let out = out.chunks_exact_mut(width - 2);
        for (((up, center), down), out) in up.zip(center).zip(down).zip(out) {
            for (((u, c), d), out) in up
                .windows(3)
                .zip(center.windows(3))
                .zip(down.windows(3))
                .zip(out)
            {
                let me = c[1];
                let side = u[1] + c[0] + c[2] + d[1];
                let diag = u[0] + u[2] + d[0] + d[2];
                *out = me * SCALE_SELF + side * SCALE_SIDE + diag * SCALE_DIAG;
            }
        }
    }

    let mut last_coords_y = [0; 3];
    let mut current_row_x = vec![0.0; dims[0].0];
    let mut current_row_y = vec![0.0; dims[1].0];
    let mut current_row_b = vec![0.0; dims[2].0];
    for row in 0..height {
        let coords_y = shifts.map(|shift| row >> shift.vshift());
        if std::iter::zip(coords_y, dims).any(|(row, (_, height))| row == 0 || row == height - 1) {
            continue;
        }

        let row_x = &mut in_x[coords_y[0] * dims[0].0..][..dims[0].0];
        let row_y = &mut in_y[coords_y[1] * dims[1].0..][..dims[1].0];
        let row_b = &mut in_b[coords_y[2] * dims[2].0..][..dims[2].0];

        if last_coords_y[0] != coords_y[0] {
            current_row_x.copy_from_slice(row_x);
        }
        if last_coords_y[1] != coords_y[1] {
            current_row_y.copy_from_slice(row_y);
        }
        if last_coords_y[2] != coords_y[2] {
            current_row_b.copy_from_slice(row_b);
        }
        last_coords_y = coords_y;

        for col in 0..width {
            let coords_x = shifts.map(|shift| col >> shift.hshift());
            if std::iter::zip(coords_x, dims).any(|(col, (width, _))| col == 0 || col == width - 1)
            {
                continue;
            }

            let skip_update: [_; 3] = std::array::from_fn(|idx| {
                let shift = shifts[idx];
                (coords_y[idx] << shift.vshift()) != row || (coords_x[idx] << shift.hshift()) != col
            });

            let x_self = current_row_x[coords_x[0]];
            let x_wa = wa_x.get(coords_x[0] - 1, coords_y[0] - 1);
            let x_gap_t = (x_wa - x_self).abs() / lf_x;

            let y_self = current_row_y[coords_x[1]];
            let y_wa = wa_y.get(coords_x[1] - 1, coords_y[1] - 1);
            let y_gap_t = (y_wa - y_self).abs() / lf_y;

            let b_self = current_row_b[coords_x[2]];
            let b_wa = wa_b.get(coords_x[2] - 1, coords_y[2] - 1);
            let b_gap_t = (b_wa - b_self).abs() / lf_b;

            let gap = 0.5f32.max(x_gap_t).max(y_gap_t).max(b_gap_t);
            let gap_scale = (3.0 - 4.0 * gap).max(0.0);

            if !skip_update[0] {
                row_x[coords_x[0]] = (x_wa - x_self) * gap_scale + x_self;
            }
            if !skip_update[1] {
                row_y[coords_x[1]] = (y_wa - y_self) * gap_scale + y_self;
            }
            if !skip_update[2] {
                row_b[coords_x[2]] = (b_wa - b_self) * gap_scale + b_self;
            }
        }
    }

    Ok(())
}
