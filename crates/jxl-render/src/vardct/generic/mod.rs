use jxl_grid::{AllocTracker, SimpleGrid};

mod dct;
mod transform;
pub use dct::dct_2d;
pub use transform::transform;

#[inline(always)]
pub fn adaptive_lf_smoothing_impl(
    width: usize,
    height: usize,
    [in_x, in_y, in_b]: [&mut [f32]; 3],
    [lf_x, lf_y, lf_b]: [f32; 3],
    tracker: Option<&AllocTracker>,
) -> crate::Result<()> {
    const SCALE_SELF: f32 = 0.052262735;
    const SCALE_SIDE: f32 = 0.2034514;
    const SCALE_DIAG: f32 = 0.03348292;

    if width <= 2 || height <= 2 {
        // Nothing to do
        return Ok(());
    }

    assert_eq!(in_x.len(), in_y.len());
    assert_eq!(in_y.len(), in_b.len());
    assert_eq!(in_x.len(), width * height);

    let mut udsum_x = SimpleGrid::with_alloc_tracker(width, height - 2, tracker)?;
    let mut udsum_y = SimpleGrid::with_alloc_tracker(width, height - 2, tracker)?;
    let mut udsum_b = SimpleGrid::with_alloc_tracker(width, height - 2, tracker)?;

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

    let mut udsum_x_row = udsum_x.buf_mut().chunks_exact(width);
    let mut udsum_y_row = udsum_y.buf_mut().chunks_exact(width);
    let mut udsum_b_row = udsum_b.buf_mut().chunks_exact(width);

    loop {
        let Some(udsum_x) = udsum_x_row.next() else {
            break;
        };
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
