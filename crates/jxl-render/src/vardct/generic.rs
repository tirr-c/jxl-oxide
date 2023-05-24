#[inline(always)]
pub fn adaptive_lf_smoothing_impl(
    width: usize,
    height: usize,
    [in_x, in_y, in_b]: [&[f32]; 3],
    [out_x, out_y, out_b]: [&mut [f32]; 3],
    [lf_x, lf_y, lf_b]: [f32; 3],
) {
    const SCALE_SELF: f32 = 0.052262735;
    const SCALE_SIDE: f32 = 0.2034514;
    const SCALE_DIAG: f32 = 0.03348292;

    if width <= 2 || height <= 2 {
        // Nothing to do
        return;
    }

    assert_eq!(in_x.len(), in_y.len());
    assert_eq!(in_y.len(), in_b.len());
    assert_eq!(in_x.len(), out_x.len());
    assert_eq!(in_y.len(), out_y.len());
    assert_eq!(in_b.len(), out_b.len());
    assert_eq!(in_x.len(), width * height);

    let mut udsum_x = vec![0.0f32; width * (height - 2)];
    let mut udsum_y = vec![0.0f32; width * (height - 2)];
    let mut udsum_b = vec![0.0f32; width * (height - 2)];

    for (g, out) in [(in_x, &mut udsum_x), (in_y, &mut udsum_y), (in_b, &mut udsum_b)] {
        let up = g.chunks_exact(width);
        let down = g[width * 2..].chunks_exact(width);
        let out = out.chunks_exact_mut(width);
        for ((up, down), out) in up.zip(down).zip(out) {
            for ((&u, &d), out) in up.iter().zip(down).zip(out) {
                *out = u + d;
            }
        }
    }

    let mut in_x_row = in_x.chunks_exact(width);
    let mut in_y_row = in_y.chunks_exact(width);
    let mut in_b_row = in_b.chunks_exact(width);
    let mut out_x_row = out_x.chunks_exact_mut(width);
    let mut out_y_row = out_y.chunks_exact_mut(width);
    let mut out_b_row = out_b.chunks_exact_mut(width);

    out_x_row.next().unwrap().copy_from_slice(in_x_row.next().unwrap());
    out_y_row.next().unwrap().copy_from_slice(in_y_row.next().unwrap());
    out_b_row.next().unwrap().copy_from_slice(in_b_row.next().unwrap());

    let mut udsum_x_row = udsum_x.chunks_exact(width);
    let mut udsum_y_row = udsum_y.chunks_exact(width);
    let mut udsum_b_row = udsum_b.chunks_exact(width);

    loop {
        let Some(udsum_x) = udsum_x_row.next() else { break; };
        let udsum_y = udsum_y_row.next().unwrap();
        let udsum_b = udsum_b_row.next().unwrap();
        let in_x = in_x_row.next().unwrap();
        let in_y = in_y_row.next().unwrap();
        let in_b = in_b_row.next().unwrap();
        let out_x = out_x_row.next().unwrap();
        let out_y = out_y_row.next().unwrap();
        let out_b = out_b_row.next().unwrap();

        out_x[0] = in_x[0];
        out_y[0] = in_y[0];
        out_b[0] = in_b[0];

        for x in 1..(width - 1) {
            let x_self = in_x[x];
            let x_side = in_x[x - 1] + in_x[x + 1] + udsum_x[x];
            let x_diag = udsum_x[x - 1] + udsum_x[x + 1];
            let x_wa = x_self * SCALE_SELF + x_side * SCALE_SIDE + x_diag * SCALE_DIAG;
            let x_gap_t = (x_wa - x_self).abs() / lf_x;

            let y_self = in_y[x];
            let y_side = in_y[x - 1] + in_y[x + 1] + udsum_y[x];
            let y_diag = udsum_y[x - 1] + udsum_y[x + 1];
            let y_wa = y_self * SCALE_SELF + y_side * SCALE_SIDE + y_diag * SCALE_DIAG;
            let y_gap_t = (y_wa - y_self).abs() / lf_y;

            let b_self = in_b[x];
            let b_side = in_b[x - 1] + in_b[x + 1] + udsum_b[x];
            let b_diag = udsum_b[x - 1] + udsum_b[x + 1];
            let b_wa = b_self * SCALE_SELF + b_side * SCALE_SIDE + b_diag * SCALE_DIAG;
            let b_gap_t = (b_wa - b_self).abs() / lf_b;

            let gap = 0.5f32.max(x_gap_t).max(y_gap_t).max(b_gap_t);
            let gap_scale = (3.0 - 4.0 * gap).max(0.0);

            out_x[x] = (x_wa - x_self) * gap_scale + x_self;
            out_y[x] = (y_wa - y_self) * gap_scale + y_self;
            out_b[x] = (b_wa - b_self) * gap_scale + b_self;
        }

        out_x[width - 1] = in_x[width - 1];
        out_y[width - 1] = in_y[width - 1];
        out_b[width - 1] = in_b[width - 1];
    }

    out_x_row.next().unwrap().copy_from_slice(in_x_row.next().unwrap());
    out_y_row.next().unwrap().copy_from_slice(in_y_row.next().unwrap());
    out_b_row.next().unwrap().copy_from_slice(in_b_row.next().unwrap());
}
