use crate::filter::gabor::GaborRow;

pub(crate) fn gabor_row_edge(
    row_c: &[f32],
    row_a: Option<&[f32]>,
    out: &mut [f32],
    weights: [f32; 2],
) {
    let width = out.len();
    assert!(width > 0);
    assert_eq!(row_c.len(), width);

    let [w0, w1] = weights;
    let global_weight = (1.0 + w0 * 4.0 + w1 * 4.0).recip();

    if let Some(row_a) = row_a {
        // Bottom/top edge case
        if width == 1 {
            let u = row_a[0];
            let c = row_c[0];
            out[0] = (c * (1.0 + 3.0 * w0 + 2.0 * w1) + u * (w0 + 2.0 * w1)) * global_weight;
            return;
        }

        {
            // 1 | 1 0
            let a1 = row_a[0];
            let a0 = row_a[1];
            let c1 = row_c[0];
            let c0 = row_c[1];
            out[0] = (c1 * (1.0 + 2.0 * w0 + w1) + (a1 + c0) * (w0 + w1) + a0 * w1) * global_weight;
        }

        // 0 1 2
        let it = row_c
            .windows(3)
            .zip(row_a.windows(3))
            .zip(&mut out[1..width - 1]);
        for ((window_c, window_a), out) in it {
            let [a0, a1, a2] = window_a else {
                unreachable!()
            };
            let [c0, c1, c2] = window_c else {
                unreachable!()
            };
            *out = (c1 + (a1 + c0 + c1 + c2) * w0 + (a0 + a2 + c0 + c2) * w1) * global_weight;
        }

        {
            // 0 1 | 1
            let a0 = row_a[width - 2];
            let a1 = row_a[width - 1];
            let c0 = row_c[width - 2];
            let c1 = row_c[width - 1];
            out[width - 1] =
                (c1 * (1.0 + 2.0 * w0 + w1) + (a1 + c0) * (w0 + w1) + a0 * w1) * global_weight;
        }
    } else {
        // Single row case
        if width == 1 {
            out[0] = row_c[0];
            return;
        }

        let merged_w0 = 1.0 + 2.0 + w0;
        let merged_w1 = w0 + 2.0 * w1;

        {
            let c1 = row_c[0];
            let c0 = row_c[1];
            out[0] = (c1 * (merged_w0 + merged_w1) + c0 * merged_w1) * global_weight;
        }

        let it = row_c.windows(3).zip(&mut out[1..width - 1]);
        for (window, out) in it {
            let [c0, c1, c2] = window else { unreachable!() };
            *out = (c1 * merged_w0 + (c0 + c2) * merged_w1) * global_weight;
        }

        {
            let c0 = row_c[width - 2];
            let c1 = row_c[width - 1];
            out[width - 1] = (c1 * (merged_w0 + merged_w1) + c0 * merged_w1) * global_weight;
        }
    }
}

#[inline(always)]
pub(crate) fn run_gabor_row_generic(row: GaborRow) {
    let GaborRow {
        input_rows: [input_row_t, input_row_c, input_row_b],
        output_row,
        weights,
    } = row;
    let width = output_row.len();
    assert_eq!(input_row_t.len(), width);
    assert_eq!(input_row_c.len(), width);
    assert_eq!(input_row_b.len(), width);

    if width == 0 {
        return;
    }

    let [w0, w1] = weights;
    let global_weight = (1.0 + w0 * 4.0 + w1 * 4.0).recip();

    if width == 1 {
        let t = input_row_t[0];
        let c = input_row_c[0];
        let b = input_row_b[0];

        let sum_side = t + 2.0 * c + b;
        let sum_diag = 2.0 * (t + b);
        let unweighted_sum = c + sum_side * w0 + sum_diag * w1;
        output_row[0] = unweighted_sum * global_weight;
        return;
    }

    {
        // t1 t1 t0
        // c1 c1 c0
        // b1 b1 b0
        let t1 = input_row_t[0];
        let c1 = input_row_c[0];
        let b1 = input_row_b[0];
        let t0 = input_row_t[1];
        let c0 = input_row_c[1];
        let b0 = input_row_b[1];

        let sum_side = t1 + c0 + c1 + b1;
        let sum_diag = t0 + t1 + b0 + b1;
        let unweighted_sum = c1 + sum_side * w0 + sum_diag * w1;
        output_row[0] = unweighted_sum * global_weight;
    }

    let it = input_row_t
        .windows(3)
        .zip(input_row_c.windows(3))
        .zip(input_row_b.windows(3))
        .zip(&mut output_row[1..width - 1]);
    for (((t, c), b), out) in it {
        let [t0, t1, t2] = t else { unreachable!() };
        let [c0, c1, c2] = c else { unreachable!() };
        let [b0, b1, b2] = b else { unreachable!() };

        let sum_side = t1 + c0 + c2 + b1;
        let sum_diag = t0 + t2 + b0 + b2;
        let unweighted_sum = c1 + sum_side * w0 + sum_diag * w1;
        *out = unweighted_sum * global_weight;
    }

    {
        // t0 t1 t1
        // c0 c1 c1
        // b0 b1 b1
        let t1 = input_row_t[width - 1];
        let c1 = input_row_c[width - 1];
        let b1 = input_row_b[width - 1];
        let t0 = input_row_t[width - 2];
        let c0 = input_row_c[width - 2];
        let b0 = input_row_b[width - 2];

        let sum_side = t1 + c0 + c1 + b1;
        let sum_diag = t0 + t1 + b0 + b1;
        let unweighted_sum = c1 + sum_side * w0 + sum_diag * w1;
        output_row[width - 1] = unweighted_sum * global_weight;
    }
}
