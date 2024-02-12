use std::arch::aarch64::*;

use crate::filter::gabor::GaborRow;

#[target_feature(enable = "neon")]
pub(super) unsafe fn run_gabor_row_aarch64_neon(row: GaborRow) {
    let GaborRow {
        input_rows,
        input_start: start_x,
        input_stride: stride,
        output_row,
        weights,
    } = row;
    let width = output_row.len();
    assert_eq!(input_rows.len(), 3 * stride);
    assert!(stride >= start_x + width);

    if width == 0 {
        return;
    }

    let [w0, w1] = weights;
    let global_weight = (1.0 + w0 * 4.0 + w1 * 4.0).recip();

    let input_ptr_t = input_rows.as_ptr().add(start_x);
    let input_ptr_c = input_rows.as_ptr().add(start_x + stride);
    let input_ptr_b = input_rows.as_ptr().add(start_x + stride * 2);
    let output_ptr = output_row.as_mut_ptr();

    let mut tl = vld1_dup_f32(input_ptr_t);
    let mut cl = vld1_dup_f32(input_ptr_c);
    let mut bl = vld1_dup_f32(input_ptr_b);
    for dx2 in 0..(width - 1) / 2 {
        let x = dx2 * 2;

        let tr = vld1_f32(input_ptr_t.add(1 + x));
        let cr = vld1_f32(input_ptr_c.add(1 + x));
        let br = vld1_f32(input_ptr_b.add(1 + x));

        let t = vext_f32::<1>(tl, tr);
        let c = vext_f32::<1>(cl, cr);
        let b = vext_f32::<1>(bl, br);

        let sum_side = vadd_f32(vadd_f32(vadd_f32(t, cl), cr), b);
        let sum_diag = vadd_f32(vadd_f32(vadd_f32(tl, tr), bl), br);
        let unweighted_sum = vfma_n_f32(vfma_n_f32(c, sum_side, w0), sum_diag, w1);
        let sum = vmul_n_f32(unweighted_sum, global_weight);

        vst1_f32(output_ptr.add(x), sum);
        tl = tr;
        cl = cr;
        bl = br;
    }

    if width % 2 == 0 {
        let x = width - 2;

        let tr = vld1_dup_f32(input_ptr_t.add(1 + x));
        let cr = vld1_dup_f32(input_ptr_c.add(1 + x));
        let br = vld1_dup_f32(input_ptr_b.add(1 + x));

        let t = vext_f32::<1>(tl, tr);
        let c = vext_f32::<1>(cl, cr);
        let b = vext_f32::<1>(bl, br);

        let sum_side = vadd_f32(vadd_f32(vadd_f32(t, cl), cr), b);
        let sum_diag = vadd_f32(vadd_f32(vadd_f32(tl, tr), bl), br);
        let unweighted_sum = vfma_n_f32(vfma_n_f32(c, sum_side, w0), sum_diag, w1);
        let sum = vmul_n_f32(unweighted_sum, global_weight);

        vst1_f32(output_ptr.add(x), sum);
    } else {
        let x = width - 1;
        // t0 t1 t1
        // c0 c1 c1
        // b0 b1 b1
        let t0 = vget_lane_f32::<0>(tl);
        let t1 = vget_lane_f32::<1>(tl);
        let c0 = vget_lane_f32::<0>(cl);
        let c1 = vget_lane_f32::<1>(cl);
        let b0 = vget_lane_f32::<0>(bl);
        let b1 = vget_lane_f32::<1>(bl);
        let sum_side = t1 + c0 + c1 + b1;
        let sum_diag = t0 + t1 + b0 + b1;
        let unweighted_sum = sum_diag.mul_add(w1, sum_side.mul_add(w0, c1));
        *output_ptr.add(x) = unweighted_sum * global_weight;
    }
}
