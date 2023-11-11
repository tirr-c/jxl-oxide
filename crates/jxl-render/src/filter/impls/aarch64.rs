use std::arch::is_aarch64_feature_detected;
use std::arch::aarch64::*;

use jxl_grid::SimpleGrid;

use super::generic::epf_common;

mod epf;

pub fn epf_step0(
    input: &[SimpleGrid<f32>; 3],
    output: &mut [SimpleGrid<f32>; 3],
    sigma_grid: &SimpleGrid<f32>,
    channel_scale: [f32; 3],
    border_sad_mul: f32,
    step_multiplier: f32,
) {
    if is_aarch64_feature_detected!("neon") {
        // SAFETY: Features are checked above.
        unsafe {
            return epf_common(
                input,
                output,
                sigma_grid,
                channel_scale,
                border_sad_mul,
                step_multiplier,
                epf::epf_row_step0_neon,
            );
        }
    }

    // SAFETY: row handler is safe.
    unsafe {
        epf_common(
            input,
            output,
            sigma_grid,
            channel_scale,
            border_sad_mul,
            step_multiplier,
            super::generic::epf_row_step0,
        );
    }
}

pub fn epf_step1(
    input: &[SimpleGrid<f32>; 3],
    output: &mut [SimpleGrid<f32>; 3],
    sigma_grid: &SimpleGrid<f32>,
    channel_scale: [f32; 3],
    border_sad_mul: f32,
    step_multiplier: f32,
) {
    if is_aarch64_feature_detected!("neon") {
        // SAFETY: Features are checked above.
        unsafe {
            return epf_common(
                input,
                output,
                sigma_grid,
                channel_scale,
                border_sad_mul,
                step_multiplier,
                epf::epf_row_step1_neon,
            );
        }
    }

    // SAFETY: row handler is safe.
    unsafe {
        epf_common(
            input,
            output,
            sigma_grid,
            channel_scale,
            border_sad_mul,
            step_multiplier,
            super::generic::epf_row_step1,
        );
    }
}

pub fn epf_step2(
    input: &[SimpleGrid<f32>; 3],
    output: &mut [SimpleGrid<f32>; 3],
    sigma_grid: &SimpleGrid<f32>,
    channel_scale: [f32; 3],
    border_sad_mul: f32,
    step_multiplier: f32,
) {
    if is_aarch64_feature_detected!("neon") {
        // SAFETY: Features are checked above.
        unsafe {
            return epf_common(
                input,
                output,
                sigma_grid,
                channel_scale,
                border_sad_mul,
                step_multiplier,
                epf::epf_row_step2_neon,
            );
        }
    }

    // SAFETY: row handler is safe.
    unsafe {
        epf_common(
            input,
            output,
            sigma_grid,
            channel_scale,
            border_sad_mul,
            step_multiplier,
            super::generic::epf_row_step2,
        );
    }
}

pub fn apply_gabor_like(fb: [&mut SimpleGrid<f32>; 3], weights_xyb: [[f32; 2]; 3]) {
    if is_aarch64_feature_detected!("neon") {
        // SAFETY: Features are checked above.
        unsafe {
            for (fb, [weight1, weight2]) in fb.into_iter().zip(weights_xyb) {
                run_gabor_inner_neon(fb, weight1, weight2)
            }
        }
        return;
    }

    for (fb, [weight1, weight2]) in fb.into_iter().zip(weights_xyb) {
        super::generic::run_gabor_inner(fb, weight1, weight2);
    }
}

#[target_feature(enable = "neon")]
unsafe fn run_gabor_inner_neon(fb: &mut SimpleGrid<f32>, weight1: f32, weight2: f32) {
    let global_weight = (1.0 + weight1 * 4.0 + weight2 * 4.0).recip();

    let width = fb.width();
    let height = fb.height();
    if width * height <= 1 {
        return;
    }

    let io = fb.buf_mut();
    let mut prev_row = io[..width].to_vec();

    let input_t = prev_row.as_mut_ptr();
    let mut input_c = io.as_mut_ptr();
    let mut input_b = input_c.add(width);

    for _ in 0..height - 1 {
        let mut tl = vld1_dup_f32(input_t);
        let mut cl = vld1_dup_f32(input_c);
        let mut bl = vld1_dup_f32(input_b);
        *input_t = *input_c;

        for vx in 0..(width - 1) / 2 {
            let tr = vld1_f32(input_t.add(1 + vx * 2));
            let cr = vld1_f32(input_c.add(1 + vx * 2));
            let br = vld1_f32(input_b.add(1 + vx * 2));
            let t = vext_f32::<1>(tl, tr);
            let c = vext_f32::<1>(cl, cr);
            let b = vext_f32::<1>(bl, br);

            let sum_side = vadd_f32(vadd_f32(vadd_f32(t, cl), cr), b);
            let sum_diag = vadd_f32(vadd_f32(vadd_f32(tl, tr), bl), br);
            let unweighted_sum = vfma_n_f32(
                vfma_n_f32(
                    c,
                    sum_side,
                    weight1,
                ),
                sum_diag,
                weight2,
            );
            let sum = vmul_n_f32(unweighted_sum, global_weight);

            vst1_f32(input_t.add(1 + vx * 2), cr);
            vst1_f32(input_c.add(vx * 2), sum);
            tl = tr;
            cl = cr;
            bl = br;
        }

        let tr;
        let cr;
        let br;
        if width % 2 == 0 {
            tr = vld1_dup_f32(input_t.add(width - 1));
            cr = vld1_dup_f32(input_c.add(width - 1));
            br = vld1_dup_f32(input_b.add(width - 1));
        } else {
            tr = vdup_lane_f32::<1>(tl);
            cr = vdup_lane_f32::<1>(cl);
            br = vdup_lane_f32::<1>(bl);
        };
        let t = vext_f32::<1>(tl, tr);
        let c = vext_f32::<1>(cl, cr);
        let b = vext_f32::<1>(bl, br);

        let sum_side = vadd_f32(vadd_f32(vadd_f32(t, cl), cr), b);
        let sum_diag = vadd_f32(vadd_f32(vadd_f32(tl, tr), bl), br);
        let unweighted_sum = vfma_n_f32(
            vfma_n_f32(
                c,
                sum_side,
                weight1,
            ),
            sum_diag,
            weight2,
        );
        let sum = vmul_n_f32(unweighted_sum, global_weight);

        if width % 2 == 0 {
            *input_t.add(width - 1) = vget_lane_f32::<0>(cr);
            vst1_f32(input_c.add(width - 2), sum);
        } else {
            *input_c.add(width - 1) = vget_lane_f32::<0>(sum);
        }

        input_c = input_c.add(width);
        input_b = input_b.add(width);
    }

    let mut tl = vld1_dup_f32(input_t);
    let mut cl = vld1_dup_f32(input_c);

    for vx in 0..(width - 1) / 2 {
        let tr = vld1_f32(input_t.add(1 + vx * 2));
        let cr = vld1_f32(input_c.add(1 + vx * 2));
        let t = vext_f32::<1>(tl, tr);
        let c = vext_f32::<1>(cl, cr);

        let sum_side = vadd_f32(vadd_f32(vadd_f32(t, cl), cr), c);
        let sum_diag = vadd_f32(vadd_f32(vadd_f32(tl, tr), cl), cr);
        let unweighted_sum = vfma_n_f32(
            vfma_n_f32(
                c,
                sum_side,
                weight1,
            ),
            sum_diag,
            weight2,
        );
        let sum = vmul_n_f32(unweighted_sum, global_weight);

        vst1_f32(input_c.add(vx * 2), sum);
        tl = tr;
        cl = cr;
    }

    let tr;
    let cr;
    if width % 2 == 0 {
        tr = vld1_dup_f32(input_t.add(width - 1));
        cr = vld1_dup_f32(input_c.add(width - 1));
    } else {
        tr = vdup_lane_f32::<1>(tl);
        cr = vdup_lane_f32::<1>(cl);
    };
    let t = vext_f32::<1>(tl, tr);
    let c = vext_f32::<1>(cl, cr);

    let sum_side = vadd_f32(vadd_f32(vadd_f32(t, cl), cr), c);
    let sum_diag = vadd_f32(vadd_f32(vadd_f32(tl, tr), cl), cr);
    let unweighted_sum = vfma_n_f32(
        vfma_n_f32(
            c,
            sum_side,
            weight1,
        ),
        sum_diag,
        weight2,
    );
    let sum = vmul_n_f32(unweighted_sum, global_weight);

    if width % 2 == 0 {
        vst1_f32(input_c.add(width - 2), sum);
    } else {
        *input_c.add(width - 1) = vget_lane_f32::<0>(sum);
    }
}
