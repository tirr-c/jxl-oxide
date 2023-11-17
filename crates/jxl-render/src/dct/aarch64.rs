use jxl_grid::{CutGrid, SimdVector};

use super::{consts, DctDirection};
use std::arch::aarch64::*;

const LANE_SIZE: usize = 4;
type Lane = float32x4_t;

fn transpose_lane(lanes: &[Lane]) -> float32x4x4_t {
    assert_eq!(lanes.len(), 4);
    unsafe {
        let ptr = lanes.as_ptr() as *mut f32;
        vld4q_f32(ptr as *const _)
    }
}

pub fn dct_2d(io: &mut CutGrid<'_>, direction: DctDirection) {
    if !Lane::available() || io.width() % LANE_SIZE != 0 || io.height() % LANE_SIZE != 0 {
        return super::generic::dct_2d(io, direction);
    }

    let Some(mut io) = io.as_vectored() else {
        tracing::trace!("Input buffer is not aligned");
        return super::generic::dct_2d(io, direction);
    };

    if io.width() == 2 && io.height() == 8 {
        unsafe {
            return dct8x8(&mut io, direction);
        }
    }

    dct_2d_lane(&mut io, direction);
}

fn dct_2d_lane(io: &mut CutGrid<'_, Lane>, direction: DctDirection) {
    let scratch_size = io.height().max(io.width() * LANE_SIZE) * 2;
    unsafe {
        let mut scratch_lanes = vec![Lane::zero(); scratch_size];
        column_dct_lane(io, &mut scratch_lanes, direction);
        row_dct_lane(io, &mut scratch_lanes, direction);
    }
}

unsafe fn dct4_vec_forward(v: Lane) -> Lane {
    const SEC0: f32 = 0.5411961;
    const SEC1: f32 = 1.306563;

    let v01 = vget_low_f32(v);
    let v23 = vget_high_f32(v);
    let addsub = vcombine_f32(
        vadd_f32(v01, vrev64_f32(v23)),
        vfma_n_f32(vrev64_f32(v01), v23, -1f32),
    );

    let addsub3012 = vextq_f32(addsub, addsub, 3);
    let addsub03 = vrev64_f32(vget_low_f32(addsub3012));
    let addsub12 = vget_high_f32(addsub3012);

    let a = vcombine_f32(addsub03, addsub12);
    let mul_a = Lane::set([
        0.25,
        (std::f32::consts::FRAC_1_SQRT_2 / 2.0 + 0.25) * SEC0,
        -0.25,
        -0.25 * SEC1,
    ]);
    let b = vcombine_f32(addsub12, addsub03);
    let mul_b = Lane::set([
        0.25,
        (std::f32::consts::FRAC_1_SQRT_2 / 2.0 - 0.25) * SEC1,
        0.25,
        0.25 * SEC0,
    ]);
    a.muladd(mul_a, b.mul(mul_b))
}

unsafe fn dct4_vec_inverse(v: Lane) -> Lane {
    const SEC0: f32 = 0.5411961;
    const SEC1: f32 = 1.306563;

    let v_flip = vextq_f32(v, v, 2);
    let mul_a = Lane::set([1.0, (std::f32::consts::SQRT_2 + 1.0) * SEC0, -1.0, -SEC1]);
    let mul_b = Lane::set([1.0, SEC0, 1.0, (std::f32::consts::SQRT_2 - 1.0) * SEC1]);
    let tmp = v.muladd(mul_a, v_flip.mul(mul_b));

    let float32x4x2_t(tmp_a, tmp_b) = vuzpq_f32(tmp, vextq_f32(tmp, tmp, 2));
    let mul = vcombine_f32(vdup_n_f32(1.0), vdup_n_f32(-1.0));
    tmp_b.muladd(mul, tmp_a)
}

unsafe fn dct8_vec_forward(vl: Lane, vr: Lane) -> (Lane, Lane) {
    #[allow(clippy::excessive_precision)]
    let sec_vec = Lane::set([
        0.2548977895520796,
        0.30067244346752264,
        0.4499881115682078,
        1.2814577238707527,
    ]);
    let vr_rev = vrev64q_f32(vextq_f32(vr, vr, 2));
    let input0 = vmulq_n_f32(vl.add(vr_rev), 0.5);
    let input1 = vl.sub(vr_rev).mul(sec_vec);
    let output0 = dct4_vec_forward(input0);
    let output1 = dct4_vec_forward(input1);
    let output1_shifted = vextq_f32(output1, Lane::zero(), 1);
    let output1_mul = vsetq_lane_f32(std::f32::consts::SQRT_2, Lane::splat_f32(1.0), 0);
    let output1 = output1.muladd(output1_mul, output1_shifted);
    (
        vcombine_f32(vget_low_f32(output0), vget_low_f32(output1)),
        vcombine_f32(vget_high_f32(output0), vget_high_f32(output1)),
    )
}

unsafe fn dct8_vec_inverse(vl: Lane, vr: Lane) -> (Lane, Lane) {
    #[allow(clippy::excessive_precision)]
    let sec_vec = Lane::set([
        0.5097955791041592,
        0.6013448869350453,
        0.8999762231364156,
        2.5629154477415055,
    ]);
    let float32x4x2_t(input0, input1) = vuzpq_f32(vl, vr);
    let input1_shifted = vextq_f32(Lane::zero(), input1, 3);
    let input1_mul = vsetq_lane_f32(std::f32::consts::SQRT_2, Lane::splat_f32(1.0), 0);
    let input1 = input1.muladd(input1_mul, input1_shifted);
    let output0 = dct4_vec_inverse(input0);
    let output1 = dct4_vec_inverse(input1);
    let output1 = output1.mul(sec_vec);
    let sub = output0.sub(output1);
    (output0.add(output1), vrev64q_f32(vextq_f32(sub, sub, 2)))
}

unsafe fn dct8x8(io: &mut CutGrid<'_, Lane>, direction: DctDirection) {
    let (mut col0, mut col1) = io.split_horizontal(1);

    if direction == DctDirection::Forward {
        dct8_forward(&mut col0);
        dct8_forward(&mut col1);
        for y in 0..8 {
            let row = io.get_row_mut(y);
            let (vl, vr) = dct8_vec_forward(row[0], row[1]);
            row[0] = vl;
            row[1] = vr;
        }
    } else {
        dct8_inverse(&mut col0);
        dct8_inverse(&mut col1);
        for y in 0..8 {
            let row = io.get_row_mut(y);
            let (vl, vr) = dct8_vec_inverse(row[0], row[1]);
            row[0] = vl;
            row[1] = vr;
        }
    }
}

unsafe fn column_dct_lane(
    io: &mut CutGrid<'_, Lane>,
    scratch: &mut [Lane],
    direction: DctDirection,
) {
    let width = io.width();
    let height = io.height();
    let (io_lanes, scratch_lanes) = scratch[..height * 2].split_at_mut(height);
    for x in 0..width {
        for (y, input) in io_lanes.iter_mut().enumerate() {
            *input = io.get(x, y);
        }
        dct(io_lanes, scratch_lanes, direction);
        for (y, output) in io_lanes.chunks_exact(LANE_SIZE).enumerate() {
            let float32x4x4_t(o0, o1, o2, o3) = transpose_lane(output);
            *io.get_mut(x, y * LANE_SIZE) = o0;
            *io.get_mut(x, y * LANE_SIZE + 1) = o1;
            *io.get_mut(x, y * LANE_SIZE + 2) = o2;
            *io.get_mut(x, y * LANE_SIZE + 3) = o3;
        }
    }
}

unsafe fn row_dct_lane(io: &mut CutGrid<'_, Lane>, scratch: &mut [Lane], direction: DctDirection) {
    let width = io.width() * LANE_SIZE;
    let height = io.height();
    let (io_lanes, scratch_lanes) = scratch[..width * 2].split_at_mut(width);
    for y in (0..height).step_by(LANE_SIZE) {
        for (x, input) in io_lanes.chunks_exact_mut(LANE_SIZE).enumerate() {
            for (dy, input) in input.iter_mut().enumerate() {
                *input = io.get(x, y + dy);
            }
        }
        dct(io_lanes, scratch_lanes, direction);
        for (x, output) in io_lanes.chunks_exact(LANE_SIZE).enumerate() {
            let float32x4x4_t(o0, o1, o2, o3) = transpose_lane(output);
            *io.get_mut(x, y) = o0;
            *io.get_mut(x, y + 1) = o1;
            *io.get_mut(x, y + 2) = o2;
            *io.get_mut(x, y + 3) = o3;
        }
    }
}

unsafe fn dct4_forward(input: [Lane; 4]) -> [Lane; 4] {
    let sec0 = 0.5411961 / 4.0;
    let sec1 = 1.306563 / 4.0;

    let sum03 = input[0].add(input[3]);
    let sum12 = input[1].add(input[2]);
    let tmp0 = vmulq_n_f32(input[0].sub(input[3]), sec0);
    let tmp1 = vmulq_n_f32(input[1].sub(input[2]), sec1);
    let out0 = tmp0.add(tmp1);
    let out1 = tmp0.sub(tmp1);

    [
        vmulq_n_f32(sum03.add(sum12), 0.25),
        vfmaq_n_f32(out1, out0, std::f32::consts::SQRT_2),
        vmulq_n_f32(sum03.sub(sum12), 0.25),
        out1,
    ]
}

unsafe fn dct4_inverse(input: [Lane; 4]) -> [Lane; 4] {
    let sec0 = 0.5411961;
    let sec1 = 1.306563;

    let tmp0 = vmulq_n_f32(input[1], std::f32::consts::SQRT_2);
    let tmp1 = input[1].add(input[3]);
    let out0 = vmulq_n_f32(tmp0.add(tmp1), sec0);
    let out1 = vmulq_n_f32(tmp0.sub(tmp1), sec1);
    let sum02 = input[0].add(input[2]);
    let sub02 = input[0].sub(input[2]);

    [
        sum02.add(out0),
        sub02.add(out1),
        sub02.sub(out1),
        sum02.sub(out0),
    ]
}

unsafe fn dct8_forward(io: &mut CutGrid<'_, Lane>) {
    assert!(io.height() == 8);
    let sec = consts::sec_half_small(8);

    let input0 = [
        vmulq_n_f32(io.get(0, 0).add(io.get(0, 7)), 0.5),
        vmulq_n_f32(io.get(0, 1).add(io.get(0, 6)), 0.5),
        vmulq_n_f32(io.get(0, 2).add(io.get(0, 5)), 0.5),
        vmulq_n_f32(io.get(0, 3).add(io.get(0, 4)), 0.5),
    ];
    let input1 = [
        vmulq_n_f32(io.get(0, 0).sub(io.get(0, 7)), sec[0] / 2.0),
        vmulq_n_f32(io.get(0, 1).sub(io.get(0, 6)), sec[1] / 2.0),
        vmulq_n_f32(io.get(0, 2).sub(io.get(0, 5)), sec[2] / 2.0),
        vmulq_n_f32(io.get(0, 3).sub(io.get(0, 4)), sec[3] / 2.0),
    ];
    let output0 = dct4_forward(input0);
    for (idx, v) in output0.into_iter().enumerate() {
        *io.get_mut(0, idx * 2) = v;
    }
    let mut output1 = dct4_forward(input1);
    output1[0] = vmulq_n_f32(output1[0], std::f32::consts::SQRT_2);
    for idx in 0..3 {
        *io.get_mut(0, idx * 2 + 1) = output1[idx].add(output1[idx + 1]);
    }
    *io.get_mut(0, 7) = output1[3];
}

unsafe fn dct8_inverse(io: &mut CutGrid<'_, Lane>) {
    assert!(io.height() == 8);
    let sec = consts::sec_half_small(8);

    let input0 = [io.get(0, 0), io.get(0, 2), io.get(0, 4), io.get(0, 6)];
    let input1 = [
        vmulq_n_f32(io.get(0, 1), std::f32::consts::SQRT_2),
        io.get(0, 3).add(io.get(0, 1)),
        io.get(0, 5).add(io.get(0, 3)),
        io.get(0, 7).add(io.get(0, 5)),
    ];
    let output0 = dct4_inverse(input0);
    let output1 = dct4_inverse(input1);
    for (idx, &sec) in sec.iter().enumerate() {
        let r = vmulq_n_f32(output1[idx], sec);
        *io.get_mut(0, idx) = output0[idx].add(r);
        *io.get_mut(0, 7 - idx) = output0[idx].sub(r);
    }
}

unsafe fn dct(io: &mut [Lane], scratch: &mut [Lane], direction: DctDirection) {
    let n = io.len();
    assert!(scratch.len() == n);

    if n == 0 {
        return;
    }
    if n == 1 {
        return;
    }

    if n == 2 {
        let tmp0 = io[0].add(io[1]);
        let tmp1 = io[0].sub(io[1]);
        if direction == DctDirection::Forward {
            io[0] = vmulq_n_f32(tmp0, 0.5);
            io[1] = vmulq_n_f32(tmp1, 0.5);
        } else {
            io[0] = tmp0;
            io[1] = tmp1;
        }
        return;
    }

    if n == 4 {
        if direction == DctDirection::Forward {
            io.copy_from_slice(&dct4_forward([io[0], io[1], io[2], io[3]]));
        } else {
            io.copy_from_slice(&dct4_inverse([io[0], io[1], io[2], io[3]]));
        }
        return;
    }

    if n == 8 {
        if direction == DctDirection::Forward {
            dct8_forward(&mut CutGrid::from_buf(io, 1, 8, 1));
        } else {
            dct8_inverse(&mut CutGrid::from_buf(io, 1, 8, 1));
        }
        return;
    }

    assert!(n.is_power_of_two());

    if direction == DctDirection::Forward {
        let (input0, input1) = scratch.split_at_mut(n / 2);
        for (idx, &sec) in consts::sec_half(n).iter().enumerate() {
            input0[0] = vmulq_n_f32(io[idx].add(io[n - idx - 1]), 0.5);
            input1[idx] = vmulq_n_f32(io[idx].sub(io[n - idx - 1]), sec / 2.0);
        }
        let (output0, output1) = io.split_at_mut(n / 2);
        dct(input0, output0, DctDirection::Forward);
        dct(input1, output1, DctDirection::Forward);
        for (idx, v) in input0.iter().enumerate() {
            io[idx * 2] = *v;
        }
        input1[0] = vmulq_n_f32(input1[0], std::f32::consts::SQRT_2);
        for idx in 0..(n / 2 - 1) {
            io[idx * 2 + 1] = input1[idx].add(input1[idx + 1]);
        }
        io[n - 1] = input1[n / 2 - 1];
    } else {
        let (input0, input1) = scratch.split_at_mut(n / 2);
        for idx in 1..(n / 2) {
            let idx = n / 2 - idx;
            input0[idx] = io[idx * 2];
            input1[idx] = io[idx * 2 + 1].add(io[idx * 2 - 1]);
        }
        input0[0] = io[0];
        input1[0] = vmulq_n_f32(io[1], std::f32::consts::SQRT_2);
        let (output0, output1) = io.split_at_mut(n / 2);
        dct(input0, output0, DctDirection::Inverse);
        dct(input1, output1, DctDirection::Inverse);
        for (idx, &sec) in consts::sec_half(n).iter().enumerate() {
            let r = vmulq_n_f32(input1[idx], sec);
            output0[idx] = input0[idx].add(r);
            output1[n / 2 - idx - 1] = input0[idx].sub(r);
        }
    }
}
