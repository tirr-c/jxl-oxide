use jxl_grid::{CutGrid, SimdVector};

use super::consts;
use std::arch::x86_64::*;

const LANE_SIZE: usize = 4;
type Lane = __m128;

fn transpose_lane(lanes: &mut [Lane]) {
    let [row0, row1, row2, row3] = lanes else { panic!() };
    unsafe { _MM_TRANSPOSE4_PS(row0, row1, row2, row3); }
}

pub fn dct_2d(io: &mut CutGrid<'_>) {
    dct_2d_generic(io, false)
}

pub fn idct_2d(io: &mut CutGrid<'_>) {
    dct_2d_generic(io, true)
}

pub fn dct_2d_generic(io: &mut CutGrid<'_>, inverse: bool) {
    if io.width() % LANE_SIZE != 0 || io.height() % LANE_SIZE != 0 {
        return super::generic::dct_2d_generic(io, inverse);
    }

    let Some(mut io) = io.as_vectored() else {
        tracing::trace!("Input buffer is not aligned");
        return super::generic::dct_2d_generic(io, inverse);
    };
    dct_2d_lane(&mut io, inverse);
}

fn dct_2d_lane(io: &mut CutGrid<'_, Lane>, inverse: bool) {
    let scratch_size = io.height().max(io.width() * LANE_SIZE) * 2;
    unsafe {
        let mut scratch_lanes = vec![_mm_setzero_ps(); scratch_size];
        column_dct_lane(io, &mut scratch_lanes, inverse);
        row_dct_lane(io, &mut scratch_lanes, inverse);
    }
}

fn column_dct_lane(
    io: &mut CutGrid<'_, Lane>,
    scratch: &mut [Lane],
    inverse: bool,
) {
    let width = io.width();
    let height = io.height();
    let (io_lanes, scratch_lanes) = scratch[..height * 2].split_at_mut(height);
    for x in 0..width {
        for (y, input) in io_lanes.iter_mut().enumerate() {
            *input = io.get(x, y);
        }
        dct(io_lanes, scratch_lanes, inverse);
        for (y, output) in io_lanes.chunks_exact_mut(LANE_SIZE).enumerate() {
            transpose_lane(output);
            for (dy, output) in output.iter_mut().enumerate() {
                *io.get_mut(x, y * LANE_SIZE + dy) = *output;
            }
        }
    }
}

fn row_dct_lane(
    io: &mut CutGrid<'_, Lane>,
    scratch: &mut [Lane],
    inverse: bool,
) {
    let width = io.width() * LANE_SIZE;
    let height = io.height();
    let (io_lanes, scratch_lanes) = scratch[..width * 2].split_at_mut(width);
    for y in (0..height).step_by(LANE_SIZE) {
        for (x, input) in io_lanes.chunks_exact_mut(LANE_SIZE).enumerate() {
            for (dy, input) in input.iter_mut().enumerate() {
                *input = io.get(x, y + dy);
            }
        }
        dct(io_lanes, scratch_lanes, inverse);
        for (x, output) in io_lanes.chunks_exact_mut(LANE_SIZE).enumerate() {
            transpose_lane(output);
            for (dy, output) in output.iter_mut().enumerate() {
                *io.get_mut(x, y + dy) = *output;
            }
        }
    }
}

fn dct4(input: [Lane; 4], inverse: bool) -> [Lane; 4] {
    let sec0 = Lane::splat_f32(0.5411961);
    let sec1 = Lane::splat_f32(1.306563);

    let quarter = Lane::splat_f32(0.25);
    let sqrt2 = Lane::splat_f32(std::f32::consts::SQRT_2);
    if !inverse {
        let sum03 = input[0].add(input[3]);
        let sum12 = input[1].add(input[2]);
        let tmp0 = input[0].sub(input[3]).mul(sec0);
        let tmp1 = input[1].sub(input[2]).mul(sec1);
        let out0 = tmp0.add(tmp1).mul(quarter);
        let out1 = tmp0.sub(tmp1).mul(quarter);

        [
            sum03.add(sum12).mul(quarter),
            out0.muladd(sqrt2, out1),
            sum03.sub(sum12).mul(quarter),
            out1,
        ]
    } else {
        let tmp0 = input[1].mul(sqrt2);
        let tmp1 = input[1].add(input[3]);
        let out0 = tmp0.add(tmp1).mul(sec0);
        let out1 = tmp0.sub(tmp1).mul(sec1);
        let sum02 = input[0].add(input[2]);
        let sub02 = input[0].sub(input[2]);

        [
            sum02.add(out0),
            sub02.add(out1),
            sub02.sub(out1),
            sum02.sub(out0),
        ]
    }
}

fn dct(io: &mut [Lane], scratch: &mut [Lane], inverse: bool) {
    let n = io.len();
    assert!(scratch.len() == n);

    if n == 0 {
        return;
    }
    if n == 1 {
        return;
    }

    let half = Lane::splat_f32(0.5);
    if n == 2 {
        let tmp0 = io[0].add(io[1]);
        let tmp1 = io[0].sub(io[1]);
        if inverse {
            io[0] = tmp0;
            io[1] = tmp1;
        } else {
            io[0] = tmp0.mul(half);
            io[1] = tmp1.mul(half);
        }
        return;
    }

    if n == 4 {
        io.copy_from_slice(&dct4([io[0], io[1], io[2], io[3]], inverse));
        return;
    }

    let sqrt2 = Lane::splat_f32(std::f32::consts::SQRT_2);
    if n == 8 {
        let sec = consts::sec_half_small(8);
        if !inverse {
            let input0 = [
                io[0].add(io[7]).mul(half),
                io[1].add(io[6]).mul(half),
                io[2].add(io[5]).mul(half),
                io[3].add(io[4]).mul(half),
            ];
            let input1 = [
                io[0].sub(io[7]).mul(Lane::splat_f32(sec[0] / 2.0)),
                io[1].sub(io[6]).mul(Lane::splat_f32(sec[1] / 2.0)),
                io[2].sub(io[5]).mul(Lane::splat_f32(sec[2] / 2.0)),
                io[3].sub(io[4]).mul(Lane::splat_f32(sec[3] / 2.0)),
            ];
            let output0 = dct4(input0, false);
            for (idx, v) in output0.into_iter().enumerate() {
                io[idx * 2] = v;
            }
            let mut output1 = dct4(input1, false);
            output1[0] = output1[0].mul(sqrt2);
            for idx in 0..3 {
                io[idx * 2 + 1] = output1[idx].add(output1[idx + 1]);
            }
            io[7] = output1[3];
        } else {
            let input0 = [io[0], io[2], io[4], io[6]];
            let input1 = [
                io[1].mul(sqrt2),
                io[3].add(io[1]),
                io[5].add(io[3]),
                io[7].add(io[5]),
            ];
            let output0 = dct4(input0, true);
            let output1 = dct4(input1, true);
            for (idx, &sec) in sec.iter().enumerate() {
                let r = output1[idx].mul(Lane::splat_f32(sec));
                io[idx] = output0[idx].add(r);
                io[7 - idx] = output0[idx].sub(r);
            }
        }
        return;
    }

    assert!(n.is_power_of_two());

    if !inverse {
        let (input0, input1) = scratch.split_at_mut(n / 2);
        for (idx, &sec) in consts::sec_half(n).iter().enumerate() {
            input0[idx] = io[idx].add(io[n - idx - 1]).mul(half);
            input1[idx] = io[idx].sub(io[n - idx - 1]).mul(Lane::splat_f32(sec / 2.0));
        }
        let (output0, output1) = io.split_at_mut(n / 2);
        dct(input0, output0, false);
        dct(input1, output1, false);
        for (idx, v) in input0.iter().enumerate() {
            io[idx * 2] = *v;
        }
        input1[0] = input1[0].mul(sqrt2);
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
        input1[0] = io[1].mul(sqrt2);
        let (output0, output1) = io.split_at_mut(n / 2);
        dct(input0, output0, true);
        dct(input1, output1, true);
        for (idx, &sec) in consts::sec_half(n).iter().enumerate() {
            let r = input1[idx].mul(Lane::splat_f32(sec));
            output0[idx] = input0[idx].add(r);
            output1[n / 2 - idx - 1] = input0[idx].sub(r);
        }
    }
}
