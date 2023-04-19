use jxl_grid::SimpleGrid;

use crate::cut_grid::CutGrid;

use super::consts;
use std::arch::x86_64::*;

const LANE_SIZE: usize = 4;
type Lane = __m128;

fn transpose_lane(lanes: &mut [Lane]) {
    let [row0, row1, row2, row3] = lanes else { panic!() };
    unsafe { _MM_TRANSPOSE4_PS(row0, row1, row2, row3); }
}

pub fn dct_2d(io: &mut SimpleGrid<f32>) {
    let width = io.width();
    let height = io.height();
    if width % LANE_SIZE != 0 || height % LANE_SIZE != 0 {
        return super::generic::dct_2d(io);
    }

    let io_buf = io.buf_mut();
    dct_2d_generic(io_buf, width, height, false)
}

pub fn dct_2d_generic(io_buf: &mut [f32], width: usize, height: usize, inverse: bool) {
    let mut io = CutGrid::from_buf(io_buf, width, height, width);
    let Some(mut io) = CutGrid::<'_, Lane>::convert_grid(&mut io) else {
        tracing::debug!("Input buffer is not aligned");
        return super::generic::dct_2d_generic(io_buf, width, height, inverse);
    };
    dct_2d_lane(&mut io, inverse);
}

pub fn idct_2d(io: &mut CutGrid<'_>) {
    let Some(mut io) = CutGrid::<'_, Lane>::convert_grid(io) else {
        tracing::debug!("Input buffer is not aligned");
        return super::generic::idct_2d(io);
    };
    dct_2d_lane(&mut io, true);
}

fn dct_2d_lane(io: &mut CutGrid<'_, Lane>, inverse: bool) {
    let scratch_size = io.height().max(io.width() * LANE_SIZE) * 2;
    unsafe {
        let mut scratch_lanes = vec![_mm_setzero_ps(); scratch_size];
        column_dct_lane(io, &mut scratch_lanes, inverse);
        row_dct_lane(io, &mut scratch_lanes, inverse);
    }
}

unsafe fn column_dct_lane(
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

unsafe fn row_dct_lane(
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
            if width != height {
                transpose_lane(output);
            }
            for (dy, output) in output.iter_mut().enumerate() {
                *io.get_mut(x, y + dy) = *output;
            }
        }
    }

    if width == height {
        let mut swap = |x: usize, y: usize, dy: usize| {
            let t = io.get(x, y * LANE_SIZE + dy);
            *io.get_mut(x, y * LANE_SIZE + dy) = io.get(y, x * LANE_SIZE + dy);
            *io.get_mut(y, x * LANE_SIZE + dy) = t;
        };

        for y in 0..height / LANE_SIZE {
            for x in (y + 1)..width / LANE_SIZE {
                swap(x, y, 0);
                swap(x, y, 1);
                swap(x, y, 2);
                swap(x, y, 3);
            }
        }
    }
}

unsafe fn dct4(input: [Lane; 4], inverse: bool) -> [Lane; 4] {
    let sec0 = _mm_set1_ps(0.5411961);
    let sec1 = _mm_set1_ps(1.306563);

    let quarter = _mm_set1_ps(0.25);
    let sqrt2 = _mm_set1_ps(std::f32::consts::SQRT_2);
    if !inverse {
        let sum03 = _mm_add_ps(input[0], input[3]);
        let sum12 = _mm_add_ps(input[1], input[2]);
        let tmp0 = _mm_mul_ps(_mm_sub_ps(input[0], input[3]), sec0);
        let tmp1 = _mm_mul_ps(_mm_sub_ps(input[1], input[2]), sec1);
        let out0 = _mm_mul_ps(_mm_add_ps(tmp0, tmp1), quarter);
        let out1 = _mm_mul_ps(_mm_sub_ps(tmp0, tmp1), quarter);

        [
            _mm_mul_ps(_mm_add_ps(sum03, sum12), quarter),
            _mm_add_ps(_mm_mul_ps(out0, sqrt2), out1),
            _mm_mul_ps(_mm_sub_ps(sum03, sum12), quarter),
            out1,
        ]
    } else {
        let tmp0 = _mm_mul_ps(input[1], sqrt2);
        let tmp1 = _mm_add_ps(input[1], input[3]);
        let out0 = _mm_mul_ps(_mm_add_ps(tmp0, tmp1), sec0);
        let out1 = _mm_mul_ps(_mm_sub_ps(tmp0, tmp1), sec1);
        let sum02 = _mm_add_ps(input[0], input[2]);
        let sub02 = _mm_sub_ps(input[0], input[2]);

        [
            _mm_add_ps(sum02, out0),
            _mm_add_ps(sub02, out1),
            _mm_sub_ps(sub02, out1),
            _mm_sub_ps(sum02, out0),
        ]
    }
}

unsafe fn dct(io: &mut [Lane], scratch: &mut [Lane], inverse: bool) {
    let n = io.len();
    assert!(scratch.len() == n);

    if n == 0 {
        return;
    }
    if n == 1 {
        return;
    }
    if n == 2 {
        let tmp0 = _mm_add_ps(io[0], io[1]);
        let tmp1 = _mm_sub_ps(io[0], io[1]);
        if inverse {
            io[0] = tmp0;
            io[1] = tmp1;
        } else {
            let half = _mm_set1_ps(0.5);
            io[0] = _mm_mul_ps(tmp0, half);
            io[1] = _mm_mul_ps(tmp1, half);
        }
        return;
    }

    if n == 4 {
        io.copy_from_slice(&dct4([io[0], io[1], io[2], io[3]], inverse));
        return;
    }

    let half = _mm_set1_ps(0.5);
    let sqrt2 = _mm_set1_ps(std::f32::consts::SQRT_2);
    if n == 8 {
        let sec = consts::sec_half_small(8);
        if !inverse {
            let input0 = [
                _mm_mul_ps(_mm_add_ps(io[0], io[7]), half),
                _mm_mul_ps(_mm_add_ps(io[1], io[6]), half),
                _mm_mul_ps(_mm_add_ps(io[2], io[5]), half),
                _mm_mul_ps(_mm_add_ps(io[3], io[4]), half),
            ];
            let input1 = [
                _mm_mul_ps(_mm_sub_ps(io[0], io[7]), _mm_set1_ps(sec[0] / 2.0)),
                _mm_mul_ps(_mm_sub_ps(io[1], io[6]), _mm_set1_ps(sec[1] / 2.0)),
                _mm_mul_ps(_mm_sub_ps(io[2], io[5]), _mm_set1_ps(sec[2] / 2.0)),
                _mm_mul_ps(_mm_sub_ps(io[3], io[4]), _mm_set1_ps(sec[3] / 2.0)),
            ];
            let output0 = dct4(input0, false);
            let mut output1 = dct4(input1, false);
            output1[0] = _mm_mul_ps(output1[0], sqrt2);
            for idx in 0..3 {
                output1[idx] = _mm_add_ps(output1[idx], output1[idx + 1]);
            }
            for (idx, v) in output0.into_iter().enumerate() {
                io[idx * 2] = v;
            }
            for (idx, v) in output1.into_iter().enumerate() {
                io[idx * 2 + 1] = v;
            }
        } else {
            let input0 = [io[0], io[2], io[4], io[6]];
            let input1 = [
                _mm_mul_ps(io[1], sqrt2),
                _mm_add_ps(io[3], io[1]),
                _mm_add_ps(io[5], io[3]),
                _mm_add_ps(io[7], io[5]),
            ];
            let output0 = dct4(input0, true);
            let output1 = dct4(input1, true);
            for (idx, &sec) in sec.iter().enumerate() {
                let r = _mm_mul_ps(output1[idx], _mm_set1_ps(sec));
                io[idx] = _mm_add_ps(output0[idx], r);
                io[7 - idx] = _mm_sub_ps(output0[idx], r);
            }
        }
        return;
    }

    assert!(n.is_power_of_two());

    if !inverse {
        let (input0, input1) = scratch.split_at_mut(n / 2);
        for (idx, &sec) in consts::sec_half(n).iter().enumerate() {
            input0[idx] = _mm_mul_ps(_mm_add_ps(io[idx], io[n - idx - 1]), half);
            input1[idx] = _mm_mul_ps(_mm_sub_ps(io[idx], io[n - idx - 1]), _mm_set1_ps(sec / 2.0));
        }
        let (output0, output1) = io.split_at_mut(n / 2);
        dct(input0, output0, false);
        dct(input1, output1, false);
        input1[0] = _mm_mul_ps(input1[0], sqrt2);
        for idx in 0..(n / 2 - 1) {
            input1[idx] = _mm_add_ps(input1[idx], input1[idx + 1]);
        }
        for (idx, v) in input0.iter().enumerate() {
            io[idx * 2] = *v;
        }
        for (idx, v) in input1.iter().enumerate() {
            io[idx * 2 + 1] = *v;
        }
    } else {
        let (input0, input1) = scratch.split_at_mut(n / 2);
        for idx in 1..(n / 2) {
            let idx = n / 2 - idx;
            input0[idx] = io[idx * 2];
            input1[idx] = _mm_add_ps(io[idx * 2 + 1], io[idx * 2 - 1]);
        }
        input0[0] = io[0];
        input1[0] = _mm_mul_ps(io[1], sqrt2);
        let (output0, output1) = io.split_at_mut(n / 2);
        dct(input0, output0, true);
        dct(input1, output1, true);
        for (idx, &sec) in consts::sec_half(n).iter().enumerate() {
            let r = _mm_mul_ps(input1[idx], _mm_set1_ps(sec));
            output0[idx] = _mm_add_ps(input0[idx], r);
            output1[n / 2 - idx - 1] = _mm_sub_ps(input0[idx], r);
        }
    }
}
