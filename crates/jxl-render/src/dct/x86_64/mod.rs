use super::consts;
use std::arch::x86_64::*;

const LANE_SIZE: usize = 4;
type Lane = __m128;

pub fn dct_2d(io: &mut [f32], scratch: &mut [f32], width: usize, height: usize) {
    if width % LANE_SIZE != 0 || height % LANE_SIZE != 0 {
        return super::generic::dct_2d(io, scratch, width, height);
    }

    let target_width = width.max(height);
    let target_height = width.min(height);

    unsafe {
        let mut scratch_lanes = vec![_mm_setzero_ps(); target_width * 2];

        if width <= height {
            column_dct_lane(io, scratch, width, height, &mut scratch_lanes, false);
        } else {
            row_dct_transpose_lane(io, scratch, width, height, &mut scratch_lanes, false);
        }

        // transposed
        row_dct_transpose_lane(scratch, io, target_height, target_width, &mut scratch_lanes, false);
    }
}

pub fn idct_2d(coeffs_output: &mut [f32], scratch: &mut [f32], target_width: usize, target_height: usize) {
    if target_width % LANE_SIZE != 0 || target_height % LANE_SIZE != 0 {
        return super::generic::idct_2d(coeffs_output, scratch, target_width, target_height);
    }

    let width = target_width.max(target_height);
    let height = target_width.min(target_height);

    unsafe {
        let mut scratch_lanes = vec![_mm_setzero_ps(); width * 2];

        if target_width <= target_height {
            column_dct_lane(coeffs_output, scratch, width, height, &mut scratch_lanes, true);
        } else {
            row_dct_transpose_lane(coeffs_output, scratch, width, height, &mut scratch_lanes, true);
        }

        row_dct_transpose_lane(scratch, coeffs_output, target_height, target_width, &mut scratch_lanes, true);
    }
}

unsafe fn column_dct_lane(
    input: &mut [f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    scratch: &mut [Lane],
    inverse: bool,
) {
    let (input_lanes, output_lanes) = scratch[..height * 2].split_at_mut(height);
    for x in (0..width).step_by(LANE_SIZE) {
        for (input, lane) in input.chunks_exact(width).zip(&mut *input_lanes) {
            *lane = _mm_loadu_ps(input[x..][..LANE_SIZE].as_ptr());
        }
        dct(input_lanes, output_lanes, inverse);
        for (output, lane) in output.chunks_exact_mut(width).zip(&*output_lanes) {
            _mm_storeu_ps(output[x..][..LANE_SIZE].as_mut_ptr(), *lane);
        }
    }
}

unsafe fn row_dct_transpose_lane(
    input: &mut [f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    scratch: &mut [Lane],
    inverse: bool,
) {
    let (input_lanes, output_lanes) = scratch[..width * 2].split_at_mut(width);
    for y in (0..height).step_by(LANE_SIZE) {
        let base_ptr = &input[y * width..];

        for (x, lane) in input_lanes.iter_mut().enumerate() {
            *lane = _mm_set_ps(
                base_ptr[x + 3 * width],
                base_ptr[x + 2 * width],
                base_ptr[x + width],
                base_ptr[x],
            );
        }

        dct(input_lanes, output_lanes, inverse);
        for (output, lane) in output.chunks_exact_mut(height).zip(&*output_lanes) {
            _mm_storeu_ps(output[y..][..LANE_SIZE].as_mut_ptr(), *lane);
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

unsafe fn dct(input_scratch: &mut [Lane], output: &mut [Lane], inverse: bool) {
    let n = input_scratch.len();
    assert!(output.len() == n);

    if n == 0 {
        return;
    }
    if n == 1 {
        output[0] = input_scratch[0];
        return;
    }
    if n == 2 {
        output[0] = _mm_add_ps(input_scratch[0], input_scratch[1]);
        output[1] = _mm_sub_ps(input_scratch[0], input_scratch[1]);
        if !inverse {
            let half = _mm_set1_ps(0.5);
            output[0] = _mm_mul_ps(output[0], half);
            output[1] = _mm_mul_ps(output[1], half);
        }
        return;
    }

    if n == 4 {
        let input = input_scratch;
        output.copy_from_slice(&dct4([input[0], input[1], input[2], input[3]], inverse));
        return;
    }

    let half = _mm_set1_ps(0.5);
    let sqrt2 = _mm_set1_ps(std::f32::consts::SQRT_2);
    if n == 8 {
        let sec = consts::sec_half_small(8);
        if !inverse {
            let input0 = std::array::from_fn(|idx| {
                _mm_mul_ps(_mm_add_ps(input_scratch[idx], input_scratch[7 - idx]), half)
            });
            let input1 = std::array::from_fn(|idx| {
                _mm_mul_ps(
                    _mm_mul_ps(
                        _mm_sub_ps(input_scratch[idx], input_scratch[7 - idx]),
                        half,
                    ),
                    _mm_set1_ps(sec[idx]),
                )
            });
            let output0 = dct4(input0, false);
            let mut output1 = dct4(input1, false);
            output1[0] = _mm_mul_ps(output1[0], sqrt2);
            for idx in 0..3 {
                output1[idx] = _mm_add_ps(output1[idx], output1[idx + 1]);
            }
            for (idx, v) in output0.into_iter().enumerate() {
                output[idx * 2] = v;
            }
            for (idx, v) in output1.into_iter().enumerate() {
                output[idx * 2 + 1] = v;
            }
        } else {
            let input0 = std::array::from_fn(|idx| input_scratch[idx * 2]);
            let mut input1 = std::array::from_fn(|idx| input_scratch[idx * 2 + 1]);
            for idx in 1..4 {
                input1[4 - idx] = _mm_add_ps(input1[4 - idx], input1[3 - idx]);
            }
            input1[0] = _mm_mul_ps(input1[0], sqrt2);
            let output0 = dct4(input0, true);
            let mut output1 = dct4(input1, true);
            for (v, &sec) in output1.iter_mut().zip(sec) {
                *v = _mm_mul_ps(*v, _mm_set1_ps(sec));
            }
            for idx in 0..4 {
                output[idx] = _mm_add_ps(output0[idx], output1[idx]);
                output[7 - idx] = _mm_sub_ps(output0[idx], output1[idx]);
            }
        }
    }

    assert!(n.is_power_of_two());

    if !inverse {
        let (input0, input1) = output.split_at_mut(n / 2);
        for idx in 0..(n / 2) {
            input0[idx] = _mm_mul_ps(_mm_add_ps(input_scratch[idx], input_scratch[n - idx - 1]), half);
            input1[idx] = _mm_mul_ps(_mm_sub_ps(input_scratch[idx], input_scratch[n - idx - 1]), half);
        }
        let (output0, output1) = input_scratch.split_at_mut(n / 2);
        for (v, &sec) in input1.iter_mut().zip(consts::sec_half(n)) {
            *v = _mm_mul_ps(*v, _mm_set1_ps(sec));
        }
        dct(input0, output0, false);
        dct(input1, output1, false);
        output1[0] = _mm_mul_ps(output1[0], sqrt2);
        for idx in 0..(n / 2 - 1) {
            output1[idx] = _mm_add_ps(output1[idx], output1[idx + 1]);
        }
        for (idx, v) in output0.iter().enumerate() {
            output[idx * 2] = *v;
        }
        for (idx, v) in output1.iter().enumerate() {
            output[idx * 2 + 1] = *v;
        }
    } else {
        let (input0, input1) = output.split_at_mut(n / 2);
        for idx in 0..(n / 2) {
            input0[idx] = input_scratch[idx * 2];
            input1[idx] = input_scratch[idx * 2 + 1];
        }
        for idx in 1..(n / 2) {
            input1[n / 2 - idx] = _mm_add_ps(input1[n / 2 - idx], input1[n / 2 - idx - 1]);
        }
        input1[0] = _mm_mul_ps(input1[0], sqrt2);
        let (output0, output1) = input_scratch.split_at_mut(n / 2);
        dct(input0, output0, true);
        dct(input1, output1, true);
        for (v, &sec) in output1.iter_mut().zip(consts::sec_half(n)) {
            *v = _mm_mul_ps(*v, _mm_set1_ps(sec));
        }
        for idx in 0..(n / 2) {
            input0[idx] = _mm_add_ps(input_scratch[idx], input_scratch[idx + n / 2]);
            input1[n / 2 - idx - 1] = _mm_sub_ps(input_scratch[idx], input_scratch[idx + n / 2]);
        }
    }
}
