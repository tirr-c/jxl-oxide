use jxl_grid::CutGrid;

use super::{consts, DctDirection};

pub fn dct_2d(io: &mut CutGrid<'_>, direction: DctDirection) {
    let width = io.width();
    let height = io.height();
    let mut buf = vec![0f32; width.max(height)];

    let row = &mut buf[..width];
    for y in 0..height {
        dct(io.get_row_mut(y), row, direction);
    }

    let block_size = width.min(height);
    for by in (0..height).step_by(block_size) {
        for bx in (0..width).step_by(block_size) {
            for dy in 0..block_size {
                for dx in (dy + 1)..block_size {
                    io.swap((bx + dx, by + dy), (bx + dy, by + dx));
                }
            }
        }
    }

    let scratch = &mut buf[..height];
    if block_size == height {
        for y in 0..height {
            let grouped_row = io.get_row_mut(y);
            for row in grouped_row.chunks_exact_mut(height) {
                dct(row, scratch, direction);
            }
        }
    } else {
        let mut row = vec![0f32; height];
        for y in 0..width {
            for (idx, chunk) in row.chunks_exact_mut(width).enumerate() {
                let y = y + idx * block_size;
                chunk.copy_from_slice(io.get_row(y));
            }
            dct(&mut row, scratch, direction);
            for (idx, chunk) in row.chunks_exact(width).enumerate() {
                let y = y + idx * block_size;
                io.get_row_mut(y).copy_from_slice(chunk);
            }
        }
    }

    for by in (0..height).step_by(block_size) {
        for bx in (0..width).step_by(block_size) {
            for dy in 0..block_size {
                for dx in (dy + 1)..block_size {
                    io.swap((bx + dx, by + dy), (bx + dy, by + dx));
                }
            }
        }
    }
}

fn dct4(input: [f32; 4], direction: DctDirection) -> [f32; 4] {
    let sec0 = 0.5411961;
    let sec1 = 1.306563;

    if direction == DctDirection::Forward {
        let sum03 = input[0] + input[3];
        let sum12 = input[1] + input[2];
        let tmp0 = (input[0] - input[3]) * sec0;
        let tmp1 = (input[1] - input[2]) * sec1;
        let out0 = (tmp0 + tmp1) / 4.0;
        let out1 = (tmp0 - tmp1) / 4.0;

        [
            (sum03 + sum12) / 4.0,
            out0 * std::f32::consts::SQRT_2 + out1,
            (sum03 - sum12) / 4.0,
            out1,
        ]
    } else {
        let tmp0 = input[1] * std::f32::consts::SQRT_2;
        let tmp1 = input[1] + input[3];
        let out0 = (tmp0 + tmp1) * sec0;
        let out1 = (tmp0 - tmp1) * sec1;
        let sum02 = input[0] + input[2];
        let sub02 = input[0] - input[2];

        [
            sum02 + out0,
            sub02 + out1,
            sub02 - out1,
            sum02 - out0,
        ]
    }
}

fn dct(input_output: &mut [f32], scratch: &mut [f32], direction: DctDirection) {
    let n = input_output.len();
    assert!(scratch.len() == n);

    if n == 0 {
        return;
    }
    if n == 1 {
        return;
    }
    if n == 2 {
        let tmp0 = input_output[0] + input_output[1];
        let tmp1 = input_output[0] - input_output[1];
        if direction == DctDirection::Forward {
            input_output[0] = tmp0 / 2.0;
            input_output[1] = tmp1 / 2.0;
        } else {
            input_output[0] = tmp0;
            input_output[1] = tmp1;
        }
        return;
    }

    if n == 4 {
        let io = input_output;
        io.copy_from_slice(&dct4([io[0], io[1], io[2], io[3]], direction));
        return;
    }

    if n == 8 {
        let io = input_output;
        let sec = consts::sec_half_small(8);
        if direction == DctDirection::Forward {
            let input0 = [
                (io[0] + io[7]) / 2.0,
                (io[1] + io[6]) / 2.0,
                (io[2] + io[5]) / 2.0,
                (io[3] + io[4]) / 2.0,
            ];
            let input1 = [
                (io[0] - io[7]) * sec[0] / 2.0,
                (io[1] - io[6]) * sec[1] / 2.0,
                (io[2] - io[5]) * sec[2] / 2.0,
                (io[3] - io[4]) * sec[3] / 2.0,
            ];
            let output0 = dct4(input0, DctDirection::Forward);
            for (idx, v) in output0.into_iter().enumerate() {
                io[idx * 2] = v;
            }
            let mut output1 = dct4(input1, DctDirection::Forward);
            output1[0] *= std::f32::consts::SQRT_2;
            for idx in 0..3 {
                io[idx * 2 + 1] = output1[idx] + output1[idx + 1];
            }
            io[7] = output1[3];
        } else {
            let input0 = [io[0], io[2], io[4], io[6]];
            let input1 = [
                io[1] * std::f32::consts::SQRT_2,
                io[3] + io[1],
                io[5] + io[3],
                io[7] + io[5],
            ];
            let output0 = dct4(input0, DctDirection::Inverse);
            let output1 = dct4(input1, DctDirection::Inverse);
            for (idx, &sec) in sec.iter().enumerate() {
                let r = output1[idx] * sec;
                io[idx] = output0[idx] + r;
                io[7 - idx] = output0[idx] - r;
            }
        }
        return;
    }

    assert!(n.is_power_of_two());

    if direction == DctDirection::Forward {
        let (input0, input1) = scratch.split_at_mut(n / 2);
        for idx in 0..(n / 2) {
            input0[idx] = (input_output[idx] + input_output[n - idx - 1]) / 2.0;
            input1[idx] = (input_output[idx] - input_output[n - idx - 1]) / 2.0;
        }
        let (output0, output1) = input_output.split_at_mut(n / 2);
        for (v, &sec) in input1.iter_mut().zip(consts::sec_half(n)) {
            *v *= sec;
        }
        dct(input0, output0, DctDirection::Forward);
        dct(input1, output1, DctDirection::Forward);
        input1[0] *= std::f32::consts::SQRT_2;
        for idx in 0..(n / 2 - 1) {
            input1[idx] += input1[idx + 1];
        }
        for (idx, v) in input0.iter().enumerate() {
            input_output[idx * 2] = *v;
        }
        for (idx, v) in input1.iter().enumerate() {
            input_output[idx * 2 + 1] = *v;
        }
    } else {
        let (input0, input1) = scratch.split_at_mut(n / 2);
        for idx in 0..(n / 2) {
            input0[idx] = input_output[idx * 2];
            input1[idx] = input_output[idx * 2 + 1];
        }
        for idx in 1..(n / 2) {
            input1[n / 2 - idx] += input1[n / 2 - idx - 1];
        }
        input1[0] *= std::f32::consts::SQRT_2;
        let (output0, output1) = input_output.split_at_mut(n / 2);
        dct(input0, output0, DctDirection::Inverse);
        dct(input1, output1, DctDirection::Inverse);
        for (v, &sec) in input1.iter_mut().zip(consts::sec_half(n)) {
            *v *= sec;
        }
        for idx in 0..(n / 2) {
            output0[idx] = scratch[idx] + scratch[idx + n / 2];
            output1[n / 2 - idx - 1] = scratch[idx] - scratch[idx + n / 2];
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::dct::DctDirection;

    #[test]
    fn forward_dct_2() {
        let original = [-1.0, 3.0];
        let mut io = original;
        let mut scratch = [0.0f32; 2];
        super::dct(&mut io, &mut scratch, DctDirection::Forward);

        let s = original.len();
        for (k, output) in io.iter().enumerate() {
            let mut exp_value = 0.0f64;
            for (n, input) in original.iter().enumerate() {
                let cos = ((k * (2 * n + 1)) as f64 / s as f64 * std::f64::consts::FRAC_PI_2).cos();
                exp_value += *input as f64 * cos;
            }
            exp_value /= s as f64;
            if k != 0 {
                exp_value *= std::f64::consts::SQRT_2;
            }

            let q_expected = (exp_value * 65536.0) as i32;
            let q_actual = (*output * 65536.0) as i32;
            assert_eq!(q_expected, q_actual);
        }
    }

    #[test]
    fn forward_dct_4() {
        let original = [-1.0, 2.0, 3.0, -4.0];
        let mut io = original;
        let mut scratch = [0.0f32; 4];
        super::dct(&mut io, &mut scratch, DctDirection::Forward);

        let s = original.len();
        for (k, output) in io.iter().enumerate() {
            let mut exp_value = 0.0f64;
            for (n, input) in original.iter().enumerate() {
                let cos = ((k * (2 * n + 1)) as f64 / s as f64 * std::f64::consts::FRAC_PI_2).cos();
                exp_value += *input as f64 * cos;
            }
            exp_value /= s as f64;
            if k != 0 {
                exp_value *= std::f64::consts::SQRT_2;
            }

            let q_expected = (exp_value * 65536.0) as i32;
            let q_actual = (*output * 65536.0) as i32;
            assert_eq!(q_expected, q_actual);
        }
    }

    #[test]
    fn forward_dct_8() {
        let original = [1.0, 0.3, 1.0, 2.0, -2.0, -0.1, 1.0, 0.1];
        let mut io = original;
        let mut scratch = [0.0f32; 8];
        super::dct(&mut io, &mut scratch, DctDirection::Forward);

        let s = original.len();
        for (k, output) in io.iter().enumerate() {
            let mut exp_value = 0.0f64;
            for (n, input) in original.iter().enumerate() {
                let cos = ((k * (2 * n + 1)) as f64 / s as f64 * std::f64::consts::FRAC_PI_2).cos();
                exp_value += *input as f64 * cos;
            }
            exp_value /= s as f64;
            if k != 0 {
                exp_value *= std::f64::consts::SQRT_2;
            }

            let q_expected = (exp_value * 65536.0) as i32;
            let q_actual = (*output * 65536.0) as i32;
            assert_eq!(q_expected, q_actual);
        }
    }

    #[test]
    fn backward_dct_2() {
        let original = [3.0, 0.2];
        let mut io = original;
        let mut scratch = [0.0f32; 2];
        super::dct(&mut io, &mut scratch, DctDirection::Inverse);

        let s = original.len();
        for (k, output) in io.iter().enumerate() {
            let mut exp_value = original[0] as f64;
            for (n, input) in original.iter().enumerate().skip(1) {
                let cos = ((n * (2 * k + 1)) as f64 / s as f64 * std::f64::consts::FRAC_PI_2).cos();
                exp_value += *input as f64 * cos * std::f64::consts::SQRT_2;
            }

            let q_expected = (exp_value * 65536.0) as i32;
            let q_actual = (*output * 65536.0) as i32;
            assert_eq!(q_expected, q_actual);
        }
    }

    #[test]
    fn backward_dct_4() {
        let original = [3.0, 0.2, 0.3, -1.0];
        let mut io = original;
        let mut scratch = [0.0f32; 4];
        super::dct(&mut io, &mut scratch, DctDirection::Inverse);

        let s = original.len();
        for (k, output) in io.iter().enumerate() {
            let mut exp_value = original[0] as f64;
            for (n, input) in original.iter().enumerate().skip(1) {
                let cos = ((n * (2 * k + 1)) as f64 / s as f64 * std::f64::consts::FRAC_PI_2).cos();
                exp_value += *input as f64 * cos * std::f64::consts::SQRT_2;
            }

            let q_expected = (exp_value * 65536.0) as i32;
            let q_actual = (*output * 65536.0) as i32;
            assert_eq!(q_expected, q_actual);
        }
    }

    #[test]
    fn backward_dct_8() {
        let original = [3.0, 0.0, 0.0, -1.0, 0.0, 0.3, 0.2, 0.0];
        let mut io = original;
        let mut scratch = [0.0f32; 8];
        super::dct(&mut io, &mut scratch, DctDirection::Inverse);

        let s = original.len();
        for (k, output) in io.iter().enumerate() {
            let mut exp_value = original[0] as f64;
            for (n, input) in original.iter().enumerate().skip(1) {
                let cos = ((n * (2 * k + 1)) as f64 / s as f64 * std::f64::consts::FRAC_PI_2).cos();
                exp_value += *input as f64 * cos * std::f64::consts::SQRT_2;
            }

            let q_expected = (exp_value * 65536.0) as i32;
            let q_actual = (*output * 65536.0) as i32;
            assert_eq!(q_expected, q_actual);
        }
    }
}
