use super::consts;

pub fn dct_2d(io: &mut [f32], scratch: &mut [f32], width: usize, height: usize) {
    let scratch = &mut scratch[..io.len()];
    let mut buf = vec![0.0f32; width.max(height)];

    // Performs row DCT instead of column DCT, it should be okay
    // r x c => c x r
    let row = &mut buf[..width];
    for (y, input_row) in io.chunks_exact_mut(width).enumerate() {
        dct(input_row, row, false);
        for (tmp_row, v) in scratch.chunks_exact_mut(height).zip(&*row) {
            tmp_row[y] = *v;
        }
    }

    // c x r => if c > r then r x c else c x r
    if width <= height {
        for (input_row, output_row) in scratch.chunks_exact_mut(height).zip(io.chunks_exact_mut(height)) {
            dct(input_row, output_row, false);
        }
    } else {
        let col = &mut buf[..height];
        for (x, input_col) in scratch.chunks_exact_mut(height).enumerate() {
            dct(input_col, col, false);
            for (output_row, v) in io.chunks_exact_mut(width).zip(&*col) {
                output_row[x] = *v;
            }
        }
    }
}

pub fn idct_2d(coeffs_output: &mut [f32], scratch: &mut [f32], target_width: usize, target_height: usize) {
    let scratch = &mut scratch[..coeffs_output.len()];
    let width = target_width.max(target_height);
    let height = target_width.min(target_height);
    let mut buf = vec![0.0f32; width];

    // Performs row DCT instead of column DCT, it should be okay
    // r x c => c x r
    let row = &mut buf[..width];
    for (y, input_row) in coeffs_output.chunks_exact_mut(width).enumerate() {
        dct(input_row, row, true);
        for (tmp_row, v) in scratch.chunks_exact_mut(height).zip(&*row) {
            tmp_row[y] = *v;
        }
    }

    // c x r => if c > r then r x c else c x r
    if target_height >= target_width {
        for (input_row, output_row) in scratch.chunks_exact_mut(height).zip(coeffs_output.chunks_exact_mut(height)) {
            dct(input_row, output_row, true);
        }
    } else {
        let col = &mut buf[..height];
        for (x, input_col) in scratch.chunks_exact_mut(height).enumerate() {
            dct(input_col, col, true);
            for (output_row, v) in coeffs_output.chunks_exact_mut(width).zip(&*col) {
                output_row[x] = *v;
            }
        }
    }
}

fn dct(input_scratch: &mut [f32], output: &mut [f32], inverse: bool) {
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
        output[0] = input_scratch[0] + input_scratch[1];
        output[1] = input_scratch[0] - input_scratch[1];
        if !inverse {
            output[0] /= 2.0;
            output[1] /= 2.0;
        }
        return;
    }

    if n == 4 {
        let sec0 = 0.5411961;
        let sec1 = 1.306563;

        let input = input_scratch;
        if !inverse {
            let sum03 = input[0] + input[3];
            let sum12 = input[1] + input[2];
            output[0] = (sum03 + sum12) / 4.0;
            output[2] = (sum03 - sum12) / 4.0;

            let tmp0 = (input[0] - input[3]) * sec0;
            let tmp1 = (input[1] - input[2]) * sec1;
            let out0 = (tmp0 + tmp1) / 4.0;
            let out1 = (tmp0 - tmp1) / 4.0;
            output[1] = out0 * std::f32::consts::SQRT_2 + out1;
            output[3] = out1;
        } else {
            let tmp0 = input[1] * std::f32::consts::SQRT_2;
            let tmp1 = input[1] + input[3];
            let out0 = (tmp0 + tmp1) * sec0;
            let out1 = (tmp0 - tmp1) * sec1;
            let sum02 = input[0] + input[2];
            let sub02 = input[0] - input[2];

            output[0] = sum02 + out0;
            output[1] = sub02 + out1;
            output[2] = sub02 - out1;
            output[3] = sum02 - out0;
        }
        return;
    }
    assert!(n.is_power_of_two());

    if !inverse {
        let (input0, input1) = output.split_at_mut(n / 2);
        for idx in 0..(n / 2) {
            input0[idx] = (input_scratch[idx] + input_scratch[n - idx - 1]) / 2.0;
            input1[idx] = (input_scratch[idx] - input_scratch[n - idx - 1]) / 2.0;
        }
        let (output0, output1) = input_scratch.split_at_mut(n / 2);
        for (v, &sec) in input1.iter_mut().zip(consts::sec_half(n)) {
            *v *= sec;
        }
        dct(input0, output0, false);
        dct(input1, output1, false);
        output1[0] *= std::f32::consts::SQRT_2;
        for idx in 0..(n / 2 - 1) {
            output1[idx] += output1[idx + 1];
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
            input1[n / 2 - idx] += input1[n / 2 - idx - 1];
        }
        input1[0] *= std::f32::consts::SQRT_2;
        let (output0, output1) = input_scratch.split_at_mut(n / 2);
        dct(input0, output0, true);
        dct(input1, output1, true);
        for (v, &sec) in output1.iter_mut().zip(consts::sec_half(n)) {
            *v *= sec;
        }
        for idx in 0..(n / 2) {
            input0[idx] = input_scratch[idx] + input_scratch[idx + n / 2];
            input1[n / 2 - idx - 1] = input_scratch[idx] - input_scratch[idx + n / 2];
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn forward_dct_2() {
        let original = [-1.0, 3.0];
        let mut input = original;
        let mut output = [0.0f32; 2];
        super::dct(&mut input, &mut output, false);

        let s = original.len();
        for (k, output) in output.iter().enumerate() {
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
        let mut input = original;
        let mut output = [0.0f32; 4];
        super::dct(&mut input, &mut output, false);

        let s = original.len();
        for (k, output) in output.iter().enumerate() {
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
        let mut input = original;
        let mut output = [0.0f32; 8];
        super::dct(&mut input, &mut output, false);

        let s = original.len();
        for (k, output) in output.iter().enumerate() {
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
        let mut input = original;
        let mut output = [0.0f32; 2];
        super::dct(&mut input, &mut output, true);

        let s = original.len();
        for (k, output) in output.iter().enumerate() {
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
        let mut input = original;
        let mut output = [0.0f32; 4];
        super::dct(&mut input, &mut output, true);

        let s = original.len();
        for (k, output) in output.iter().enumerate() {
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
        let mut input = original;
        let mut output = [0.0f32; 8];
        super::dct(&mut input, &mut output, true);

        let s = original.len();
        for (k, output) in output.iter().enumerate() {
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
