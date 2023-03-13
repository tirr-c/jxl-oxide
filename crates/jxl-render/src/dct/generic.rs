use super::{consts, reorder};

pub fn dct_2d(io: &mut [f32], scratch: &mut [f32], width: usize, height: usize) {
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
        return dct4(input_scratch, output, inverse);
    }
    if n == 8 {
        return dct8(input_scratch, output, inverse);
    }
    assert!(n.is_power_of_two());

    let cos_sin_table_4n = consts::cos_sin(4 * n);

    if inverse {
        output[0] = (input_scratch[0] + input_scratch[n / 2]) * std::f32::consts::SQRT_2;
        output[1] = (input_scratch[0] - input_scratch[n / 2]) * std::f32::consts::SQRT_2;
        for (i, o) in input_scratch[1..n / 2].iter().zip(output[2..].iter_mut().step_by(2)) {
            *o = *i;
        }
        for (i, o) in input_scratch[n / 2 + 1..].iter().rev().zip(output[3..].iter_mut().step_by(2)) {
            *o = *i;
        }
        for (idx, slice) in output.chunks_exact_mut(2).enumerate().skip(1) {
            let [r, i] = slice else { unreachable!() };
            let cos = cos_sin_table_4n[idx];
            let sin = cos_sin_table_4n[idx + n];
            let tr = *r * cos - *i * sin;
            let ti = *i * cos + *r * sin;
            *r = tr;
            *i = ti;
        }
        for idx in 1..(n / 4) {
            let lr = output[idx * 2];
            let li = output[idx * 2 + 1];
            let rr = output[n - idx * 2];
            let ri = output[n - idx * 2 + 1];

            let tr = lr + rr;
            let ti = li - ri;
            let ur = lr - rr;
            let ui = li + ri;

            let cos = cos_sin_table_4n[idx * 4];
            let sin = cos_sin_table_4n[idx * 4 + n];
            let vr = ur * sin + ui * cos;
            let vi = ur * cos - ui * sin;

            output[idx * 2] = tr + vr;
            output[idx * 2 + 1] = vi - ti;
            output[n - idx * 2] = tr - vr;
            output[n - idx * 2 + 1] = vi + ti;
        }
        output[n / 2] *= 2.0;
        output[n / 2 + 1] *= 2.0;
        reorder(output, input_scratch);

        let (real, imag) = input_scratch.split_at_mut(n / 2);
        fft_in_place(imag, real);

        let scale = std::f32::consts::FRAC_1_SQRT_2;
        let it = (0..n).step_by(4).chain((0..n).rev().step_by(4)).zip(real);
        for (idx, i) in it {
            output[idx] = *i * scale;
        }
        let it = (2..n).step_by(4).chain((0..n - 2).rev().step_by(4)).zip(imag);
        for (idx, i) in it {
            output[idx] = *i * scale;
        }
    } else {
        let it = input_scratch.iter().step_by(2).chain(input_scratch.iter().rev().step_by(2)).zip(&mut *output);
        for (i, o) in it {
            *o = *i;
        }
        reorder(output, input_scratch);

        let (real, imag) = input_scratch.split_at_mut(n / 2);
        fft_in_place(real, imag);

        let (oreal, oimag) = output.split_at_mut(n / 2);

        oreal[0] = real[0] + imag[0];
        oimag[0] = real[0] - imag[0];

        for idx in 1..(n / 4) {
            let lr = real[idx];
            let li = imag[idx];
            let rr = real[n / 2 - idx];
            let ri = imag[n / 2 - idx];

            let tr = lr + rr;
            let ti = li - ri;
            let ur = lr - rr;
            let ui = li + ri;

            let cos = cos_sin_table_4n[idx * 4];
            let sin = cos_sin_table_4n[idx * 4 + n];
            let vr = ur * sin + ui * cos;
            let vi = ui * sin - ur * cos;

            oreal[idx] = tr + vr;
            oimag[idx] = ti + vi;
            oreal[n / 2 - idx] = tr - vr;
            oimag[n / 2 - idx] = vi - ti;
        }
        oreal[n / 4] *= 2.0;
        oimag[n / 4] *= -2.0;

        let scale = (n as f32).recip() * std::f32::consts::FRAC_1_SQRT_2;
        for (idx, (r, i)) in oreal.iter_mut().zip(&mut *oimag).enumerate().skip(1) {
            let cos = cos_sin_table_4n[idx];
            let sin = -cos_sin_table_4n[idx + n];

            let tr = *r * cos + *i * sin;
            let ti = *r * sin - *i * cos;
            *r = tr * scale;
            *i = ti * scale;
        }
        oreal[0] /= n as f32;
        oimag[0] /= n as f32;
        oimag[1..].reverse();
    }
}

fn dct4(input: &[f32], output: &mut [f32], inverse: bool) {
    assert_eq!(input.len(), 4);
    assert_eq!(output.len(), 4);

    const COS: f32 = 0.9238795 * std::f32::consts::SQRT_2;
    const SIN: f32 = -0.38268343 * std::f32::consts::SQRT_2;

    if inverse {
        output[0] = input[0] + input[2];
        output[2] = input[0] - input[2];
        let r = input[1];
        let i = input[3];
        let tr = r * COS - i * SIN;
        let ti = i * COS + r * SIN;
        output[1] = tr;
        output[3] = ti;

        let (real, imag) = output.split_at_mut(2);
        small_fft_in_place::<2>(imag, real);

        output.swap(1, 3);
    } else {
        output[0] = input[0];
        output[1] = input[3];
        output[2] = input[2];
        output[3] = input[1];

        let (real, imag) = output.split_at_mut(2);
        small_fft_in_place::<2>(real, imag);

        let l = real[0] / 4.0;
        let r = imag[0] / 4.0;
        real[0] = l + r;
        imag[0] = l - r;

        let r = real[1] / 4.0;
        let i = imag[1] / 4.0;
        let tr = r * COS + i * SIN;
        let ti = i * COS - r * SIN;
        real[1] = tr;
        imag[1] = ti;
    }
}

fn dct8(input: &[f32], output: &mut [f32], inverse: bool) {
    assert_eq!(input.len(), 8);
    assert_eq!(output.len(), 8);

    let cos_sin_table_4n = consts::cos_sin_small(32);
    if inverse {
        output[0] = input[0] + input[4];
        output[4] = input[0] - input[4];

        output[1] = input[2];
        output[2] = input[1];
        output[3] = input[3];

        output[5] = input[6];
        output[6] = input[7];
        output[7] = input[5];

        for (i, idx) in [(2, 6), (1, 5), (3, 7)].into_iter().enumerate() {
            let cos = cos_sin_table_4n[i + 1] * std::f32::consts::FRAC_1_SQRT_2;
            let sin = cos_sin_table_4n[i + 9] * std::f32::consts::FRAC_1_SQRT_2;
            let r = output[idx.0];
            let i = output[idx.1];
            let tr = r * cos - i * sin;
            let ti = i * cos + r * sin;
            output[idx.0] = tr;
            output[idx.1] = ti;
        }
        let lr = output[2];
        let li = output[6];
        let rr = output[3];
        let ri = output[7];

        let tr = lr + rr;
        let ti = li - ri;
        let ur = lr - rr;
        let ui = li + ri;

        let cos = cos_sin_table_4n[4];
        let sin = cos_sin_table_4n[12];
        let vr = ur * sin + ui * cos;
        let vi = ur * cos - ui * sin;

        output[2] = tr + vr;
        output[6] = vi - ti;
        output[3] = tr - vr;
        output[7] = vi + ti;
        output[1] *= 2.0;
        output[5] *= 2.0;

        let (real, imag) = output.split_at_mut(4);
        small_fft_in_place::<4>(imag, real);

        output.swap(5, 6);
        output.swap(2, 1);
        output.swap(4, 2);
        output.swap(1, 7);
    } else {
        output.copy_from_slice(input);
        output.swap(1, 7);
        output.swap(2, 4);

        let (real, imag) = output.split_at_mut(4);
        small_fft_in_place::<4>(real, imag);

        let l = real[0];
        let r = imag[0];
        real[0] = l + r;
        imag[0] = l - r;

        let lr = real[1];
        let li = imag[1];
        let rr = real[3];
        let ri = imag[3];

        let tr = lr + rr;
        let ti = li - ri;
        let ur = lr - rr;
        let ui = li + ri;

        let cos = cos_sin_table_4n[4];
        let sin = cos_sin_table_4n[12];
        let vr = ur * sin + ui * cos;
        let vi = ui * sin - ur * cos;

        real[1] = tr + vr;
        imag[1] = ti + vi;
        real[3] = tr - vr;
        imag[3] = vi - ti;
        real[2] *= 2.0;
        imag[2] *= -2.0;

        let scale = std::f32::consts::FRAC_1_SQRT_2 / 8.0;
        for (idx, (r, i)) in real.iter_mut().zip(&mut *imag).enumerate().skip(1) {
            let cos = cos_sin_table_4n[idx];
            let sin = -cos_sin_table_4n[idx + 8];

            let tr = *r * cos + *i * sin;
            let ti = *r * sin - *i * cos;
            *r = tr * scale;
            *i = ti * scale;
        }
        real[0] /= 8.0;
        imag[0] /= 8.0;
        imag[1..].reverse();
    }
}

/// Assumes that inputs are reordered.
fn fft_in_place(real: &mut [f32], imag: &mut [f32]) {
    let n = real.len();
    if n < 2 {
        return;
    }

    assert!(imag.len() == n);
    if n == 2 {
        let lr = real[0];
        let li = imag[0];
        let rr = real[1];
        let ri = imag[1];
        real[0] = lr + rr;
        imag[0] = li + ri;
        real[1] = lr - rr;
        imag[1] = li - ri;
        return;
    }

    assert!(n.is_power_of_two());
    let cos_sin_table = consts::cos_sin(n);

    let mut m;
    let mut k_iter;
    m = 1;
    k_iter = n;

    for _ in 0..n.trailing_zeros() {
        m <<= 1;
        k_iter >>= 1;

        for k in 0..k_iter {
            let k = k * m;
            for j in 0..(m / 2) {
                let cos = cos_sin_table[j * k_iter];
                let sin = cos_sin_table[j * k_iter + n / 4];

                let r = real[k + m / 2 + j];
                let i = imag[k + m / 2 + j];
                // (a + ib) (cos + isin) = (a cos - b sin) + i(b cos + a sin)
                let tr = r * cos - i * sin;
                let ti = i * cos + r * sin;
                let ur = real[k + j];
                let ui = imag[k + j];

                real[k + j] = ur + tr;
                imag[k + j] = ui + ti;
                real[k + m / 2 + j] = ur - tr;
                imag[k + m / 2 + j] = ui - ti;
            }
        }
    }
}

/// Assumes that inputs are reordered.
fn small_fft_in_place<const N: usize>(real: &mut [f32], imag: &mut [f32]) {
    if N < 2 {
        return;
    }

    assert!(real.len() >= N);
    assert!(imag.len() >= N);
    if N == 2 {
        let lr = real[0];
        let li = imag[0];
        let rr = real[1];
        let ri = imag[1];
        real[0] = lr + rr;
        imag[0] = li + ri;
        real[1] = lr - rr;
        imag[1] = li - ri;
        return;
    }

    assert!(N.is_power_of_two());
    let iters = N.trailing_zeros();
    assert!(iters <= 5);

    let cos_sin_table = consts::cos_sin_small(N);
    for it in 0..iters {
        let m = 1usize << (it + 1);
        let k_iter = N >> (it + 1);

        for k in 0..k_iter {
            let k = k * m;
            for j in 0..(m / 2) {
                let cos = cos_sin_table[j * k_iter];
                let sin = cos_sin_table[j * k_iter + N / 4];

                let ur = real[k + j];
                let ui = imag[k + j];
                let r = real[k + m / 2 + j];
                let i = imag[k + m / 2 + j];
                // (a + ib) (cos + isin) = (a cos - b sin) + i(b cos + a sin)
                let tr = r * cos - i * sin;
                let ti = i * cos + r * sin;

                real[k + j] = ur + tr;
                imag[k + j] = ui + ti;
                real[k + m / 2 + j] = ur - tr;
                imag[k + m / 2 + j] = ui - ti;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn forward_dct_2() {
        let mut input = [-1.0, 3.0];
        let mut output = [0.0f32; 2];
        super::dct(&mut input, &mut output, false);

        let s = input.len();
        for (k, output) in output.iter().enumerate() {
            let mut exp_value = 0.0f64;
            for (n, input) in input.iter().enumerate() {
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
        let mut input = [-1.0, 2.0, 3.0, -4.0];
        let mut output = [0.0f32; 4];
        super::dct(&mut input, &mut output, false);

        let s = input.len();
        for (k, output) in output.iter().enumerate() {
            let mut exp_value = 0.0f64;
            for (n, input) in input.iter().enumerate() {
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
        let mut input = [1.0, 0.3, 1.0, 2.0, -2.0, -0.1, 1.0, 0.1];
        let mut output = [0.0f32; 8];
        super::dct(&mut input, &mut output, false);

        let s = input.len();
        for (k, output) in output.iter().enumerate() {
            let mut exp_value = 0.0f64;
            for (n, input) in input.iter().enumerate() {
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
        let mut input = [3.0, 0.2];
        let mut output = [0.0f32; 2];
        super::dct(&mut input, &mut output, true);

        let s = input.len();
        for (k, output) in output.iter().enumerate() {
            let mut exp_value = input[0] as f64;
            for (n, input) in input.iter().enumerate().skip(1) {
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
        let mut input = [3.0, 0.2, 0.3, -1.0];
        let mut output = [0.0f32; 4];
        super::dct(&mut input, &mut output, true);

        let s = input.len();
        for (k, output) in output.iter().enumerate() {
            let mut exp_value = input[0] as f64;
            for (n, input) in input.iter().enumerate().skip(1) {
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
        let mut input = [3.0, 0.0, 0.0, -1.0, 0.0, 0.3, 0.2, 0.0];
        let mut output = [0.0f32; 8];
        super::dct(&mut input, &mut output, true);

        let s = input.len();
        for (k, output) in output.iter().enumerate() {
            let mut exp_value = input[0] as f64;
            for (n, input) in input.iter().enumerate().skip(1) {
                let cos = ((n * (2 * k + 1)) as f64 / s as f64 * std::f64::consts::FRAC_PI_2).cos();
                exp_value += *input as f64 * cos * std::f64::consts::SQRT_2;
            }

            let q_expected = (exp_value * 65536.0) as i32;
            let q_actual = (*output * 65536.0) as i32;
            assert_eq!(q_expected, q_actual);
        }
    }
}
