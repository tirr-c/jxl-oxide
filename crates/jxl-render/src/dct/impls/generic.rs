use super::super::{consts, reorder, small_reorder};

pub fn dct_2d(io: &mut [f32], scratch: &mut [f32], width: usize, height: usize) {
    let mut buf = vec![0.0f32; width.max(height)];

    // Performs row DCT instead of column DCT, it should be okay
    // r x c => c x r
    let row = &mut buf[..width];
    for (y, input_row) in io.chunks_exact(width).enumerate() {
        dct(input_row, row, false);
        for (tmp_row, v) in scratch.chunks_exact_mut(height).zip(&*row) {
            tmp_row[y] = *v;
        }
    }

    // c x r => if c > r then r x c else c x r
    if width <= height {
        for (input_row, output_row) in scratch.chunks_exact(height).zip(io.chunks_exact_mut(height)) {
            dct(input_row, output_row, false);
        }
    } else {
        let col = &mut buf[..height];
        for (x, input_col) in scratch.chunks_exact(height).enumerate() {
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
    for (y, input_row) in coeffs_output.chunks_exact(width).enumerate() {
        dct(input_row, row, true);
        for (tmp_row, v) in scratch.chunks_exact_mut(height).zip(&*row) {
            tmp_row[y] = *v;
        }
    }

    // c x r => if c > r then r x c else c x r
    if target_height >= target_width {
        for (input_row, output_row) in scratch.chunks_exact(height).zip(coeffs_output.chunks_exact_mut(height)) {
            dct(input_row, output_row, true);
        }
    } else {
        let col = &mut buf[..height];
        for (x, input_col) in scratch.chunks_exact(height).enumerate() {
            dct(input_col, col, true);
            for (output_row, v) in coeffs_output.chunks_exact_mut(width).zip(&*col) {
                output_row[x] = *v;
            }
        }
    }
}

pub fn dct(input: &[f32], output: &mut [f32], inverse: bool) {
    let n = input.len();
    assert!(n.is_power_of_two());
    assert!(output.len() == n);
    if n <= 8 {
        let f = match (n, inverse) {
            (0, _) => return,
            (1, false) => small_dct::<1, 4, false>,
            (2, false) => small_dct::<2, 8, false>,
            (4, false) => small_dct::<4, 16, false>,
            (8, false) => small_dct::<8, 32, false>,
            (1, true) => small_dct::<1, 4, true>,
            (2, true) => small_dct::<2, 8, true>,
            (4, true) => small_dct::<4, 16, true>,
            (8, true) => small_dct::<8, 32, true>,
            _ => unreachable!(),
        };
        f(input, output);
        return;
    }

    let mut real = vec![0.0f32; n * 4];
    let mut imag = vec![0.0f32; n * 4];

    if inverse {
        real[..n].copy_from_slice(input);
        for (idx, (i, o)) in input.iter().zip(&mut real[..n]).enumerate() {
            *o = if idx == 0 {
                *i * 2.0
            } else {
                *i * std::f32::consts::SQRT_2
            };
        }
        let (l, r) = real[..2 * n].split_at_mut(n);
        for (i, o) in l[1..].iter().rev().zip(&mut r[1..]) {
            *o = -*i;
        }
        let (l, r) = real.split_at_mut(2 * n);
        for (i, o) in l.iter_mut().zip(r) {
            *o = -*i;
        }
        reorder(&real, &mut imag);
        real.fill(0.0f32);
    } else {
        reorder(input, output);
        for (idx, val) in output.iter().enumerate() {
            real[2 * n + 2 * idx] = *val;
            real[4 * n - 2 * idx - 1] = *val;
        }
    }

    fft_in_place(&mut real, &mut imag);

    if inverse {
        for (i, o) in imag[..2 * n].chunks_exact(2).zip(output) {
            *o = i[1] / 4.0;
        }
    } else {
        let div = (2 * n) as f32;
        for (idx, (i, o)) in real[..n].iter().zip(output).enumerate() {
            let scale = if idx == 0 { div.recip() } else { std::f32::consts::SQRT_2 / div };
            *o = *i * scale;
        }
    }
}

fn small_dct<const N: usize, const N4: usize, const INV: bool>(input: &[f32], output: &mut [f32]) {
    assert_eq!(N * 4, N4);
    assert!(N.is_power_of_two());
    assert!(input.len() == N);
    assert!(output.len() == N);
    assert!(N.trailing_zeros() <= 3);

    let fft = match N {
        0 => return,
        1 => small_fft_in_place::<4>,
        2 => small_fft_in_place::<8>,
        4 => small_fft_in_place::<16>,
        8 => small_fft_in_place::<32>,
        _ => unreachable!(),
    };

    let mut real = [0.0f32; N4];
    let mut imag = [0.0f32; N4];

    if INV {
        for (idx, (i, o)) in input.iter().zip(&mut real[..N]).enumerate() {
            *o = if idx == 0 {
                *i * 2.0
            } else {
                *i * std::f32::consts::SQRT_2
            };
        }
        let (l, r) = real[..2 * N].split_at_mut(N);
        for (i, o) in l[1..].iter().rev().zip(&mut r[1..]) {
            *o = -*i;
        }
        let (l, r) = real.split_at_mut(2 * N);
        for (i, o) in l.iter_mut().zip(r) {
            *o = -*i;
        }
        small_reorder::<N4, _>(&real, &mut imag);
        real.fill(0.0f32);
    } else {
        small_reorder::<N, _>(input, output);
        for (idx, val) in output.iter().enumerate() {
            real[2 * N + 2 * idx] = *val;
            real[4 * N - 2 * idx - 1] = *val;
        }
    }

    fft(&mut real, &mut imag);

    if INV {
        for (i, o) in imag[..2 * N].chunks_exact(2).zip(output) {
            *o = i[1] / 4.0;
        }
    } else {
        let div = (2 * N) as f32;
        for (idx, (i, o)) in real[..N].iter().zip(output).enumerate() {
            let scale = if idx == 0 { div.recip() } else { std::f32::consts::SQRT_2 / div };
            *o = *i * scale;
        }
    }
}

/// Assumes that inputs are reordered.
fn fft_in_place(real: &mut [f32], imag: &mut [f32]) {
    let n = real.len();
    assert!(n.is_power_of_two());
    assert!(imag.len() == n);

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
    assert!(N.is_power_of_two());
    let iters = N.trailing_zeros();
    assert!(iters <= 5);
    assert!(real.len() >= N);
    assert!(imag.len() >= N);

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
    fn forward_dct() {
        let input = [-1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0];
        let mut output = [0.0f32; 8];
        super::dct(&input, &mut output, false);

        let s = input.len();
        for (k, output) in output.iter().enumerate() {
            let mut exp_value = 0.0f32;
            for (n, input) in input.iter().enumerate() {
                let cos = ((k * (2 * n + 1)) as f32 / s as f32 * std::f32::consts::FRAC_PI_2).cos();
                exp_value += *input * cos;
            }
            exp_value /= s as f32;
            if k != 0 {
                exp_value *= std::f32::consts::SQRT_2;
            }

            let q_expected = (exp_value * 65536.0) as i32;
            let q_actual = (*output * 65536.0) as i32;
            assert_eq!(q_expected, q_actual);
        }
    }

    #[test]
    fn backward_dct() {
        let input = [3.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0];
        let mut output = [0.0f32; 8];
        super::dct(&input, &mut output, true);

        let s = input.len();
        for (k, output) in output.iter().enumerate() {
            let mut exp_value = input[0];
            for (n, input) in input.iter().enumerate().skip(1) {
                let cos = ((n * (2 * k + 1)) as f32 / s as f32 * std::f32::consts::FRAC_PI_2).cos();
                exp_value += *input * cos * std::f32::consts::SQRT_2;
            }

            let q_expected = (exp_value * 65536.0) as i32;
            let q_actual = (*output * 65536.0) as i32;
            assert_eq!(q_expected, q_actual);
        }
    }
}
