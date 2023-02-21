fn dct(input: &[f32], output: &mut [f32], inverse: bool) {
    let n = input.len();
    assert!(n.is_power_of_two());
    assert!(output.len() == n);
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

    match n {
        1 => small_fft_in_place::<4>(&mut real, &mut imag),
        2 => small_fft_in_place::<8>(&mut real, &mut imag),
        4 => small_fft_in_place::<16>(&mut real, &mut imag),
        8 => small_fft_in_place::<32>(&mut real, &mut imag),
        _ => fft_in_place(&mut real, &mut imag),
    }

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

/// Assumes that inputs are reordered.
fn fft_in_place(real: &mut [f32], imag: &mut [f32]) {
    let n = real.len();
    assert!(n.is_power_of_two());
    assert!(imag.len() == n);

    let mut m;
    let mut k_iter;
    let mut theta;
    m = 1;
    k_iter = n;
    theta = std::f32::consts::PI * (-2.0);

    for _ in 0..n.trailing_zeros() {
        m <<= 1;
        k_iter >>= 1;
        theta /= 2.0;

        for k in 0..k_iter {
            let k = k * m;
            for j in 0..(m / 2) {
                let theta = theta * (j as f32);
                let (sin, cos) = theta.sin_cos();

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
    const COS_SIN: [f32; 24] = [
        1.0,
        0.98078525,
        0.9238795,
        0.8314696,
        std::f32::consts::FRAC_1_SQRT_2,
        0.55557024,
        0.38268343,
        0.19509032,

        0.0,
        -0.19509032,
        -0.38268343,
        -0.55557024,
        -std::f32::consts::FRAC_1_SQRT_2,
        -0.8314696,
        -0.9238795,
        -0.98078525,

        -1.0,
        -0.98078525,
        -0.9238795,
        -0.8314696,
        -std::f32::consts::FRAC_1_SQRT_2,
        -0.55557024,
        -0.38268343,
        -0.19509032,
    ];

    assert!(N.is_power_of_two());
    let iters = N.trailing_zeros();
    assert!(iters <= 5);
    assert!(real.len() >= N);
    assert!(imag.len() >= N);

    for it in 0..iters {
        let m = 1usize << (it + 1);
        let k_iter = N >> (it + 1);
        let skip = 16usize >> it;

        for k in 0..k_iter {
            let k = k * m;
            for j in 0..(m / 2) {
                let cos = COS_SIN[j * skip];
                let sin = COS_SIN[j * skip + 8];

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

fn reorder(input: &[f32], output: &mut [f32]) {
    let n = input.len();
    assert!(n.is_power_of_two());
    assert!(output.len() >= n);
    let bits = n.trailing_zeros();
    let shift_bits = usize::BITS - bits;

    for (idx, i) in input.iter().enumerate() {
        let target = idx.reverse_bits() >> shift_bits;
        output[target] = *i;
    }
}

pub fn dct_2d_in_place(input: &mut [f32], width: usize, height: usize) {
    let mut tmp = vec![0.0f32; width * height];
    let mut buf = vec![0.0f32; width.max(height)];

    // Performs row DCT instead of column DCT, it should be okay
    // r x c => c x r
    let row = &mut buf[..width];
    for (y, input_row) in input.chunks_exact(width).enumerate() {
        dct(input_row, row, false);
        for (tmp_row, v) in tmp.chunks_exact_mut(height).zip(&*row) {
            tmp_row[y] = *v;
        }
    }

    // c x r => if c > r then r x c else c x r
    if width <= height {
        for (input_row, output_row) in tmp.chunks_exact(height).zip(input.chunks_exact_mut(height)) {
            dct(input_row, output_row, false);
        }
    } else {
        let col = &mut buf[..height];
        for (x, input_col) in tmp.chunks_exact(height).enumerate() {
            dct(input_col, col, false);
            for (output_row, v) in input.chunks_exact_mut(width).zip(&*col) {
                output_row[x] = *v;
            }
        }
    }
}

pub fn idct_2d_in_place(coeffs: &mut [f32], target_width: usize, target_height: usize) {
    let mut tmp = vec![0.0f32; target_width * target_height];
    let width = target_width.max(target_height);
    let height = target_width.min(target_height);
    let mut buf = vec![0.0f32; width];

    // Performs row DCT instead of column DCT, it should be okay
    // r x c => c x r
    let row = &mut buf[..width];
    for (y, input_row) in coeffs.chunks_exact(width).enumerate() {
        dct(input_row, row, true);
        for (tmp_row, v) in tmp.chunks_exact_mut(height).zip(&*row) {
            tmp_row[y] = *v;
        }
    }

    // c x r => if c > r then r x c else c x r
    if target_height >= target_width {
        for (input_row, output_row) in tmp.chunks_exact(height).zip(coeffs.chunks_exact_mut(height)) {
            dct(input_row, output_row, true);
        }
    } else {
        let col = &mut buf[..height];
        for (x, input_col) in tmp.chunks_exact(height).enumerate() {
            dct(input_col, col, true);
            for (output_row, v) in coeffs.chunks_exact_mut(width).zip(&*col) {
                output_row[x] = *v;
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
