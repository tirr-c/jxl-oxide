pub fn dct(input: &[f32], output: &mut [f32], inverse: bool) {
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
            eprintln!("output={output}, exp_value={exp_value}");

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
            eprintln!("output={output}, exp_value={exp_value}");

            let q_expected = (exp_value * 65536.0) as i32;
            let q_actual = (*output * 65536.0) as i32;
            assert_eq!(q_expected, q_actual);
        }
    }
}
