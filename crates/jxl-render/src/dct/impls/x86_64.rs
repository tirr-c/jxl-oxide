use super::super::{consts, reorder, small_reorder};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub fn dct_2d(io: &mut [f32], scratch: &mut [f32], width: usize, height: usize) {
    if width % 4 != 0 || height % 4 != 0 {
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
    if target_width % 4 != 0 || target_height % 4 != 0 {
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

unsafe fn dct_lane(input: &[__m128], output: &mut [__m128], inverse: bool) {
    let n = input.len();
    assert!(n.is_power_of_two());
    assert!(output.len() == n);
    if n <= 8 {
        let f = match (n, inverse) {
            (0, _) => return,
            (1, false) => small_dct_lane::<1, 4, false>,
            (2, false) => small_dct_lane::<2, 8, false>,
            (4, false) => small_dct_lane::<4, 16, false>,
            (8, false) => small_dct_lane::<8, 32, false>,
            (1, true) => small_dct_lane::<1, 4, true>,
            (2, true) => small_dct_lane::<2, 8, true>,
            (4, true) => small_dct_lane::<4, 16, true>,
            (8, true) => small_dct_lane::<8, 32, true>,
            _ => unreachable!(),
        };
        f(input, output);
        return;
    }

    let mut real = vec![_mm_setzero_ps(); n * 4];
    let mut imag = vec![_mm_setzero_ps(); n * 4];

    if inverse {
        real[..n].copy_from_slice(input);
        for (idx, (i, o)) in input.iter().zip(&mut real[..n]).enumerate() {
            let mulvec = if idx == 0 {
                2.0f32
            } else {
                std::f32::consts::SQRT_2
            };
            let mulvec = _mm_set1_ps(mulvec);
            *o = _mm_mul_ps(*i, mulvec);
        }
        let neg = _mm_set1_ps(-1.0);
        let (l, r) = real[..2 * n].split_at_mut(n);
        for (i, o) in l[1..].iter().rev().zip(&mut r[1..]) {
            *o = _mm_mul_ps(*i, neg);
        }
        let (l, r) = real.split_at_mut(2 * n);
        for (i, o) in l.iter_mut().zip(r) {
            *o = _mm_mul_ps(*i, neg);
        }
        reorder(&real, &mut imag);
        real.fill(_mm_setzero_ps());
    } else {
        reorder(input, output);
        for (idx, val) in output.iter().enumerate() {
            real[2 * n + 2 * idx] = *val;
            real[4 * n - 2 * idx - 1] = *val;
        }
    }

    fft_in_place(&mut real, &mut imag);

    if inverse {
        let scale = _mm_set1_ps(0.25);
        for (i, o) in imag[..2 * n].chunks_exact(2).zip(output) {
            *o = _mm_mul_ps(i[1], scale);
        }
    } else {
        let div = (2 * n) as f32;
        for (idx, (i, o)) in real[..n].iter().zip(output).enumerate() {
            let scale = if idx == 0 { div.recip() } else { std::f32::consts::SQRT_2 / div };
            let scale = _mm_set1_ps(scale);
            *o = _mm_mul_ps(*i, scale);
        }
    }
}

unsafe fn small_dct_lane<const N: usize, const N4: usize, const INV: bool>(
    input: &[__m128],
    output: &mut [__m128],
) {
    assert_eq!(N * 4, N4);
    assert!(N.is_power_of_two());
    assert!(input.len() == N);
    assert!(output.len() == N);
    assert!(N.trailing_zeros() <= 3);

    let fft_lane = match N {
        0 => return,
        1 => small_fft_in_place::<4>,
        2 => small_fft_in_place::<8>,
        4 => small_fft_in_place::<16>,
        8 => small_fft_in_place::<32>,
        _ => unreachable!(),
    };

    let mut real = [_mm_setzero_ps(); N4];
    let mut imag = [_mm_setzero_ps(); N4];

    if INV {
        for (idx, (i, o)) in input.iter().zip(&mut real[..N]).enumerate() {
            let mulvec = if idx == 0 {
                2.0f32
            } else {
                std::f32::consts::SQRT_2
            };
            let mulvec = _mm_set1_ps(mulvec);
            *o = _mm_mul_ps(*i, mulvec);
        }
        let neg = _mm_set1_ps(-1.0);
        let (l, r) = real[..2 * N].split_at_mut(N);
        for (i, o) in l[1..].iter().rev().zip(&mut r[1..]) {
            *o = _mm_mul_ps(*i, neg);
        }
        let (l, r) = real.split_at_mut(2 * N);
        for (i, o) in l.iter_mut().zip(r) {
            *o = _mm_mul_ps(*i, neg);
        }
        small_reorder::<N4, _>(&real, &mut imag);
        real.fill(_mm_setzero_ps());
    } else {
        small_reorder::<N, _>(input, output);
        for (idx, val) in output.iter().enumerate() {
            real[2 * N + 2 * idx] = *val;
            real[4 * N - 2 * idx - 1] = *val;
        }
    }

    fft_lane(&mut real, &mut imag);

    if INV {
        let scale = _mm_set1_ps(0.25);
        for (i, o) in imag[..2 * N].chunks_exact(2).zip(output) {
            *o = _mm_mul_ps(i[1], scale);
        }
    } else {
        let div = (2 * N) as f32;
        for (idx, (i, o)) in real[..N].iter().zip(output).enumerate() {
            let scale = if idx == 0 { div.recip() } else { std::f32::consts::SQRT_2 / div };
            let scale = _mm_set1_ps(scale);
            *o = _mm_mul_ps(*i, scale);
        }
    }
}

/// Assumes that inputs are reordered.
unsafe fn fft_in_place(real: &mut [__m128], imag: &mut [__m128]) {
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
                let cos = _mm_set1_ps(cos);
                let sin = cos_sin_table[j * k_iter + n / 4];
                let sin = _mm_set1_ps(sin);

                let r = real[k + m / 2 + j];
                let i = imag[k + m / 2 + j];
                // (a + ib) (cos + isin) = (a cos - b sin) + i(b cos + a sin)
                let rcos = _mm_mul_ps(r, cos);
                let rsin = _mm_mul_ps(r, sin);
                let icos = _mm_mul_ps(i, cos);
                let isin = _mm_mul_ps(i, sin);
                let tr = _mm_sub_ps(rcos, isin);
                let ti = _mm_add_ps(icos, rsin);

                let ur = real[k + j];
                let ui = imag[k + j];

                real[k + j] = _mm_add_ps(ur, tr);
                imag[k + j] = _mm_add_ps(ui, ti);
                real[k + m / 2 + j] = _mm_sub_ps(ur, tr);
                imag[k + m / 2 + j] = _mm_sub_ps(ui, ti);
            }
        }
    }
}

/// Assumes that inputs are reordered.
unsafe fn small_fft_in_place<const N: usize>(real: &mut [__m128], imag: &mut [__m128]) {
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
                let cos = _mm_set1_ps(cos);
                let sin = cos_sin_table[j * k_iter + N / 4];
                let sin = _mm_set1_ps(sin);

                let r = real[k + m / 2 + j];
                let i = imag[k + m / 2 + j];
                // (a + ib) (cos + isin) = (a cos - b sin) + i(b cos + a sin)
                let rcos = _mm_mul_ps(r, cos);
                let rsin = _mm_mul_ps(r, sin);
                let icos = _mm_mul_ps(i, cos);
                let isin = _mm_mul_ps(i, sin);
                let tr = _mm_sub_ps(rcos, isin);
                let ti = _mm_add_ps(icos, rsin);

                let ur = real[k + j];
                let ui = imag[k + j];

                real[k + j] = _mm_add_ps(ur, tr);
                imag[k + j] = _mm_add_ps(ui, ti);
                real[k + m / 2 + j] = _mm_sub_ps(ur, tr);
                imag[k + m / 2 + j] = _mm_sub_ps(ui, ti);
            }
        }
    }
}

unsafe fn column_dct_lane(
    input: &mut [f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    scratch: &mut [__m128],
    inverse: bool,
) {
    let (input_lanes, output_lanes) = scratch[..height * 2].split_at_mut(height);
    for x in (0..width).step_by(4) {
        for (input, lane) in input.chunks_exact(width).zip(&mut *input_lanes) {
            *lane = _mm_loadu_ps(input[x..][..4].as_ptr());
        }
        dct_lane(input_lanes, output_lanes, inverse);
        for (output, lane) in output.chunks_exact_mut(width).zip(&*output_lanes) {
            _mm_storeu_ps(output[x..][..4].as_mut_ptr(), *lane);
        }
    }
}

unsafe fn row_dct_transpose_lane(
    input: &mut [f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    scratch: &mut [__m128],
    inverse: bool,
) {
    let (input_lanes, output_lanes) = scratch[..width * 2].split_at_mut(width);
    for y in (0..height).step_by(4) {
        let base_ptr = &input[y * width..];

        for (x, lane) in input_lanes.iter_mut().enumerate() {
            *lane = _mm_set_ps(base_ptr[x + 3 * width], base_ptr[x + 2 * width], base_ptr[x + width], base_ptr[x]);
        }

        dct_lane(input_lanes, output_lanes, inverse);
        for (output, lane) in output.chunks_exact_mut(height).zip(&*output_lanes) {
            _mm_storeu_ps(output[y..][..4].as_mut_ptr(), *lane);
        }
    }
}
