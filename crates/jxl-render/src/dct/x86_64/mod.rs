use super::{consts, reorder};
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

unsafe fn dct(input: &[Lane], output: &mut [Lane], inverse: bool) {
    let n = input.len();
    assert!(output.len() == n);

    if n == 0 {
        return;
    }
    if n == 1 {
        output[0] = input[0];
        return;
    }
    if n == 2 {
        output[0] = _mm_add_ps(input[0], input[1]);
        output[1] = _mm_sub_ps(input[0], input[1]);
        if !inverse {
            let half = _mm_set1_ps(0.5);
            output[0] = _mm_mul_ps(output[0], half);
            output[1] = _mm_mul_ps(output[1], half);
        }
        return;
    }
    if n == 4 {
        return dct4(input, output, inverse);
    }
    if n == 8 {
        return dct8(input, output, inverse);
    }
    assert!(n.is_power_of_two());

    let mut scratch = vec![_mm_setzero_ps(); n];
    let cos_sin_table_4n = consts::cos_sin(4 * n);

    if inverse {
        output[0] = _mm_add_ps(input[0], input[n / 2]);
        output[1] = _mm_sub_ps(input[0], input[n / 2]);
        for (i, o) in input[1..n / 2].iter().zip(output[2..].iter_mut().step_by(2)) {
            *o = *i;
        }
        for (i, o) in input[n / 2 + 1..].iter().rev().zip(output[3..].iter_mut().step_by(2)) {
            *o = *i;
        }
        for (idx, slice) in output.chunks_exact_mut(2).enumerate().skip(1) {
            let [r, i] = slice else { unreachable!() };
            let cos = _mm_set1_ps(cos_sin_table_4n[idx] * std::f32::consts::FRAC_1_SQRT_2);
            let sin = _mm_set1_ps(cos_sin_table_4n[idx + n] * std::f32::consts::FRAC_1_SQRT_2);
            let rcos = _mm_mul_ps(*r, cos);
            let rsin = _mm_mul_ps(*r, sin);
            let icos = _mm_mul_ps(*i, cos);
            let isin = _mm_mul_ps(*i, sin);
            let tr = _mm_sub_ps(rcos, isin);
            let ti = _mm_add_ps(icos, rsin);
            *r = tr;
            *i = ti;
        }
        for idx in 1..(n / 4) {
            let lr = output[idx * 2];
            let li = output[idx * 2 + 1];
            let rr = output[n - idx * 2];
            let ri = output[n - idx * 2 + 1];

            let tr = _mm_add_ps(lr, rr);
            let ti = _mm_sub_ps(li, ri);
            let ur = _mm_sub_ps(lr, rr);
            let ui = _mm_add_ps(li, ri);

            let cos = _mm_load1_ps(&cos_sin_table_4n[idx * 4]);
            let sin = _mm_load1_ps(&cos_sin_table_4n[idx * 4 + n]);
            let rcos = _mm_mul_ps(ur, cos);
            let rsin = _mm_mul_ps(ur, sin);
            let icos = _mm_mul_ps(ui, cos);
            let isin = _mm_mul_ps(ui, sin);
            let vr = _mm_add_ps(rsin, icos);
            let vi = _mm_sub_ps(rcos, isin);

            output[idx * 2] = _mm_add_ps(tr, vr);
            output[idx * 2 + 1] = _mm_sub_ps(vi, ti);
            output[n - idx * 2] = _mm_sub_ps(tr, vr);
            output[n - idx * 2 + 1] = _mm_add_ps(vi, ti);
        }
        output[n / 2] = _mm_mul_ps(output[n / 2], _mm_set1_ps(2.0));
        output[n / 2 + 1] = _mm_mul_ps(output[n / 2 + 1], _mm_set1_ps(2.0));
        reorder(output, &mut scratch);

        let (real, imag) = scratch.split_at_mut(n / 2);
        fft_in_place(imag, real);

        let it = (0..n).step_by(4).chain((0..n).rev().step_by(4)).zip(real);
        for (idx, i) in it {
            output[idx] = *i;
        }
        let it = (2..n).step_by(4).chain((0..n - 2).rev().step_by(4)).zip(imag);
        for (idx, i) in it {
            output[idx] = *i;
        }
    } else {
        let it = input.iter().step_by(2).chain(input.iter().rev().step_by(2)).zip(&mut scratch);
        for (i, o) in it {
            *o = *i;
        }
        reorder(&scratch, output);

        let (real, imag) = output.split_at_mut(n / 2);
        fft_in_place(real, imag);

        let l = real[0];
        let r = imag[0];
        real[0] = _mm_add_ps(l, r);
        imag[0] = _mm_sub_ps(l, r);

        for idx in 1..(n / 4) {
            let lr = real[idx];
            let li = imag[idx];
            let rr = real[n / 2 - idx];
            let ri = imag[n / 2 - idx];

            let tr = _mm_add_ps(lr, rr);
            let ti = _mm_sub_ps(li, ri);
            let ur = _mm_sub_ps(lr, rr);
            let ui = _mm_add_ps(li, ri);

            let cos = _mm_load1_ps(&cos_sin_table_4n[idx * 4]);
            let sin = _mm_load1_ps(&cos_sin_table_4n[idx * 4 + n]);
            let rcos = _mm_mul_ps(ur, cos);
            let rsin = _mm_mul_ps(ur, sin);
            let icos = _mm_mul_ps(ui, cos);
            let isin = _mm_mul_ps(ui, sin);
            let vr = _mm_add_ps(rsin, icos);
            let vi = _mm_sub_ps(isin, rcos);

            real[idx] = _mm_add_ps(tr, vr);
            imag[idx] = _mm_add_ps(ti, vi);
            real[n / 2 - idx] = _mm_sub_ps(tr, vr);
            imag[n / 2 - idx] = _mm_sub_ps(vi, ti);
        }
        real[n / 4] = _mm_mul_ps(real[n / 4], _mm_set1_ps(2.0));
        imag[n / 4] = _mm_mul_ps(imag[n / 4], _mm_set1_ps(-2.0));

        let scale = _mm_set1_ps((n as f32).recip() * std::f32::consts::FRAC_1_SQRT_2);
        for (idx, (r, i)) in real.iter_mut().zip(&mut *imag).enumerate().skip(1) {
            let cos = _mm_load1_ps(&cos_sin_table_4n[idx]);
            let sin = _mm_set1_ps(-cos_sin_table_4n[idx + n]);

            let rcos = _mm_mul_ps(*r, cos);
            let rsin = _mm_mul_ps(*r, sin);
            let icos = _mm_mul_ps(*i, cos);
            let isin = _mm_mul_ps(*i, sin);
            let tr = _mm_add_ps(rcos, isin);
            let ti = _mm_sub_ps(rsin, icos);
            *r = _mm_mul_ps(tr, scale);
            *i = _mm_mul_ps(ti, scale);
        }
        let n = _mm_set1_ps(n as f32);
        real[0] = _mm_div_ps(real[0], n);
        imag[0] = _mm_div_ps(imag[0], n);
        imag[1..].reverse();
    }
}

unsafe fn dct4(input: &[Lane], output: &mut [Lane], inverse: bool) {
    assert_eq!(input.len(), 4);
    assert_eq!(output.len(), 4);

    const COS: f32 = 0.9238795;
    const SIN: f32 = -0.38268343;
    let cos = _mm_set1_ps(COS * std::f32::consts::SQRT_2);
    let sin = _mm_set1_ps(SIN * std::f32::consts::SQRT_2);

    if inverse {
        output[0] = _mm_add_ps(input[0], input[2]);
        output[2] = _mm_sub_ps(input[0], input[2]);
        let r = input[1];
        let i = input[3];
        let rcos = _mm_mul_ps(r, cos);
        let rsin = _mm_mul_ps(r, sin);
        let icos = _mm_mul_ps(i, cos);
        let isin = _mm_mul_ps(i, sin);
        let tr = _mm_sub_ps(rcos, isin);
        let ti = _mm_add_ps(icos, rsin);
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

        let one_fourth = _mm_set1_ps(0.25);
        let l = _mm_mul_ps(real[0], one_fourth);
        let r = _mm_mul_ps(imag[0], one_fourth);
        real[0] = _mm_add_ps(l, r);
        imag[0] = _mm_sub_ps(l, r);

        let r = real[1];
        let i = imag[1];
        let rcos = _mm_mul_ps(r, cos);
        let rsin = _mm_mul_ps(r, sin);
        let icos = _mm_mul_ps(i, cos);
        let isin = _mm_mul_ps(i, sin);
        let tr = _mm_add_ps(rcos, isin);
        let ti = _mm_sub_ps(icos, rsin);
        real[1] = _mm_mul_ps(tr, one_fourth);
        imag[1] = _mm_mul_ps(ti, one_fourth);
    }
}

unsafe fn dct8(input: &[Lane], output: &mut [Lane], inverse: bool) {
    assert_eq!(input.len(), 8);
    assert_eq!(output.len(), 8);

    let cos_sin_table_4n = consts::cos_sin_small(32);
    if inverse {
        output[0] = _mm_add_ps(input[0], input[4]);
        output[4] = _mm_sub_ps(input[0], input[4]);

        output[1] = input[2];
        output[2] = input[1];
        output[3] = input[3];

        output[5] = input[6];
        output[6] = input[7];
        output[7] = input[5];

        for (i, idx) in [(2, 6), (1, 5), (3, 7)].into_iter().enumerate() {
            let cos = _mm_set1_ps(cos_sin_table_4n[i + 1] * std::f32::consts::FRAC_1_SQRT_2);
            let sin = _mm_set1_ps(cos_sin_table_4n[i + 9] * std::f32::consts::FRAC_1_SQRT_2);
            let r = output[idx.0];
            let i = output[idx.1];
            let rcos = _mm_mul_ps(r, cos);
            let rsin = _mm_mul_ps(r, sin);
            let icos = _mm_mul_ps(i, cos);
            let isin = _mm_mul_ps(i, sin);
            let tr = _mm_sub_ps(rcos, isin);
            let ti = _mm_add_ps(icos, rsin);
            output[idx.0] = tr;
            output[idx.1] = ti;
        }
        let lr = output[2];
        let li = output[6];
        let rr = output[3];
        let ri = output[7];

        let tr = _mm_add_ps(lr, rr);
        let ti = _mm_sub_ps(li, ri);
        let ur = _mm_sub_ps(lr, rr);
        let ui = _mm_add_ps(li, ri);

        let cos = _mm_set1_ps(cos_sin_table_4n[4]);
        let sin = _mm_set1_ps(cos_sin_table_4n[12]);
        let rcos = _mm_mul_ps(ur, cos);
        let rsin = _mm_mul_ps(ur, sin);
        let icos = _mm_mul_ps(ui, cos);
        let isin = _mm_mul_ps(ui, sin);
        let vr = _mm_add_ps(rsin, icos);
        let vi = _mm_sub_ps(rcos, isin);

        output[2] = _mm_add_ps(tr, vr);
        output[6] = _mm_sub_ps(vi, ti);
        output[3] = _mm_sub_ps(tr, vr);
        output[7] = _mm_add_ps(vi, ti);
        output[1] = _mm_mul_ps(output[1], _mm_set1_ps(2.0));
        output[5] = _mm_mul_ps(output[5], _mm_set1_ps(2.0));

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
        real[0] = _mm_add_ps(l, r);
        imag[0] = _mm_sub_ps(l, r);

        let lr = real[1];
        let li = imag[1];
        let rr = real[3];
        let ri = imag[3];

        let tr = _mm_add_ps(lr, rr);
        let ti = _mm_sub_ps(li, ri);
        let ur = _mm_sub_ps(lr, rr);
        let ui = _mm_add_ps(li, ri);

        let cos = _mm_set1_ps(cos_sin_table_4n[4]);
        let sin = _mm_set1_ps(cos_sin_table_4n[12]);
        let rcos = _mm_mul_ps(ur, cos);
        let rsin = _mm_mul_ps(ur, sin);
        let icos = _mm_mul_ps(ui, cos);
        let isin = _mm_mul_ps(ui, sin);
        let vr = _mm_add_ps(rsin, icos);
        let vi = _mm_sub_ps(isin, rcos);

        real[1] = _mm_add_ps(tr, vr);
        imag[1] = _mm_add_ps(ti, vi);
        real[3] = _mm_sub_ps(tr, vr);
        imag[3] = _mm_sub_ps(vi, ti);
        real[2] = _mm_mul_ps(real[2], _mm_set1_ps(2.0));
        imag[2] = _mm_mul_ps(imag[2], _mm_set1_ps(-2.0));

        let scale = _mm_set1_ps(std::f32::consts::FRAC_1_SQRT_2 / 8.0);
        for (idx, (r, i)) in real.iter_mut().zip(&mut *imag).enumerate().skip(1) {
            let cos = _mm_load1_ps(&cos_sin_table_4n[idx]);
            let sin = _mm_set1_ps(-cos_sin_table_4n[idx + 8]);

            let rcos = _mm_mul_ps(*r, cos);
            let rsin = _mm_mul_ps(*r, sin);
            let icos = _mm_mul_ps(*i, cos);
            let isin = _mm_mul_ps(*i, sin);
            let tr = _mm_add_ps(rcos, isin);
            let ti = _mm_sub_ps(rsin, icos);
            *r = _mm_mul_ps(tr, scale);
            *i = _mm_mul_ps(ti, scale);
        }
        let one_eighth = _mm_set1_ps(0.125);
        real[0] = _mm_mul_ps(real[0], one_eighth);
        imag[0] = _mm_mul_ps(imag[0], one_eighth);
        imag[1..].reverse();
    }
}

/// Assumes that inputs are reordered.
unsafe fn fft_in_place(real: &mut [Lane], imag: &mut [Lane]) {
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
        real[0] = _mm_add_ps(lr, rr);
        imag[0] = _mm_add_ps(li, ri);
        real[1] = _mm_sub_ps(lr, rr);
        imag[1] = _mm_sub_ps(li, ri);
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
                let cos = _mm_load1_ps(&cos_sin_table[j * k_iter]);
                let sin = _mm_load1_ps(&cos_sin_table[j * k_iter + n / 4]);

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
unsafe fn small_fft_in_place<const N: usize>(real: &mut [Lane], imag: &mut [Lane]) {
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
        real[0] = _mm_add_ps(lr, rr);
        imag[0] = _mm_add_ps(li, ri);
        real[1] = _mm_sub_ps(lr, rr);
        imag[1] = _mm_sub_ps(li, ri);
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
                let cos = _mm_set1_ps(cos_sin_table[j * k_iter]);
                let sin = _mm_set1_ps(cos_sin_table[j * k_iter + N / 4]);

                let ur = real[k + j];
                let ui = imag[k + j];
                let r = real[k + m / 2 + j];
                let i = imag[k + m / 2 + j];
                // (a + ib) (cos + isin) = (a cos - b sin) + i(b cos + a sin)
                let rcos = _mm_mul_ps(r, cos);
                let rsin = _mm_mul_ps(r, sin);
                let icos = _mm_mul_ps(i, cos);
                let isin = _mm_mul_ps(i, sin);
                let tr = _mm_sub_ps(rcos, isin);
                let ti = _mm_add_ps(icos, rsin);

                real[k + j] = _mm_add_ps(ur, tr);
                imag[k + j] = _mm_add_ps(ui, ti);
                real[k + m / 2 + j] = _mm_sub_ps(ur, tr);
                imag[k + m / 2 + j] = _mm_sub_ps(ui, ti);
            }
        }
    }
}
