#[cfg(target_arch = "aarch64")]
use std::arch::is_aarch64_feature_detected;

use jxl_grid::CutGrid;

use crate::Sample;

pub fn inverse_h<S: Sample>(merged: &mut CutGrid<'_, S>) {
    if let Some(merged) = S::try_as_i16_cut_grid_mut(merged) {
        inverse_h_i16(merged)
    } else if let Some(merged) = S::try_as_i32_cut_grid_mut(merged) {
        inverse_h_i32(merged)
    }
}

fn inverse_h_i32(merged: &mut CutGrid<i32>) {
    inverse_h_i32_base(merged)
}

fn inverse_h_i16(merged: &mut CutGrid<i16>) {
    #[cfg(target_arch = "aarch64")]
    if is_aarch64_feature_detected!("neon") {
        unsafe {
            inverse_h_i16_aarch64_neon(merged);
            return;
        }
    }

    inverse_h_i16_base(merged)
}

fn inverse_h_i32_base(merged: &mut CutGrid<'_, i32>) {
    let height = merged.height();
    let width = merged.width();
    let mut scratch = vec![0i32; width];
    let avg_width = (width + 1) / 2;
    for y in 0..height {
        let row_out = merged.get_row_mut(y);
        scratch.copy_from_slice(row_out);

        let (avg_row, residu_row) = scratch.split_at_mut(avg_width);
        let mut avg = avg_row[0];
        let mut left = avg;
        let mut row_out_it = row_out.chunks_exact_mut(2);
        for (x, pair) in (&mut row_out_it).enumerate() {
            let residu = residu_row[x];
            let next_avg = avg_row.get(x + 1).copied().unwrap_or(avg);
            let diff = residu + tendency_i32(left, avg, next_avg);
            let first = avg + diff / 2;
            pair[0] = first;
            pair[1] = first - diff;
            avg = next_avg;
            left = first - diff;
        }

        if let [v] = row_out_it.into_remainder() {
            *v = avg_row[avg_width - 1];
        }
    }
}

fn inverse_h_i16_base(merged: &mut CutGrid<'_, i16>) {
    let height = merged.height();
    let width = merged.width();
    let mut scratch = vec![0i16; width];
    let avg_width = (width + 1) / 2;
    for y in 0..height {
        let row_out = merged.get_row_mut(y);
        scratch.copy_from_slice(row_out);

        let (avg_row, residu_row) = scratch.split_at_mut(avg_width);
        let mut avg = avg_row[0];
        let mut left = avg;
        let mut row_out_it = row_out.chunks_exact_mut(2);
        for (x, pair) in (&mut row_out_it).enumerate() {
            let residu = residu_row[x];
            let next_avg = avg_row.get(x + 1).copied().unwrap_or(avg);
            let diff = residu + tendency_i16(left, avg, next_avg);
            let first = avg + diff / 2;
            pair[0] = first;
            pair[1] = first - diff;
            avg = next_avg;
            left = first - diff;
        }

        if let [v] = row_out_it.into_remainder() {
            *v = avg_row[avg_width - 1];
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn inverse_h_i16_aarch64_neon(merged: &mut CutGrid<'_, i16>) {
    use std::arch::aarch64::*;
    use std::mem::MaybeUninit;

    #[inline]
    unsafe fn transpose([v0, v1, v2, v3]: [int16x4_t; 4]) -> [int16x4_t; 4] {
        let int16x4x2_t(tr0, tr1) = vtrn_s16(v0, v1);
        let int16x4x2_t(tr2, tr3) = vtrn_s16(v2, v3);
        let int32x2x2_t(o0, o2) = vtrn_s32(vreinterpret_s32_s16(tr0), vreinterpret_s32_s16(tr2));
        let int32x2x2_t(o1, o3) = vtrn_s32(vreinterpret_s32_s16(tr1), vreinterpret_s32_s16(tr3));
        [
            vreinterpret_s16_s32(o0),
            vreinterpret_s16_s32(o1),
            vreinterpret_s16_s32(o2),
            vreinterpret_s16_s32(o3),
        ]
    }

    let height = merged.height();
    let width = merged.width();

    if width <= 8 {
        return inverse_h_i16_base(merged);
    }

    // SAFETY: int16x4_t doesn't need to be dropped.
    let mut scratch = vec![MaybeUninit::<int16x4_t>::uninit(); width];
    let avg_width = (width + 1) / 2;

    let h4 = height / 4;
    for y4 in 0..h4 {
        let y = y4 * 4;

        // SAFETY: Rows are disjoint.
        let rows = [
            merged.get_row_mut(y).as_mut_ptr(),
            merged.get_row_mut(y + 1).as_mut_ptr(),
            merged.get_row_mut(y + 2).as_mut_ptr(),
            merged.get_row_mut(y + 3).as_mut_ptr(),
        ];

        let mut avg = {
            let v = vld1_lane_s16::<0>(rows[0] as *const _, vdup_n_s16(0));
            let v = vld1_lane_s16::<1>(rows[1] as *const _, v);
            let v = vld1_lane_s16::<2>(rows[2] as *const _, v);
            vld1_lane_s16::<3>(rows[3] as *const _, v)
        };
        let mut left = avg;
        for x4 in 0..(avg_width - 1) / 4 {
            let x = x4 * 4 + 1;
            let avgs = transpose([
                vld1_s16(rows[0].add(x) as *const _),
                vld1_s16(rows[1].add(x) as *const _),
                vld1_s16(rows[2].add(x) as *const _),
                vld1_s16(rows[3].add(x) as *const _),
            ]);
            let residuals = transpose([
                vld1_s16(rows[0].add(avg_width - 1 + x) as *const _),
                vld1_s16(rows[1].add(avg_width - 1 + x) as *const _),
                vld1_s16(rows[2].add(avg_width - 1 + x) as *const _),
                vld1_s16(rows[3].add(avg_width - 1 + x) as *const _),
            ]);
            for (dx, (residual, next_avg)) in residuals.into_iter().zip(avgs).enumerate() {
                let diff = vadd_s16(residual, tendency_i16_neon(left, avg, next_avg));
                let diff_2 = vshr_n_s16::<1>(vadd_s16(
                    diff,
                    vreinterpret_s16_u16(vshr_n_u16::<15>(vreinterpret_u16_s16(diff))),
                ));
                let first = vadd_s16(avg, diff_2);
                let second = vsub_s16(first, diff);
                scratch[x4 * 8 + dx * 2].write(first);
                scratch[x4 * 8 + dx * 2 + 1].write(second);
                avg = next_avg;
                left = second;
            }
        }

        // Check if we have more data to process.
        if (avg_width - 1) % 4 != 0 || width % 2 == 0 {
            let mut avgs = transpose([
                vld1_s16(rows[0].add(avg_width - 4) as *const _),
                vld1_s16(rows[1].add(avg_width - 4) as *const _),
                vld1_s16(rows[2].add(avg_width - 4) as *const _),
                vld1_s16(rows[3].add(avg_width - 4) as *const _),
            ]);
            if width % 2 == 0 {
                avgs = [avgs[1], avgs[2], avgs[3], avgs[3]];
            }
            let residuals = transpose([
                vld1_s16(rows[0].add(width - 4) as *const _),
                vld1_s16(rows[1].add(width - 4) as *const _),
                vld1_s16(rows[2].add(width - 4) as *const _),
                vld1_s16(rows[3].add(width - 4) as *const _),
            ]);
            let from = (!(width / 2) + 1) % 4;
            for (dx, (residual, next_avg)) in residuals.into_iter().zip(avgs).enumerate().skip(from)
            {
                let dx = 4 - dx;
                let diff = vadd_s16(residual, tendency_i16_neon(left, avg, next_avg));
                let diff_2 = vshr_n_s16::<1>(vadd_s16(
                    diff,
                    vreinterpret_s16_u16(vshr_n_u16::<15>(vreinterpret_u16_s16(diff))),
                ));
                let first = vadd_s16(avg, diff_2);
                let second = vsub_s16(first, diff);
                scratch[width / 2 * 2 - dx * 2].write(first);
                scratch[width / 2 * 2 - dx * 2 + 1].write(second);
                avg = next_avg;
                left = second;
            }
        }

        if width % 2 == 1 {
            scratch.last_mut().unwrap().write(avg);
        }

        let mut chunks_it = scratch.chunks_exact(4);
        for (x4, chunk) in (&mut chunks_it).enumerate() {
            let x = x4 * 4;
            let v = transpose([
                chunk[0].assume_init_read(),
                chunk[1].assume_init_read(),
                chunk[2].assume_init_read(),
                chunk[3].assume_init_read(),
            ]);
            vst1_s16(rows[0].add(x), v[0]);
            vst1_s16(rows[1].add(x), v[1]);
            vst1_s16(rows[2].add(x), v[2]);
            vst1_s16(rows[3].add(x), v[3]);
        }

        for (dx, v) in chunks_it.remainder().iter().enumerate() {
            let x = width / 4 * 4 + dx;
            let v = v.assume_init_read();
            *rows[0].add(x) = vget_lane_s16::<0>(v);
            *rows[1].add(x) = vget_lane_s16::<1>(v);
            *rows[2].add(x) = vget_lane_s16::<2>(v);
            *rows[3].add(x) = vget_lane_s16::<3>(v);
        }
    }

    if height % 4 != 0 {
        inverse_h_i16_base(&mut merged.split_vertical(h4 * 4).1);
    }
}

pub fn inverse_v<S: Sample>(merged: &mut CutGrid<'_, S>) {
    if let Some(merged) = S::try_as_i16_cut_grid_mut(merged) {
        inverse_v_i16(merged)
    } else if let Some(merged) = S::try_as_i32_cut_grid_mut(merged) {
        inverse_v_i32(merged)
    }
}

fn inverse_v_i32(merged: &mut CutGrid<i32>) {
    inverse_v_i32_base(merged)
}

fn inverse_v_i16(merged: &mut CutGrid<i16>) {
    #[cfg(target_arch = "aarch64")]
    if is_aarch64_feature_detected!("neon") {
        unsafe {
            inverse_v_i16_aarch64_neon(merged);
            return;
        }
    }

    inverse_v_i16_base(merged)
}

fn inverse_v_i32_base(merged: &mut CutGrid<'_, i32>) {
    let width = merged.width();
    let height = merged.height();
    let mut scratch = vec![0i32; height];
    let avg_height = (height + 1) / 2;
    for x in 0..width {
        for (y, v) in scratch.iter_mut().enumerate() {
            *v = merged.get(x, y);
        }

        let (avg_col, residu_col) = scratch.split_at_mut(avg_height);
        let mut avg = avg_col[0];
        let mut top = avg;
        for (y, &residu) in residu_col.iter().enumerate() {
            let next_avg = avg_col.get(y + 1).copied().unwrap_or(avg);
            let diff = residu + tendency_i32(top, avg, next_avg);
            let first = avg + diff / 2;
            *merged.get_mut(x, 2 * y) = first;
            *merged.get_mut(x, 2 * y + 1) = first - diff;
            avg = next_avg;
            top = first - diff;
        }

        if height % 2 == 1 {
            *merged.get_mut(x, height - 1) = avg_col[avg_height - 1];
        }
    }
}

fn inverse_v_i16_base(merged: &mut CutGrid<'_, i16>) {
    let width = merged.width();
    let height = merged.height();
    let mut scratch = vec![0i16; height];
    let avg_height = (height + 1) / 2;
    for x in 0..width {
        for (y, v) in scratch.iter_mut().enumerate() {
            *v = merged.get(x, y);
        }

        let (avg_col, residu_col) = scratch.split_at_mut(avg_height);
        let mut avg = avg_col[0];
        let mut top = avg;
        for (y, &residu) in residu_col.iter().enumerate() {
            let next_avg = avg_col.get(y + 1).copied().unwrap_or(avg);
            let diff = residu + tendency_i16(top, avg, next_avg);
            let first = avg + diff / 2;
            *merged.get_mut(x, 2 * y) = first;
            *merged.get_mut(x, 2 * y + 1) = first - diff;
            avg = next_avg;
            top = first - diff;
        }

        if height % 2 == 1 {
            *merged.get_mut(x, height - 1) = avg_col[avg_height - 1];
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn inverse_v_i16_aarch64_neon(merged: &mut CutGrid<'_, i16>) {
    use std::arch::aarch64::*;
    use std::mem::MaybeUninit;

    let width = merged.width();
    let height = merged.height();

    if height <= 1 {
        return;
    }

    // SAFETY: int16x4_t doesn't need to be dropped.
    let mut scratch = vec![MaybeUninit::<int16x4_t>::uninit(); height];
    let avg_height = (height + 1) / 2;

    let w4 = width / 4;
    for x4 in 0..w4 {
        let x = x4 * 4;

        let mut avg = vld1_s16(merged.get_mut(x, 0) as *mut i16);
        let mut top = avg;
        let mut chunks_it = scratch.chunks_exact_mut(2);
        for (y, pair) in (&mut chunks_it).enumerate() {
            let residual = vld1_s16(merged.get_mut(x, avg_height + y) as *mut i16);
            let next_avg = if y + 1 < avg_height {
                vld1_s16(merged.get_mut(x, y + 1) as *mut i16)
            } else {
                avg
            };

            let diff = vadd_s16(residual, tendency_i16_neon(top, avg, next_avg));
            let diff_2 = vshr_n_s16::<1>(vadd_s16(
                diff,
                vreinterpret_s16_u16(vshr_n_u16::<15>(vreinterpret_u16_s16(diff))),
            ));
            let first = vadd_s16(avg, diff_2);
            let second = vsub_s16(first, diff);
            pair[0].write(first);
            pair[1].write(second);
            avg = next_avg;
            top = second;
        }

        if let [v] = chunks_it.into_remainder() {
            v.write(avg);
        }

        for (y, v) in scratch.iter().enumerate() {
            vst1_s16(merged.get_mut(x, y) as *mut _, v.assume_init_read());
        }
    }

    if width % 4 != 0 {
        inverse_v_i16_base(&mut merged.split_horizontal(w4 * 4).1);
    }
}

fn tendency_i32(a: i32, b: i32, c: i32) -> i32 {
    if a >= b && b >= c {
        let mut x = (4 * a - 3 * c - b + 6) / 12;
        if x - (x & 1) > 2 * (a - b) {
            x = 2 * (a - b) + 1;
        }
        if x + (x & 1) > 2 * (b - c) {
            x = 2 * (b - c);
        }
        x
    } else if a <= b && b <= c {
        let mut x = (4 * a - 3 * c - b - 6) / 12;
        if x + (x & 1) < 2 * (a - b) {
            x = 2 * (a - b) - 1;
        }
        if x - (x & 1) < 2 * (b - c) {
            x = 2 * (b - c);
        }
        x
    } else {
        0
    }
}

fn tendency_i16(a: i16, b: i16, c: i16) -> i16 {
    if a >= b && b >= c {
        let mut x = (4 * a - 3 * c - b + 6) / 12;
        if x - (x & 1) > 2 * (a - b) {
            x = 2 * (a - b) + 1;
        }
        if x + (x & 1) > 2 * (b - c) {
            x = 2 * (b - c);
        }
        x
    } else if a <= b && b <= c {
        let mut x = (4 * a - 3 * c - b - 6) / 12;
        if x + (x & 1) < 2 * (a - b) {
            x = 2 * (a - b) - 1;
        }
        if x - (x & 1) < 2 * (b - c) {
            x = 2 * (b - c);
        }
        x
    } else {
        0
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(dead_code)]
unsafe fn tendency_i32_avx2(
    a: std::arch::x86_64::__m256i,
    b: std::arch::x86_64::__m256i,
    c: std::arch::x86_64::__m256i,
) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;

    let a_b = _mm256_sub_epi32(a, b);
    let b_c = _mm256_sub_epi32(b, c);
    let a_c = _mm256_sub_epi32(a, c);
    let abs_a_b = _mm256_abs_epi32(a_b);
    let abs_b_c = _mm256_abs_epi32(b_c);
    let abs_a_c = _mm256_abs_epi32(a_c);
    let non_monotonic = _mm256_cmpgt_epi32(_mm256_setzero_si256(), _mm256_xor_si256(a_b, b_c));
    let skip = _mm256_andnot_si256(
        _mm256_cmpeq_epi32(a_b, _mm256_setzero_si256()),
        non_monotonic,
    );
    let skip = _mm256_andnot_si256(_mm256_cmpeq_epi32(b_c, _mm256_setzero_si256()), skip);

    let abs_a_b_3_lo =
        _mm256_srli_si256::<4>(_mm256_mul_epi32(abs_a_b, _mm256_set1_epi32(0x55555556)));
    let abs_a_b_3_hi = _mm256_mul_epi32(
        _mm256_srli_si256::<4>(abs_a_b),
        _mm256_set1_epi32(0x55555556),
    );
    let abs_a_b_3 = _mm256_blend_epi32::<0b10101010>(abs_a_b_3_lo, abs_a_b_3_hi);

    let x = _mm256_add_epi32(abs_a_b_3, _mm256_add_epi32(abs_a_c, _mm256_set1_epi32(2)));
    let x = _mm256_srai_epi32::<2>(x);

    let abs_a_b_2_add_x = _mm256_add_epi32(
        _mm256_slli_epi32::<1>(abs_a_b),
        _mm256_and_si256(x, _mm256_set1_epi32(1)),
    );
    let x = _mm256_blendv_epi8(
        x,
        _mm256_add_epi32(_mm256_slli_epi32::<1>(abs_a_b), _mm256_set1_epi32(1)),
        _mm256_cmpgt_epi32(x, abs_a_b_2_add_x),
    );

    let abs_b_c_2 = _mm256_slli_epi32::<1>(abs_b_c);
    let x = _mm256_blendv_epi8(
        x,
        abs_b_c_2,
        _mm256_cmpgt_epi32(
            _mm256_add_epi32(x, _mm256_and_si256(x, _mm256_set1_epi32(1))),
            abs_b_c_2,
        ),
    );

    let need_neg = _mm256_cmpgt_epi32(c, a);
    let mask = _mm256_andnot_si256(
        skip,
        _mm256_or_si256(_mm256_slli_epi32::<1>(need_neg), _mm256_set1_epi32(1)),
    );
    _mm256_sign_epi32(x, mask)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn tendency_i16_neon(
    a: std::arch::aarch64::int16x4_t,
    b: std::arch::aarch64::int16x4_t,
    c: std::arch::aarch64::int16x4_t,
) -> std::arch::aarch64::int16x4_t {
    use std::arch::aarch64::*;

    let a_b = vsub_s16(a, b);
    let b_c = vsub_s16(b, c);
    let a_c = vsub_s16(a, c);
    let abs_a_b = vabs_s16(a_b);
    let abs_b_c = vabs_s16(b_c);
    let abs_a_c = vabs_s16(a_c);
    let monotonic = vcgez_s16(veor_s16(a_b, b_c));
    let no_skip = vorr_u16(monotonic, vceqz_s16(a_b));
    let no_skip = vorr_u16(no_skip, vceqz_s16(b_c));
    let no_skip = vreinterpret_s16_u16(no_skip);

    let abs_a_b_3_merged = vreinterpretq_s16_s32(vmull_n_s16(abs_a_b, 0x5556));
    let abs_a_b_3 = vuzp2_s16(
        vget_low_s16(abs_a_b_3_merged),
        vget_high_s16(abs_a_b_3_merged),
    );

    let x = vshr_n_s16::<2>(vadd_s16(abs_a_b_3, vadd_s16(abs_a_c, vdup_n_s16(2))));

    let abs_a_b_2_add_x = vadd_s16(vshl_n_s16::<1>(abs_a_b), vand_s16(x, vdup_n_s16(1)));
    let x = vbsl_s16(
        vcgt_s16(x, abs_a_b_2_add_x),
        vadd_s16(vshl_n_s16::<1>(abs_a_b), vdup_n_s16(1)),
        x,
    );

    let abs_b_c_2 = vshl_n_s16::<1>(abs_b_c);
    let x = vbsl_s16(
        vcgt_s16(vadd_s16(x, vand_s16(x, vdup_n_s16(1))), abs_b_c_2),
        abs_b_c_2,
        x,
    );

    let need_neg = vcltz_s16(a_c);
    let neg_x = vneg_s16(x);
    let x = vbsl_s16(need_neg, neg_x, x);
    vand_s16(no_skip, x)
}
