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
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            unsafe {
                let mut remainder = inverse_h_i32_avx2(merged);
                inverse_h_i32_base(&mut remainder);
                return;
            }
        }
    }

    inverse_h_i32_base(merged)
}

fn inverse_h_i16(merged: &mut CutGrid<i16>) {
    inverse_h_i16_base(merged)
}

fn inverse_h_i32_base(merged: &mut CutGrid<'_, i32>) {
    let height = merged.height();
    let width = merged.width();
    let width_iters = width / 2;
    for y in 0..height {
        let mut avg = merged.get(0, y);
        let mut left = avg;
        for x2 in 0..width_iters {
            let x = x2 * 2;
            let residu = merged.get(x + 1, y);
            let next_avg = if x + 2 < width {
                merged.get(x + 2, y)
            } else {
                avg
            };
            let diff = residu + tendency_i32(left, avg, next_avg);
            let first = avg + diff / 2;
            *merged.get_mut(x, y) = first;
            *merged.get_mut(x + 1, y) = first - diff;
            avg = next_avg;
            left = first - diff;
        }
    }
}

fn inverse_h_i16_base(merged: &mut CutGrid<'_, i16>) {
    let height = merged.height();
    let width = merged.width();
    let width_iters = width / 2;
    for y in 0..height {
        let mut avg = merged.get(0, y);
        let mut left = avg;
        for x2 in 0..width_iters {
            let x = x2 * 2;
            let residu = merged.get(x + 1, y);
            let next_avg = if x + 2 < width {
                merged.get(x + 2, y)
            } else {
                avg
            };
            let diff = residu + tendency_i16(left, avg, next_avg);
            let first = avg + diff / 2;
            *merged.get_mut(x, y) = first;
            *merged.get_mut(x + 1, y) = first - diff;
            avg = next_avg;
            left = first - diff;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline(never)]
unsafe fn inverse_h_i32_avx2<'g>(merged: &'g mut CutGrid<'_, i32>) -> CutGrid<'g, i32> {
    use std::arch::x86_64::*;

    let height = merged.height();
    let h8 = height / 8;
    let width = merged.width();
    let width_iters = width / 2;

    let offset = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    let offset = _mm256_mullo_epi32(offset, _mm256_set1_epi32(merged.stride() as i32));
    for y8 in 0..h8 {
        let y = y8 * 8;

        let mut avg =
            _mm256_i32gather_epi32::<4>(merged.get_mut(0, y) as *mut _ as *const _, offset);
        let mut left = avg;
        for x2 in 0..width_iters {
            let x = x2 * 2;
            let residu =
                _mm256_i32gather_epi32::<4>(merged.get_mut(x + 1, y) as *mut _ as *const _, offset);
            let next_avg = if x + 2 < width {
                _mm256_i32gather_epi32::<4>(merged.get_mut(x + 2, y) as *mut _ as *const _, offset)
            } else {
                avg
            };
            let diff = _mm256_add_epi32(residu, tendency_i32_avx2(left, avg, next_avg));
            let first = _mm256_add_epi32(
                avg,
                _mm256_srai_epi32::<1>(_mm256_add_epi32(diff, _mm256_srli_epi32::<31>(diff))),
            );
            let second = _mm256_sub_epi32(first, diff);

            *merged.get_mut(x, y) = _mm256_extract_epi32::<0>(first);
            *merged.get_mut(x, y + 1) = _mm256_extract_epi32::<1>(first);
            *merged.get_mut(x, y + 2) = _mm256_extract_epi32::<2>(first);
            *merged.get_mut(x, y + 3) = _mm256_extract_epi32::<3>(first);
            *merged.get_mut(x, y + 4) = _mm256_extract_epi32::<4>(first);
            *merged.get_mut(x, y + 5) = _mm256_extract_epi32::<5>(first);
            *merged.get_mut(x, y + 6) = _mm256_extract_epi32::<6>(first);
            *merged.get_mut(x, y + 7) = _mm256_extract_epi32::<7>(first);
            *merged.get_mut(x + 1, y) = _mm256_extract_epi32::<0>(second);
            *merged.get_mut(x + 1, y + 1) = _mm256_extract_epi32::<1>(second);
            *merged.get_mut(x + 1, y + 2) = _mm256_extract_epi32::<2>(second);
            *merged.get_mut(x + 1, y + 3) = _mm256_extract_epi32::<3>(second);
            *merged.get_mut(x + 1, y + 4) = _mm256_extract_epi32::<4>(second);
            *merged.get_mut(x + 1, y + 5) = _mm256_extract_epi32::<5>(second);
            *merged.get_mut(x + 1, y + 6) = _mm256_extract_epi32::<6>(second);
            *merged.get_mut(x + 1, y + 7) = _mm256_extract_epi32::<7>(second);

            avg = next_avg;
            left = second;
        }
    }

    merged.split_vertical(h8 * 8).1
}

pub fn inverse_v<S: Sample>(merged: &mut CutGrid<'_, S>) {
    if let Some(merged) = S::try_as_i16_cut_grid_mut(merged) {
        inverse_v_i16(merged)
    } else if let Some(merged) = S::try_as_i32_cut_grid_mut(merged) {
        inverse_v_i32(merged)
    }
}

fn inverse_v_i32(merged: &mut CutGrid<i32>) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            unsafe {
                let mut remainder = inverse_v_i32_avx2(merged);
                inverse_v_i32_base(&mut remainder);
                return;
            }
        }
    }

    inverse_v_i32_base(merged)
}

fn inverse_v_i16(merged: &mut CutGrid<i16>) {
    if merged.step() == 1 {
        #[cfg(target_arch = "aarch64")]
        if is_aarch64_feature_detected!("neon") {
            unsafe {
                let mut remainder = inverse_v_i16_fast_neon(merged);
                inverse_v_i16_fast(&mut remainder);
                return;
            }
        }

        inverse_v_i16_fast(merged)
    } else {
        inverse_v_i16_base(merged)
    }
}

fn inverse_v_i32_base(merged: &mut CutGrid<'_, i32>) {
    let width = merged.width();
    let height = merged.height();
    let height_iters = height / 2;
    for y2 in 0..height_iters {
        let y = y2 * 2;
        for x in 0..width {
            let avg = merged.get(x, y);
            let residu = merged.get(x, y + 1);
            let next_avg = if y + 2 < height {
                merged.get(x, y + 2)
            } else {
                avg
            };
            let top = if y > 0 { merged.get(x, y - 1) } else { avg };
            let diff = residu + tendency_i32(top, avg, next_avg);
            let first = avg + diff / 2;
            *merged.get_mut(x, y) = first;
            *merged.get_mut(x, y + 1) = first - diff;
        }
    }
}

fn inverse_v_i16_fast(merged: &mut CutGrid<'_, i16>) {
    assert_eq!(merged.step(), 1);

    let width = merged.width();
    let height = merged.height();
    let height_iters = height / 2;
    for y2 in 0..height_iters {
        let y = y2 * 2;
        for x in 0..width {
            let avg = merged.get(x, y);
            let residu = merged.get(x, y + 1);
            let next_avg = if y + 2 < height {
                merged.get(x, y + 2)
            } else {
                avg
            };
            let top = if y > 0 { merged.get(x, y - 1) } else { avg };
            let diff = residu + tendency_i16(top, avg, next_avg);
            let first = avg + diff / 2;
            *merged.get_mut(x, y) = first;
            *merged.get_mut(x, y + 1) = first - diff;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline(never)]
unsafe fn inverse_v_i16_fast_neon<'g>(merged: &'g mut CutGrid<i16>) -> CutGrid<'g, i16> {
    use std::arch::aarch64::*;

    assert_eq!(merged.step(), 1);

    let width = merged.width();
    let w4 = width / 4;
    let height = merged.height();
    let height_iters = height / 2;
    for y2 in 0..height_iters {
        let y = y2 * 2;

        let avg_row_ptr = merged.get_row_mut(y).as_mut_ptr();
        let residu_row_ptr = merged.get_row_mut(y + 1).as_mut_ptr();
        let next_avg_row_ptr = if y + 2 < height {
            merged.get_row_mut(y + 2).as_mut_ptr()
        } else {
            avg_row_ptr
        };
        let top_row_ptr = if y > 0 {
            merged.get_row_mut(y - 1).as_mut_ptr()
        } else {
            avg_row_ptr
        };

        for x4 in 0..w4 {
            let x = x4 * 4;

            let avg_row_ptr = avg_row_ptr.add(x);
            let residu_row_ptr = residu_row_ptr.add(x);
            let next_avg_row_ptr = next_avg_row_ptr.add(x);
            let top_row_ptr = top_row_ptr.add(x);

            let avg = vld1_s16(avg_row_ptr as *const _);
            let residu = vld1_s16(residu_row_ptr as *const _);
            let next_avg = vld1_s16(next_avg_row_ptr as *const _);
            let top = vld1_s16(top_row_ptr as *const _);

            let diff = vadd_s16(residu, tendency_i16_neon(top, avg, next_avg));
            let first = vadd_s16(
                avg,
                vshr_n_s16::<1>(vadd_s16(
                    diff,
                    vreinterpret_s16_u16(vshr_n_u16::<15>(vreinterpret_u16_s16(diff))),
                )),
            );
            let second = vsub_s16(first, diff);

            vst1_s16(avg_row_ptr, first);
            vst1_s16(residu_row_ptr, second);
        }
    }

    merged.split_horizontal(w4 * 4).1
}

fn inverse_v_i16_base(merged: &mut CutGrid<'_, i16>) {
    let width = merged.width();
    let height = merged.height();
    let height_iters = height / 2;
    for y2 in 0..height_iters {
        let y = y2 * 2;
        for x in 0..width {
            let avg = merged.get(x, y);
            let residu = merged.get(x, y + 1);
            let next_avg = if y + 2 < height {
                merged.get(x, y + 2)
            } else {
                avg
            };
            let top = if y > 0 { merged.get(x, y - 1) } else { avg };
            let diff = residu + tendency_i16(top, avg, next_avg);
            let first = avg + diff / 2;
            *merged.get_mut(x, y) = first;
            *merged.get_mut(x, y + 1) = first - diff;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline(never)]
unsafe fn inverse_v_i32_avx2<'g>(merged: &'g mut CutGrid<'_, i32>) -> CutGrid<'g, i32> {
    use std::arch::x86_64::*;

    let width = merged.width();
    let w8 = width / 8;
    let height = merged.height();
    let height_iters = height / 2;

    let offset = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    let offset = _mm256_mullo_epi32(offset, _mm256_set1_epi32(merged.step() as i32));
    for y2 in 0..height_iters {
        let y = y2 * 2;
        for x8 in 0..w8 {
            let x = x8 * 8;
            let avg =
                _mm256_i32gather_epi32::<4>(merged.get_mut(x, y) as *mut _ as *const _, offset);
            let residu =
                _mm256_i32gather_epi32::<4>(merged.get_mut(x, y + 1) as *mut _ as *const _, offset);
            let next_avg = if y + 2 < height {
                _mm256_i32gather_epi32::<4>(merged.get_mut(x, y + 2) as *mut _ as *const _, offset)
            } else {
                avg
            };
            let top = if y > 0 {
                _mm256_i32gather_epi32::<4>(merged.get_mut(x, y - 1) as *mut _ as *const _, offset)
            } else {
                avg
            };

            let diff = _mm256_add_epi32(residu, tendency_i32_avx2(top, avg, next_avg));
            let first = _mm256_add_epi32(
                avg,
                _mm256_srai_epi32::<1>(_mm256_add_epi32(diff, _mm256_srli_epi32::<31>(diff))),
            );
            let second = _mm256_sub_epi32(first, diff);

            *merged.get_mut(x, y) = _mm256_extract_epi32::<0>(first);
            *merged.get_mut(x + 1, y) = _mm256_extract_epi32::<1>(first);
            *merged.get_mut(x + 2, y) = _mm256_extract_epi32::<2>(first);
            *merged.get_mut(x + 3, y) = _mm256_extract_epi32::<3>(first);
            *merged.get_mut(x + 4, y) = _mm256_extract_epi32::<4>(first);
            *merged.get_mut(x + 5, y) = _mm256_extract_epi32::<5>(first);
            *merged.get_mut(x + 6, y) = _mm256_extract_epi32::<6>(first);
            *merged.get_mut(x + 7, y) = _mm256_extract_epi32::<7>(first);
            *merged.get_mut(x, y + 1) = _mm256_extract_epi32::<0>(second);
            *merged.get_mut(x + 1, y + 1) = _mm256_extract_epi32::<1>(second);
            *merged.get_mut(x + 2, y + 1) = _mm256_extract_epi32::<2>(second);
            *merged.get_mut(x + 3, y + 1) = _mm256_extract_epi32::<3>(second);
            *merged.get_mut(x + 4, y + 1) = _mm256_extract_epi32::<4>(second);
            *merged.get_mut(x + 5, y + 1) = _mm256_extract_epi32::<5>(second);
            *merged.get_mut(x + 6, y + 1) = _mm256_extract_epi32::<6>(second);
            *merged.get_mut(x + 7, y + 1) = _mm256_extract_epi32::<7>(second);
        }
    }

    merged.split_horizontal(w8 * 8).1
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
