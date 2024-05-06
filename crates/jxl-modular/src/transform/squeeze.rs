#[cfg(target_arch = "aarch64")]
use std::arch::is_aarch64_feature_detected;
#[cfg(target_arch = "x86_64")]
use std::arch::is_x86_feature_detected;
use std::num::Wrapping;

use jxl_grid::MutableSubgrid;

use crate::Sample;

pub fn inverse_h<S: Sample>(merged: &mut MutableSubgrid<'_, S>) {
    if let Some(merged) = S::try_as_mutable_subgrid_i16(merged) {
        inverse_h_i16(merged)
    } else if let Some(merged) = S::try_as_mutable_subgrid_i32(merged) {
        inverse_h_i32(merged)
    }
}

fn inverse_h_i32(merged: &mut MutableSubgrid<i32>) {
    inverse_h_i32_base(merged)
}

#[allow(unreachable_code)]
fn inverse_h_i16(merged: &mut MutableSubgrid<i16>) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("sse4.1") && is_x86_feature_detected!("sse2") {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                inverse_h_i16_x86_64_avx2(merged);
                return;
            }
        } else {
            unsafe {
                inverse_h_i16_x86_64_sse41(merged);
                return;
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    if is_aarch64_feature_detected!("neon") {
        unsafe {
            inverse_h_i16_aarch64_neon(merged);
            return;
        }
    }

    #[cfg(all(target_family = "wasm", target_feature = "simd128"))]
    {
        unsafe {
            inverse_h_i16_wasm32_simd128(merged);
            return;
        }
    }

    inverse_h_i16_base(merged)
}

fn inverse_h_i32_base(merged: &mut MutableSubgrid<'_, i32>) {
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
            let diff = residu.wrapping_add(tendency_i32(left, avg, next_avg));
            let first = avg.wrapping_add(diff / 2);
            let second = first.wrapping_sub(diff);
            pair[0] = first;
            pair[1] = second;
            avg = next_avg;
            left = second;
        }

        if let [v] = row_out_it.into_remainder() {
            *v = avg_row[avg_width - 1];
        }
    }
}

#[inline(never)]
fn inverse_h_i16_base(merged: &mut MutableSubgrid<'_, i16>) {
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
            let diff = residu.wrapping_add(tendency_i16(left, avg, next_avg));
            let first = avg.wrapping_add(diff / 2);
            let second = first.wrapping_sub(diff);
            pair[0] = first;
            pair[1] = second;
            avg = next_avg;
            left = second;
        }

        if let [v] = row_out_it.into_remainder() {
            *v = avg_row[avg_width - 1];
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "sse4.1")]
#[target_feature(enable = "sse2")]
#[inline]
unsafe fn transpose_i16x16(vs: [std::arch::x86_64::__m256i; 8]) -> [std::arch::x86_64::__m256i; 8] {
    use std::arch::x86_64::*;

    let vs = [
        _mm256_unpacklo_epi16(vs[0], vs[1]),
        _mm256_unpacklo_epi16(vs[2], vs[3]),
        _mm256_unpacklo_epi16(vs[4], vs[5]),
        _mm256_unpacklo_epi16(vs[6], vs[7]),
        _mm256_unpackhi_epi16(vs[0], vs[1]),
        _mm256_unpackhi_epi16(vs[2], vs[3]),
        _mm256_unpackhi_epi16(vs[4], vs[5]),
        _mm256_unpackhi_epi16(vs[6], vs[7]),
    ];

    let vs = [
        _mm256_unpacklo_epi32(vs[0], vs[1]),
        _mm256_unpacklo_epi32(vs[2], vs[3]),
        _mm256_unpacklo_epi32(vs[4], vs[5]),
        _mm256_unpacklo_epi32(vs[6], vs[7]),
        _mm256_unpackhi_epi32(vs[0], vs[1]),
        _mm256_unpackhi_epi32(vs[2], vs[3]),
        _mm256_unpackhi_epi32(vs[4], vs[5]),
        _mm256_unpackhi_epi32(vs[6], vs[7]),
    ];

    [
        _mm256_unpacklo_epi64(vs[0], vs[1]),
        _mm256_unpackhi_epi64(vs[0], vs[1]),
        _mm256_unpacklo_epi64(vs[4], vs[5]),
        _mm256_unpackhi_epi64(vs[4], vs[5]),
        _mm256_unpacklo_epi64(vs[2], vs[3]),
        _mm256_unpackhi_epi64(vs[2], vs[3]),
        _mm256_unpacklo_epi64(vs[6], vs[7]),
        _mm256_unpackhi_epi64(vs[6], vs[7]),
    ]
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[inline]
unsafe fn transpose_i16x8(vs: [std::arch::x86_64::__m128i; 8]) -> [std::arch::x86_64::__m128i; 8] {
    use std::arch::x86_64::*;

    let vs = [
        _mm_unpacklo_epi16(vs[0], vs[1]),
        _mm_unpacklo_epi16(vs[2], vs[3]),
        _mm_unpacklo_epi16(vs[4], vs[5]),
        _mm_unpacklo_epi16(vs[6], vs[7]),
        _mm_unpackhi_epi16(vs[0], vs[1]),
        _mm_unpackhi_epi16(vs[2], vs[3]),
        _mm_unpackhi_epi16(vs[4], vs[5]),
        _mm_unpackhi_epi16(vs[6], vs[7]),
    ];

    let vs = [
        _mm_unpacklo_epi32(vs[0], vs[1]),
        _mm_unpacklo_epi32(vs[2], vs[3]),
        _mm_unpacklo_epi32(vs[4], vs[5]),
        _mm_unpacklo_epi32(vs[6], vs[7]),
        _mm_unpackhi_epi32(vs[0], vs[1]),
        _mm_unpackhi_epi32(vs[2], vs[3]),
        _mm_unpackhi_epi32(vs[4], vs[5]),
        _mm_unpackhi_epi32(vs[6], vs[7]),
    ];

    [
        _mm_unpacklo_epi64(vs[0], vs[1]),
        _mm_unpackhi_epi64(vs[0], vs[1]),
        _mm_unpacklo_epi64(vs[4], vs[5]),
        _mm_unpackhi_epi64(vs[4], vs[5]),
        _mm_unpacklo_epi64(vs[2], vs[3]),
        _mm_unpackhi_epi64(vs[2], vs[3]),
        _mm_unpacklo_epi64(vs[6], vs[7]),
        _mm_unpackhi_epi64(vs[6], vs[7]),
    ]
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "sse4.1")]
#[target_feature(enable = "sse2")]
unsafe fn inverse_h_i16_x86_64_avx2(merged: &mut MutableSubgrid<'_, i16>) {
    use std::arch::x86_64::*;
    use std::mem::MaybeUninit;

    let height = merged.height();
    let width = merged.width();

    if width <= 32 {
        return inverse_h_i16_base(merged);
    }

    // SAFETY: __m128i doesn't need to be dropped.
    let mut scratch = vec![MaybeUninit::<__m128i>::uninit(); width];
    let avg_width = (width + 1) / 2;

    let h8 = height / 8;
    for y8 in 0..h8 {
        let y = y8 * 8;

        // SAFETY: Rows are disjoint.
        let rows: [_; 8] = std::array::from_fn(|dy| merged.get_row_mut(y + dy).as_mut_ptr());

        let mut avg = _mm_setr_epi16(
            *rows[0x0], *rows[0x1], *rows[0x2], *rows[0x3], *rows[0x4], *rows[0x5], *rows[0x6],
            *rows[0x7],
        );
        let mut left = avg;
        for x16 in 0..(avg_width - 1) / 16 {
            let x = x16 * 16 + 1;
            let avgs = transpose_i16x16(std::array::from_fn(|idx| unsafe {
                _mm256_loadu_si256(rows[idx].add(x) as *const _)
            }));
            let residuals = transpose_i16x16(std::array::from_fn(|idx| unsafe {
                _mm256_loadu_si256(rows[idx].add(avg_width - 1 + x) as *const _)
            }));

            // Lower half
            for (dx, (residual, next_avg)) in residuals.into_iter().zip(avgs).enumerate() {
                let residual = _mm256_extracti128_si256::<0>(residual);
                let next_avg = _mm256_extracti128_si256::<0>(next_avg);
                let diff = _mm_add_epi16(residual, tendency_i16_x86_64_sse41(left, avg, next_avg));
                let diff_2 = _mm_srai_epi16::<1>(_mm_add_epi16(diff, _mm_srli_epi16::<15>(diff)));
                let first = _mm_add_epi16(avg, diff_2);
                let second = _mm_sub_epi16(first, diff);
                scratch[x16 * 32 + dx * 2].write(first);
                scratch[x16 * 32 + dx * 2 + 1].write(second);
                avg = next_avg;
                left = second;
            }

            // Upper half
            for (dx, (residual, next_avg)) in residuals.into_iter().zip(avgs).enumerate() {
                let residual = _mm256_extracti128_si256::<1>(residual);
                let next_avg = _mm256_extracti128_si256::<1>(next_avg);
                let diff = _mm_add_epi16(residual, tendency_i16_x86_64_sse41(left, avg, next_avg));
                let diff_2 = _mm_srai_epi16::<1>(_mm_add_epi16(diff, _mm_srli_epi16::<15>(diff)));
                let first = _mm_add_epi16(avg, diff_2);
                let second = _mm_sub_epi16(first, diff);
                scratch[x16 * 32 + 16 + dx * 2].write(first);
                scratch[x16 * 32 + 16 + dx * 2 + 1].write(second);
                avg = next_avg;
                left = second;
            }
        }

        if (avg_width - 1) % 16 >= 8 {
            let x16 = (avg_width - 1) / 16;
            let x = x16 * 16 + 1;
            let avgs = transpose_i16x8(std::array::from_fn(|idx| unsafe {
                _mm_loadu_si128(rows[idx].add(x) as *const _)
            }));
            let residuals = transpose_i16x8(std::array::from_fn(|idx| unsafe {
                _mm_loadu_si128(rows[idx].add(avg_width - 1 + x) as *const _)
            }));
            for (dx, (residual, next_avg)) in residuals.into_iter().zip(avgs).enumerate() {
                let diff = _mm_add_epi16(residual, tendency_i16_x86_64_sse41(left, avg, next_avg));
                let diff_2 = _mm_srai_epi16::<1>(_mm_add_epi16(diff, _mm_srli_epi16::<15>(diff)));
                let first = _mm_add_epi16(avg, diff_2);
                let second = _mm_sub_epi16(first, diff);
                scratch[x16 * 32 + dx * 2].write(first);
                scratch[x16 * 32 + dx * 2 + 1].write(second);
                avg = next_avg;
                left = second;
            }
        }

        // Check if we have more data to process.
        if (avg_width - 1) % 8 != 0 || width % 2 == 0 {
            let mut avgs = transpose_i16x8(std::array::from_fn(|idx| unsafe {
                _mm_loadu_si128(rows[idx].add(avg_width - 8) as *const _)
            }));
            if width % 2 == 0 {
                avgs = std::array::from_fn(|idx| if idx == 7 { avgs[7] } else { avgs[idx + 1] });
            }
            let residuals = transpose_i16x8(std::array::from_fn(|idx| unsafe {
                _mm_loadu_si128(rows[idx].add(width - 8) as *const _)
            }));
            let from = (!(width / 2) + 1) % 8;
            for (dx, (residual, next_avg)) in residuals.into_iter().zip(avgs).enumerate().skip(from)
            {
                let dx = 8 - dx;
                let diff = _mm_add_epi16(residual, tendency_i16_x86_64_sse41(left, avg, next_avg));
                let diff_2 = _mm_srai_epi16::<1>(_mm_add_epi16(diff, _mm_srli_epi16::<15>(diff)));
                let first = _mm_add_epi16(avg, diff_2);
                let second = _mm_sub_epi16(first, diff);
                scratch[width / 2 * 2 - dx * 2].write(first);
                scratch[width / 2 * 2 - dx * 2 + 1].write(second);
                avg = next_avg;
                left = second;
            }
        }

        if width % 2 == 1 {
            scratch.last_mut().unwrap().write(avg);
        }

        let mut chunks_it = scratch.chunks_exact(8);
        for (x8, chunk) in (&mut chunks_it).enumerate() {
            let x = x8 * 8;
            let v = transpose_i16x8(std::array::from_fn(|idx| unsafe {
                chunk[idx].assume_init_read()
            }));
            for (row, v) in rows.iter().zip(v) {
                _mm_storeu_si128(row.add(x) as *mut _, v);
            }
        }

        for (dx, v) in chunks_it.remainder().iter().enumerate() {
            let x = width / 8 * 8 + dx;
            let v = v.assume_init_read();
            *rows[0x0].add(x) = _mm_extract_epi16::<0x0>(v) as i16;
            *rows[0x1].add(x) = _mm_extract_epi16::<0x1>(v) as i16;
            *rows[0x2].add(x) = _mm_extract_epi16::<0x2>(v) as i16;
            *rows[0x3].add(x) = _mm_extract_epi16::<0x3>(v) as i16;
            *rows[0x4].add(x) = _mm_extract_epi16::<0x4>(v) as i16;
            *rows[0x5].add(x) = _mm_extract_epi16::<0x5>(v) as i16;
            *rows[0x6].add(x) = _mm_extract_epi16::<0x6>(v) as i16;
            *rows[0x7].add(x) = _mm_extract_epi16::<0x7>(v) as i16;
        }
    }

    if height % 8 != 0 {
        inverse_h_i16_base(&mut merged.split_vertical(h8 * 8).1);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[target_feature(enable = "sse4.1")]
unsafe fn inverse_h_i16_x86_64_sse41(merged: &mut MutableSubgrid<'_, i16>) {
    use std::arch::x86_64::*;
    use std::mem::MaybeUninit;

    let height = merged.height();
    let width = merged.width();

    if width <= 16 {
        return inverse_h_i16_base(merged);
    }

    // SAFETY: __m128i doesn't need to be dropped.
    let mut scratch = vec![MaybeUninit::<__m128i>::uninit(); width];
    let avg_width = (width + 1) / 2;

    let h8 = height / 8;
    for y8 in 0..h8 {
        let y = y8 * 8;

        // SAFETY: Rows are disjoint.
        let rows: [_; 8] = std::array::from_fn(|dy| merged.get_row_mut(y + dy).as_mut_ptr());

        let mut avg = _mm_setr_epi16(
            *rows[0x0], *rows[0x1], *rows[0x2], *rows[0x3], *rows[0x4], *rows[0x5], *rows[0x6],
            *rows[0x7],
        );
        let mut left = avg;
        for x8 in 0..(avg_width - 1) / 8 {
            let x = x8 * 8 + 1;
            let avgs = transpose_i16x8(std::array::from_fn(|idx| unsafe {
                _mm_loadu_si128(rows[idx].add(x) as *const _)
            }));
            let residuals = transpose_i16x8(std::array::from_fn(|idx| unsafe {
                _mm_loadu_si128(rows[idx].add(avg_width - 1 + x) as *const _)
            }));
            for (dx, (residual, next_avg)) in residuals.into_iter().zip(avgs).enumerate() {
                let diff = _mm_add_epi16(residual, tendency_i16_x86_64_sse41(left, avg, next_avg));
                let diff_2 = _mm_srai_epi16::<1>(_mm_add_epi16(diff, _mm_srli_epi16::<15>(diff)));
                let first = _mm_add_epi16(avg, diff_2);
                let second = _mm_sub_epi16(first, diff);
                scratch[x8 * 16 + dx * 2].write(first);
                scratch[x8 * 16 + dx * 2 + 1].write(second);
                avg = next_avg;
                left = second;
            }
        }

        // Check if we have more data to process.
        if (avg_width - 1) % 8 != 0 || width % 2 == 0 {
            let mut avgs = transpose_i16x8(std::array::from_fn(|idx| unsafe {
                _mm_loadu_si128(rows[idx].add(avg_width - 8) as *const _)
            }));
            if width % 2 == 0 {
                avgs = std::array::from_fn(|idx| if idx == 7 { avgs[7] } else { avgs[idx + 1] });
            }
            let residuals = transpose_i16x8(std::array::from_fn(|idx| unsafe {
                _mm_loadu_si128(rows[idx].add(width - 8) as *const _)
            }));
            let from = (!(width / 2) + 1) % 8;
            for (dx, (residual, next_avg)) in residuals.into_iter().zip(avgs).enumerate().skip(from)
            {
                let dx = 8 - dx;
                let diff = _mm_add_epi16(residual, tendency_i16_x86_64_sse41(left, avg, next_avg));
                let diff_2 = _mm_srai_epi16::<1>(_mm_add_epi16(diff, _mm_srli_epi16::<15>(diff)));
                let first = _mm_add_epi16(avg, diff_2);
                let second = _mm_sub_epi16(first, diff);
                scratch[width / 2 * 2 - dx * 2].write(first);
                scratch[width / 2 * 2 - dx * 2 + 1].write(second);
                avg = next_avg;
                left = second;
            }
        }

        if width % 2 == 1 {
            scratch.last_mut().unwrap().write(avg);
        }

        let mut chunks_it = scratch.chunks_exact(8);
        for (x8, chunk) in (&mut chunks_it).enumerate() {
            let x = x8 * 8;
            let v = transpose_i16x8(std::array::from_fn(|idx| unsafe {
                chunk[idx].assume_init_read()
            }));
            for (row, v) in rows.iter().zip(v) {
                _mm_storeu_si128(row.add(x) as *mut _, v);
            }
        }

        for (dx, v) in chunks_it.remainder().iter().enumerate() {
            let x = width / 8 * 8 + dx;
            let v = v.assume_init_read();
            *rows[0x0].add(x) = _mm_extract_epi16::<0x0>(v) as i16;
            *rows[0x1].add(x) = _mm_extract_epi16::<0x1>(v) as i16;
            *rows[0x2].add(x) = _mm_extract_epi16::<0x2>(v) as i16;
            *rows[0x3].add(x) = _mm_extract_epi16::<0x3>(v) as i16;
            *rows[0x4].add(x) = _mm_extract_epi16::<0x4>(v) as i16;
            *rows[0x5].add(x) = _mm_extract_epi16::<0x5>(v) as i16;
            *rows[0x6].add(x) = _mm_extract_epi16::<0x6>(v) as i16;
            *rows[0x7].add(x) = _mm_extract_epi16::<0x7>(v) as i16;
        }
    }

    if height % 8 != 0 {
        inverse_h_i16_base(&mut merged.split_vertical(h8 * 8).1);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn inverse_h_i16_aarch64_neon(merged: &mut MutableSubgrid<'_, i16>) {
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

#[cfg(all(target_family = "wasm", target_feature = "simd128"))]
unsafe fn inverse_h_i16_wasm32_simd128(merged: &mut MutableSubgrid<'_, i16>) {
    use std::arch::wasm32::*;
    use std::mem::MaybeUninit;

    #[inline]
    unsafe fn transpose(vs: [v128; 8]) -> [v128; 8] {
        let vs = [
            i16x8_shuffle::<0x0, 0x8, 0x1, 0x9, 0x2, 0xa, 0x3, 0xb>(vs[0], vs[1]),
            i16x8_shuffle::<0x0, 0x8, 0x1, 0x9, 0x2, 0xa, 0x3, 0xb>(vs[2], vs[3]),
            i16x8_shuffle::<0x0, 0x8, 0x1, 0x9, 0x2, 0xa, 0x3, 0xb>(vs[4], vs[5]),
            i16x8_shuffle::<0x0, 0x8, 0x1, 0x9, 0x2, 0xa, 0x3, 0xb>(vs[6], vs[7]),
            i16x8_shuffle::<0x4, 0xc, 0x5, 0xd, 0x6, 0xe, 0x7, 0xf>(vs[0], vs[1]),
            i16x8_shuffle::<0x4, 0xc, 0x5, 0xd, 0x6, 0xe, 0x7, 0xf>(vs[2], vs[3]),
            i16x8_shuffle::<0x4, 0xc, 0x5, 0xd, 0x6, 0xe, 0x7, 0xf>(vs[4], vs[5]),
            i16x8_shuffle::<0x4, 0xc, 0x5, 0xd, 0x6, 0xe, 0x7, 0xf>(vs[6], vs[7]),
        ];

        let vs = [
            i32x4_shuffle::<0, 4, 1, 5>(vs[0], vs[1]),
            i32x4_shuffle::<0, 4, 1, 5>(vs[2], vs[3]),
            i32x4_shuffle::<0, 4, 1, 5>(vs[4], vs[5]),
            i32x4_shuffle::<0, 4, 1, 5>(vs[6], vs[7]),
            i32x4_shuffle::<2, 6, 3, 7>(vs[0], vs[1]),
            i32x4_shuffle::<2, 6, 3, 7>(vs[2], vs[3]),
            i32x4_shuffle::<2, 6, 3, 7>(vs[4], vs[5]),
            i32x4_shuffle::<2, 6, 3, 7>(vs[6], vs[7]),
        ];

        [
            i64x2_shuffle::<0, 2>(vs[0], vs[1]),
            i64x2_shuffle::<1, 3>(vs[0], vs[1]),
            i64x2_shuffle::<0, 2>(vs[4], vs[5]),
            i64x2_shuffle::<1, 3>(vs[4], vs[5]),
            i64x2_shuffle::<0, 2>(vs[2], vs[3]),
            i64x2_shuffle::<1, 3>(vs[2], vs[3]),
            i64x2_shuffle::<0, 2>(vs[6], vs[7]),
            i64x2_shuffle::<1, 3>(vs[6], vs[7]),
        ]
    }

    let height = merged.height();
    let width = merged.width();

    if width <= 8 {
        return inverse_h_i16_base(merged);
    }

    // SAFETY: v128 doesn't need to be dropped.
    let mut scratch = vec![MaybeUninit::<v128>::uninit(); width];
    let avg_width = (width + 1) / 2;

    let h8 = height / 8;
    for y8 in 0..h8 {
        let y = y8 * 8;

        // SAFETY: Rows are disjoint.
        let rows: [_; 8] = std::array::from_fn(|dy| merged.get_row_mut(y + dy).as_mut_ptr());

        let mut avg = i16x8(
            *rows[0], *rows[1], *rows[2], *rows[3], *rows[4], *rows[5], *rows[6], *rows[7],
        );
        let mut left = avg;
        for x8 in 0..(avg_width - 1) / 8 {
            let x = x8 * 8 + 1;
            let avgs = transpose(std::array::from_fn(|idx| unsafe {
                v128_load(rows[idx].add(x) as *const _)
            }));
            let residuals = transpose(std::array::from_fn(|idx| unsafe {
                v128_load(rows[idx].add(avg_width - 1 + x) as *const _)
            }));
            for (dx, (residual, next_avg)) in residuals.into_iter().zip(avgs).enumerate() {
                let diff = i16x8_add(residual, tendency_i16_wasm32_simd128(left, avg, next_avg));
                let diff_2 = i16x8_shr(i16x8_add(diff, u16x8_shr(diff, 15)), 1);
                let first = i16x8_add(avg, diff_2);
                let second = i16x8_sub(first, diff);
                scratch[x8 * 16 + dx * 2].write(first);
                scratch[x8 * 16 + dx * 2 + 1].write(second);
                avg = next_avg;
                left = second;
            }
        }

        // Check if we have more data to process.
        if (avg_width - 1) % 8 != 0 || width % 2 == 0 {
            let mut avgs = transpose(std::array::from_fn(|idx| unsafe {
                v128_load(rows[idx].add(avg_width - 8) as *const _)
            }));
            if width % 2 == 0 {
                avgs = std::array::from_fn(|idx| if idx == 7 { avgs[7] } else { avgs[idx + 1] });
            }
            let residuals = transpose(std::array::from_fn(|idx| unsafe {
                v128_load(rows[idx].add(width - 8) as *const _)
            }));
            let from = (!(width / 2) + 1) % 8;
            for (dx, (residual, next_avg)) in residuals.into_iter().zip(avgs).enumerate().skip(from)
            {
                let dx = 8 - dx;
                let diff = i16x8_add(residual, tendency_i16_wasm32_simd128(left, avg, next_avg));
                let diff_2 = i16x8_shr(i16x8_add(diff, u16x8_shr(diff, 15)), 1);
                let first = i16x8_add(avg, diff_2);
                let second = i16x8_sub(first, diff);
                scratch[width / 2 * 2 - dx * 2].write(first);
                scratch[width / 2 * 2 - dx * 2 + 1].write(second);
                avg = next_avg;
                left = second;
            }
        }

        if width % 2 == 1 {
            scratch.last_mut().unwrap().write(avg);
        }

        let mut chunks_it = scratch.chunks_exact(8);
        for (x8, chunk) in (&mut chunks_it).enumerate() {
            let x = x8 * 8;
            let v = transpose(std::array::from_fn(|idx| unsafe {
                chunk[idx].assume_init_read()
            }));
            for (row, v) in rows.iter().zip(v) {
                v128_store(row.add(x) as *mut _, v);
            }
        }

        for (dx, v) in chunks_it.remainder().iter().enumerate() {
            let x = width / 8 * 8 + dx;
            let v = v.assume_init_read();
            *rows[0x0].add(x) = i16x8_extract_lane::<0x0>(v);
            *rows[0x1].add(x) = i16x8_extract_lane::<0x1>(v);
            *rows[0x2].add(x) = i16x8_extract_lane::<0x2>(v);
            *rows[0x3].add(x) = i16x8_extract_lane::<0x3>(v);
            *rows[0x4].add(x) = i16x8_extract_lane::<0x4>(v);
            *rows[0x5].add(x) = i16x8_extract_lane::<0x5>(v);
            *rows[0x6].add(x) = i16x8_extract_lane::<0x6>(v);
            *rows[0x7].add(x) = i16x8_extract_lane::<0x7>(v);
        }
    }

    if height % 8 != 0 {
        inverse_h_i16_base(&mut merged.split_vertical(h8 * 8).1);
    }
}

pub fn inverse_v<S: Sample>(merged: &mut MutableSubgrid<'_, S>) {
    if let Some(merged) = S::try_as_mutable_subgrid_i16(merged) {
        inverse_v_i16(merged)
    } else if let Some(merged) = S::try_as_mutable_subgrid_i32(merged) {
        inverse_v_i32(merged)
    }
}

fn inverse_v_i32(merged: &mut MutableSubgrid<i32>) {
    inverse_v_i32_base(merged)
}

#[allow(unreachable_code)]
fn inverse_v_i16(merged: &mut MutableSubgrid<i16>) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("sse4.1") && is_x86_feature_detected!("sse2") {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                inverse_v_i16_x86_64_avx2(merged);
                return;
            }
        } else {
            unsafe {
                inverse_v_i16_x86_64_sse41(merged);
                return;
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    if is_aarch64_feature_detected!("neon") {
        unsafe {
            inverse_v_i16_aarch64_neon(merged);
            return;
        }
    }

    #[cfg(all(target_family = "wasm", target_feature = "simd128"))]
    {
        unsafe {
            inverse_v_i16_wasm32_simd128(merged);
            return;
        }
    }

    inverse_v_i16_base(merged)
}

fn inverse_v_i32_base(merged: &mut MutableSubgrid<'_, i32>) {
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
            let diff = residu.wrapping_add(tendency_i32(top, avg, next_avg));
            let first = avg.wrapping_add(diff / 2);
            let second = first.wrapping_sub(diff);
            *merged.get_mut(x, 2 * y) = first;
            *merged.get_mut(x, 2 * y + 1) = second;
            avg = next_avg;
            top = second;
        }

        if height % 2 == 1 {
            *merged.get_mut(x, height - 1) = avg_col[avg_height - 1];
        }
    }
}

#[inline(never)]
fn inverse_v_i16_base(merged: &mut MutableSubgrid<'_, i16>) {
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
            let diff = residu.wrapping_add(tendency_i16(top, avg, next_avg));
            let first = avg.wrapping_add(diff / 2);
            let second = first.wrapping_sub(diff);
            *merged.get_mut(x, 2 * y) = first;
            *merged.get_mut(x, 2 * y + 1) = second;
            avg = next_avg;
            top = second;
        }

        if height % 2 == 1 {
            *merged.get_mut(x, height - 1) = avg_col[avg_height - 1];
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "sse4.1")]
#[target_feature(enable = "sse2")]
unsafe fn inverse_v_i16_x86_64_avx2(merged: &mut MutableSubgrid<'_, i16>) {
    use std::arch::x86_64::*;
    use std::mem::MaybeUninit;

    let width = merged.width();
    let height = merged.height();

    if height <= 1 {
        return;
    }

    // SAFETY: __m256i doesn't need to be dropped.
    let mut scratch = vec![MaybeUninit::<__m256i>::uninit(); height];
    let avg_height = (height + 1) / 2;

    let w16 = width / 16;
    for x16 in 0..w16 {
        let x = x16 * 16;

        let mut avg = _mm256_loadu_si256(merged.get_mut(x, 0) as *mut i16 as *const _);
        let mut top = avg;
        let mut chunks_it = scratch.chunks_exact_mut(2);
        for (y, pair) in (&mut chunks_it).enumerate() {
            let residual =
                _mm256_loadu_si256(merged.get_mut(x, avg_height + y) as *mut _ as *const _);
            let next_avg = if y + 1 < avg_height {
                _mm256_loadu_si256(merged.get_mut(x, y + 1) as *mut _ as *const _)
            } else {
                avg
            };

            let diff = _mm256_add_epi16(residual, tendency_i16_x86_64_avx2(top, avg, next_avg));
            let diff_2 =
                _mm256_srai_epi16::<1>(_mm256_add_epi16(diff, _mm256_srli_epi16::<15>(diff)));
            let first = _mm256_add_epi16(avg, diff_2);
            let second = _mm256_sub_epi16(first, diff);
            pair[0].write(first);
            pair[1].write(second);
            avg = next_avg;
            top = second;
        }

        if let [v] = chunks_it.into_remainder() {
            v.write(avg);
        }

        for (y, v) in scratch.iter().enumerate() {
            _mm256_storeu_si256(
                merged.get_mut(x, y) as *mut i16 as *mut _,
                v.assume_init_read(),
            );
        }
    }

    if width % 16 != 0 {
        inverse_v_i16_x86_64_sse41(&mut merged.split_horizontal(w16 * 16).1);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[target_feature(enable = "sse2")]
unsafe fn inverse_v_i16_x86_64_sse41(merged: &mut MutableSubgrid<'_, i16>) {
    use std::arch::x86_64::*;
    use std::mem::MaybeUninit;

    let width = merged.width();
    let height = merged.height();

    if height <= 1 {
        return;
    }

    // SAFETY: __m128i doesn't need to be dropped.
    let mut scratch = vec![MaybeUninit::<__m128i>::uninit(); height];
    let avg_height = (height + 1) / 2;

    let w8 = width / 8;
    for x8 in 0..w8 {
        let x = x8 * 8;

        let mut avg = _mm_loadu_si128(merged.get_mut(x, 0) as *mut i16 as *const _);
        let mut top = avg;
        let mut chunks_it = scratch.chunks_exact_mut(2);
        for (y, pair) in (&mut chunks_it).enumerate() {
            let residual = _mm_loadu_si128(merged.get_mut(x, avg_height + y) as *mut _ as *const _);
            let next_avg = if y + 1 < avg_height {
                _mm_loadu_si128(merged.get_mut(x, y + 1) as *mut _ as *const _)
            } else {
                avg
            };

            let diff = _mm_add_epi16(residual, tendency_i16_x86_64_sse41(top, avg, next_avg));
            let diff_2 = _mm_srai_epi16::<1>(_mm_add_epi16(diff, _mm_srli_epi16::<15>(diff)));
            let first = _mm_add_epi16(avg, diff_2);
            let second = _mm_sub_epi16(first, diff);
            pair[0].write(first);
            pair[1].write(second);
            avg = next_avg;
            top = second;
        }

        if let [v] = chunks_it.into_remainder() {
            v.write(avg);
        }

        for (y, v) in scratch.iter().enumerate() {
            _mm_storeu_si128(
                merged.get_mut(x, y) as *mut i16 as *mut _,
                v.assume_init_read(),
            );
        }
    }

    if width % 8 != 0 {
        inverse_v_i16_base(&mut merged.split_horizontal(w8 * 8).1);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn inverse_v_i16_aarch64_neon(merged: &mut MutableSubgrid<'_, i16>) {
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

#[cfg(all(target_family = "wasm", target_feature = "simd128"))]
unsafe fn inverse_v_i16_wasm32_simd128(merged: &mut MutableSubgrid<'_, i16>) {
    use std::arch::wasm32::*;
    use std::mem::MaybeUninit;

    let width = merged.width();
    let height = merged.height();

    if height <= 1 {
        return;
    }

    // SAFETY: v128 doesn't need to be dropped.
    let mut scratch = vec![MaybeUninit::<v128>::uninit(); height];
    let avg_height = (height + 1) / 2;

    let w8 = width / 8;
    for x8 in 0..w8 {
        let x = x8 * 8;

        let mut avg = v128_load(merged.get_mut(x, 0) as *mut _ as *const _);
        let mut top = avg;
        let mut chunks_it = scratch.chunks_exact_mut(2);
        for (y, pair) in (&mut chunks_it).enumerate() {
            let residual = v128_load(merged.get_mut(x, avg_height + y) as *mut _ as *const _);
            let next_avg = if y + 1 < avg_height {
                v128_load(merged.get_mut(x, y + 1) as *mut _ as *const _)
            } else {
                avg
            };

            let diff = i16x8_add(residual, tendency_i16_wasm32_simd128(top, avg, next_avg));
            let diff_2 = i16x8_shr(i16x8_add(diff, u16x8_shr(diff, 15)), 1);
            let first = i16x8_add(avg, diff_2);
            let second = i16x8_sub(first, diff);
            pair[0].write(first);
            pair[1].write(second);
            avg = next_avg;
            top = second;
        }

        if let [v] = chunks_it.into_remainder() {
            v.write(avg);
        }

        for (y, v) in scratch.iter().enumerate() {
            v128_store(
                merged.get_mut(x, y) as *mut i16 as *mut _,
                v.assume_init_read(),
            );
        }
    }

    if width % 8 != 0 {
        inverse_v_i16_base(&mut merged.split_horizontal(w8 * 8).1);
    }
}

fn tendency_i32(a: i32, b: i32, c: i32) -> i32 {
    let a = Wrapping(a);
    let b = Wrapping(b);
    let c = Wrapping(c);

    let n1 = Wrapping(1);
    let n2 = Wrapping(2);
    let n3 = Wrapping(3);
    let n4 = Wrapping(4);
    let n6 = Wrapping(6);
    let n12 = Wrapping(12);

    if a >= b && b >= c {
        let mut x = (n4 * a - n3 * c - b + n6) / n12;
        if x - (x & n1) > n2 * (a - b) {
            x = n2 * (a - b) + n1;
        }
        if x + (x & n1) > n2 * (b - c) {
            x = n2 * (b - c);
        }
        x.0
    } else if a <= b && b <= c {
        let mut x = (n4 * a - n3 * c - b - n6) / n12;
        if x + (x & n1) < n2 * (a - b) {
            x = n2 * (a - b) - n1;
        }
        if x - (x & n1) < n2 * (b - c) {
            x = n2 * (b - c);
        }
        x.0
    } else {
        0
    }
}

fn tendency_i16(a: i16, b: i16, c: i16) -> i16 {
    let a = Wrapping(a);
    let b = Wrapping(b);
    let c = Wrapping(c);

    let n1 = Wrapping(1);
    let n2 = Wrapping(2);
    let n3 = Wrapping(3);
    let n4 = Wrapping(4);
    let n6 = Wrapping(6);
    let n12 = Wrapping(12);

    if a >= b && b >= c {
        let mut x = (n4 * a - n3 * c - b + n6) / n12;
        if x - (x & n1) > n2 * (a - b) {
            x = n2 * (a - b) + n1;
        }
        if x + (x & n1) > n2 * (b - c) {
            x = n2 * (b - c);
        }
        x.0
    } else if a <= b && b <= c {
        let mut x = (n4 * a - n3 * c - b - n6) / n12;
        if x + (x & n1) < n2 * (a - b) {
            x = n2 * (a - b) - n1;
        }
        if x - (x & n1) < n2 * (b - c) {
            x = n2 * (b - c);
        }
        x.0
    } else {
        0
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn tendency_i16_x86_64_avx2(
    a: std::arch::x86_64::__m256i,
    b: std::arch::x86_64::__m256i,
    c: std::arch::x86_64::__m256i,
) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;

    let a_b = _mm256_sub_epi16(a, b);
    let b_c = _mm256_sub_epi16(b, c);
    let a_c = _mm256_sub_epi16(a, c);
    let abs_a_b = _mm256_abs_epi16(a_b);
    let abs_b_c = _mm256_abs_epi16(b_c);
    let abs_a_c = _mm256_abs_epi16(a_c);
    let non_monotonic = _mm256_cmpgt_epi16(_mm256_setzero_si256(), _mm256_xor_si256(a_b, b_c));
    let skip = _mm256_andnot_si256(
        _mm256_cmpeq_epi16(a_b, _mm256_setzero_si256()),
        non_monotonic,
    );
    let skip = _mm256_andnot_si256(_mm256_cmpeq_epi16(b_c, _mm256_setzero_si256()), skip);

    let abs_a_b_3 = _mm256_mulhi_epi16(abs_a_b, _mm256_set1_epi16(0x5556));

    let x = _mm256_add_epi16(abs_a_b_3, _mm256_add_epi16(abs_a_c, _mm256_set1_epi16(2)));
    let x = _mm256_srai_epi16::<2>(x);

    let abs_a_b_2_add_x = _mm256_add_epi16(
        _mm256_slli_epi16::<1>(abs_a_b),
        _mm256_and_si256(x, _mm256_set1_epi16(1)),
    );
    let x = _mm256_blendv_epi8(
        x,
        _mm256_add_epi16(_mm256_slli_epi16::<1>(abs_a_b), _mm256_set1_epi16(1)),
        _mm256_cmpgt_epi16(x, abs_a_b_2_add_x),
    );

    let abs_b_c_2 = _mm256_slli_epi16::<1>(abs_b_c);
    let x = _mm256_blendv_epi8(
        x,
        abs_b_c_2,
        _mm256_cmpgt_epi16(
            _mm256_add_epi16(x, _mm256_and_si256(x, _mm256_set1_epi16(1))),
            abs_b_c_2,
        ),
    );

    let need_neg = _mm256_cmpgt_epi16(c, a);
    let mask = _mm256_andnot_si256(
        skip,
        _mm256_or_si256(_mm256_slli_epi16::<1>(need_neg), _mm256_set1_epi16(1)),
    );
    _mm256_sign_epi16(x, mask)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn tendency_i16_x86_64_sse41(
    a: std::arch::x86_64::__m128i,
    b: std::arch::x86_64::__m128i,
    c: std::arch::x86_64::__m128i,
) -> std::arch::x86_64::__m128i {
    use std::arch::x86_64::*;

    let a_b = _mm_sub_epi16(a, b);
    let b_c = _mm_sub_epi16(b, c);
    let a_c = _mm_sub_epi16(a, c);
    let abs_a_b = _mm_abs_epi16(a_b);
    let abs_b_c = _mm_abs_epi16(b_c);
    let abs_a_c = _mm_abs_epi16(a_c);
    let non_monotonic = _mm_cmpgt_epi16(_mm_setzero_si128(), _mm_xor_si128(a_b, b_c));
    let skip = _mm_andnot_si128(_mm_cmpeq_epi16(a_b, _mm_setzero_si128()), non_monotonic);
    let skip = _mm_andnot_si128(_mm_cmpeq_epi16(b_c, _mm_setzero_si128()), skip);

    let abs_a_b_3 = _mm_mulhi_epi16(abs_a_b, _mm_set1_epi16(0x5556));

    let x = _mm_add_epi16(abs_a_b_3, _mm_add_epi16(abs_a_c, _mm_set1_epi16(2)));
    let x = _mm_srai_epi16::<2>(x);

    let abs_a_b_2_add_x = _mm_add_epi16(
        _mm_slli_epi16::<1>(abs_a_b),
        _mm_and_si128(x, _mm_set1_epi16(1)),
    );
    let x = _mm_blendv_epi8(
        x,
        _mm_add_epi16(_mm_slli_epi16::<1>(abs_a_b), _mm_set1_epi16(1)),
        _mm_cmpgt_epi16(x, abs_a_b_2_add_x),
    );

    let abs_b_c_2 = _mm_slli_epi16::<1>(abs_b_c);
    let x = _mm_blendv_epi8(
        x,
        abs_b_c_2,
        _mm_cmpgt_epi16(
            _mm_add_epi16(x, _mm_and_si128(x, _mm_set1_epi16(1))),
            abs_b_c_2,
        ),
    );

    let need_neg = _mm_cmpgt_epi16(c, a);
    let mask = _mm_andnot_si128(
        skip,
        _mm_or_si128(_mm_slli_epi16::<1>(need_neg), _mm_set1_epi16(1)),
    );
    _mm_sign_epi16(x, mask)
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

#[cfg(all(target_family = "wasm", target_feature = "simd128"))]
fn tendency_i16_wasm32_simd128(
    a: std::arch::wasm32::v128,
    b: std::arch::wasm32::v128,
    c: std::arch::wasm32::v128,
) -> std::arch::wasm32::v128 {
    use std::arch::wasm32::*;

    let a_b = i16x8_sub(a, b);
    let b_c = i16x8_sub(b, c);
    let a_c = i16x8_sub(a, c);
    let abs_a_b = i16x8_abs(a_b);
    let abs_b_c = i16x8_abs(b_c);
    let abs_a_c = i16x8_abs(a_c);

    let monotonic = i16x8_ge(v128_xor(a_b, b_c), i16x8_splat(0));
    let no_skip = v128_or(monotonic, i16x8_eq(a_b, i16x8_splat(0)));
    let no_skip = v128_or(no_skip, i16x8_eq(b_c, i16x8_splat(0)));

    let mul_const = i32x4_splat(0x5556);
    let mul_low = i32x4_mul(i32x4_extend_low_i16x8(abs_a_b), mul_const);
    let mul_high = i32x4_mul(i32x4_extend_high_i16x8(abs_a_b), mul_const);
    let abs_a_b_3 = i16x8_shuffle::<0x1, 0x3, 0x5, 0x7, 0x9, 0xb, 0xd, 0xf>(mul_low, mul_high);

    let x = i16x8_shr(i16x8_add(abs_a_b_3, i16x8_add(abs_a_c, i16x8_splat(2))), 2);

    let abs_a_b_2_add_x = i16x8_add(i16x8_shl(abs_a_b, 1), v128_and(x, i16x8_splat(1)));
    let x = v128_bitselect(
        i16x8_add(i16x8_shl(abs_a_b, 1), i16x8_splat(1)),
        x,
        i16x8_gt(x, abs_a_b_2_add_x),
    );

    let abs_b_c_2 = i16x8_shl(abs_b_c, 1);
    let x = v128_bitselect(
        abs_b_c_2,
        x,
        i16x8_gt(i16x8_add(x, v128_and(x, i16x8_splat(1))), abs_b_c_2),
    );

    let need_neg = i16x8_lt(a_c, i16x8_splat(0));
    let neg_x = i16x8_neg(x);
    let x = v128_bitselect(neg_x, x, need_neg);
    v128_and(no_skip, x)
}
