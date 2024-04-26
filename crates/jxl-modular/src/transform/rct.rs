#[cfg(target_arch = "aarch64")]
use std::arch::is_aarch64_feature_detected;
#[cfg(target_arch = "x86_64")]
use std::arch::is_x86_feature_detected;
use std::num::Wrapping;

use jxl_grid::CutGrid;
use jxl_threadpool::JxlThreadPool;

use crate::Sample;

type FnRow<S> = fn(&mut [&mut [S]; 3]);
type UnsafeFnRow<S> = unsafe fn(&mut [&mut [S]; 3]);

pub fn inverse_rct<S: Sample, const TYPE: u32>(
    permutation: u32,
    mut grids: [&mut CutGrid<S>; 3],
    pool: &JxlThreadPool,
) {
    let grid16 = grids.each_mut().map(|g| S::try_as_i16_cut_grid_mut(g));
    if let [Some(a), Some(b), Some(c)] = grid16 {
        let grids = [a, b, c];

        #[cfg(target_arch = "aarch64")]
        if is_aarch64_feature_detected!("neon") {
            unsafe {
                run_rows_unsafe(
                    permutation,
                    grids,
                    inverse_row_i16_aarch64_neon::<TYPE>,
                    pool,
                );
                return;
            }
        }
    }

    run_rows(permutation, grids, inverse_row_base::<S, TYPE>, pool);
}

#[inline]
fn run_rows<S: Sample>(
    permutation: u32,
    grids: [&mut CutGrid<S>; 3],
    f: FnRow<S>,
    pool: &JxlThreadPool,
) {
    // SAFETY: `f` is safe.
    unsafe { run_rows_unsafe(permutation, grids, f, pool) }
}

#[inline(never)]
unsafe fn run_rows_unsafe<S: Sample>(
    permutation: u32,
    grids: [&mut CutGrid<S>; 3],
    f: UnsafeFnRow<S>,
    pool: &JxlThreadPool,
) {
    struct RctJob<'g, S: Sample> {
        grids: [CutGrid<'g, S>; 3],
    }

    let width = grids[0].width();
    let height = grids[0].height();
    assert_eq!(width, grids[1].width());
    assert_eq!(width, grids[2].width());
    assert_eq!(height, grids[1].height());
    assert_eq!(height, grids[2].height());

    let [mut a, mut b, mut c] = grids.map(|g| g.borrow_mut().into_groups(width, 64));
    let mut jobs = Vec::new();
    while let (Some(a), Some(b), Some(c)) = (a.pop(), b.pop(), c.pop()) {
        jobs.push(RctJob { grids: [a, b, c] });
    }

    pool.for_each_vec(jobs, |job| {
        let mut grids = job.grids;
        let height = grids[0].height();
        for y in 0..height {
            let mut rows = grids.each_mut().map(|g| g.get_row_mut(y));
            f(&mut rows);
            inverse_permute(permutation, rows);
        }
    });
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn inverse_row_i16_aarch64_neon<const TYPE: u32>(rows: &mut [&mut [i16]; 3]) {
    use std::arch::aarch64::*;

    if TYPE == 0 {
        return;
    }

    let [mut a, mut b, mut c] = rows.each_mut().map(|r| r.chunks_exact_mut(4));
    for ((ra, rb), rc) in (&mut a).zip(&mut b).zip(&mut c) {
        let a = vld1_s16(ra.as_ptr());
        let b = vld1_s16(rb.as_ptr());
        let c = vld1_s16(rc.as_ptr());
        let mut d = a;
        let mut e = b;
        let mut f = c;
        match TYPE {
            1 => {
                f = vadd_s16(c, a);
            }
            2 => {
                e = vadd_s16(b, a);
            }
            3 => {
                f = vadd_s16(c, a);
                e = vadd_s16(b, a);
            }
            4 => {
                e = vadd_s16(b, vshr_n_s16::<1>(vadd_s16(a, c)));
            }
            5 => {
                f = vadd_s16(c, a);
                e = vadd_s16(b, vshr_n_s16::<1>(vadd_s16(a, f)));
            }
            6 => {
                let tmp = vsub_s16(a, vshr_n_s16::<1>(c));
                e = vadd_s16(c, tmp);
                f = vsub_s16(tmp, vshr_n_s16::<1>(b));
                d = vadd_s16(f, b);
            }
            _ => {}
        }
        vst1_s16(ra.as_mut_ptr(), d);
        vst1_s16(rb.as_mut_ptr(), e);
        vst1_s16(rc.as_mut_ptr(), f);
    }

    let mut rows = [a.into_remainder(), b.into_remainder(), c.into_remainder()];
    inverse_row_base::<i16, TYPE>(&mut rows);
}

#[inline]
fn inverse_row_base<S: Sample, const TYPE: u32>(rows: &mut [&mut [S]; 3]) {
    let width = rows[0].len();

    for x in 0..width {
        let samples = rows.each_mut().map(|r| &mut r[x]);

        let a = Wrapping(samples[0].to_i32());
        let b = Wrapping(samples[1].to_i32());
        let c = Wrapping(samples[2].to_i32());
        let d;
        let e;
        let f;
        if TYPE == 6 {
            let tmp = a - (c >> 1);
            e = c + tmp;
            f = tmp - (b >> 1);
            d = f + b;
        } else {
            d = a;
            f = if TYPE & 1 != 0 { c + a } else { c };
            e = if (TYPE >> 1) == 1 {
                b + a
            } else if (TYPE >> 1) == 2 {
                b + ((a + f) >> 1)
            } else {
                b
            };
        }
        let d = S::from_i32(d.0);
        let e = S::from_i32(e.0);
        let f = S::from_i32(f.0);
        *samples[0] = d;
        *samples[1] = e;
        *samples[2] = f;
    }
}

#[inline(always)]
fn inverse_permute<S: Sample>(permutation: u32, rows: [&mut [S]; 3]) {
    let [a, b, c] = rows;
    match permutation {
        1 => {
            a.swap_with_slice(b);
            a.swap_with_slice(c);
        }
        2 => {
            a.swap_with_slice(b);
            b.swap_with_slice(c);
        }
        3 => {
            b.swap_with_slice(c);
        }
        4 => {
            a.swap_with_slice(b);
        }
        5 => {
            a.swap_with_slice(c);
        }
        _ => {}
    }
}
