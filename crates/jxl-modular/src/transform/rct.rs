use std::num::Wrapping;

use jxl_grid::CutGrid;
use jxl_threadpool::JxlThreadPool;

use crate::Sample;

type FnRow<S> = fn(u32, &mut [&mut [S]; 3]);
type UnsafeFnRow<S> = unsafe fn(u32, &mut [&mut [S]; 3]);

#[inline]
pub fn inverse_rct<S: Sample>(
    permutation: u32,
    ty: u32,
    grids: [&mut CutGrid<S>; 3],
    pool: &JxlThreadPool,
) {
    run_rows(permutation, ty, grids, inverse_row_base, pool);
}

#[inline]
fn run_rows<S: Sample>(
    permutation: u32,
    ty: u32,
    grids: [&mut CutGrid<S>; 3],
    f: FnRow<S>,
    pool: &JxlThreadPool,
) {
    // SAFETY: `f` is safe.
    unsafe { run_rows_unsafe(permutation, ty, grids, f, pool) }
}

unsafe fn run_rows_unsafe<S: Sample>(
    permutation: u32,
    ty: u32,
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
            f(ty, &mut rows);
            inverse_permute(permutation, rows);
        }
    });
}

fn inverse_row_base<S: Sample>(ty: u32, rows: &mut [&mut [S]; 3]) {
    let width = rows[0].len();

    for x in 0..width {
        let samples = rows.each_mut().map(|r| &mut r[x]);

        let a = Wrapping(samples[0].to_i32());
        let b = Wrapping(samples[1].to_i32());
        let c = Wrapping(samples[2].to_i32());
        let d;
        let e;
        let f;
        if ty == 6 {
            let tmp = a - (c >> 1);
            e = c + tmp;
            f = tmp - (b >> 1);
            d = f + b;
        } else {
            d = a;
            f = if ty & 1 != 0 { c + a } else { c };
            e = if (ty >> 1) == 1 {
                b + a
            } else if (ty >> 1) == 2 {
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
