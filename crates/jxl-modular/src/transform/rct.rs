use std::num::Wrapping;

use jxl_grid::CutGrid;

use crate::Sample;

pub fn inverse_rct<S: Sample>(permutation: u32, ty: u32, grids: [&mut CutGrid<S>; 3]) {
    let [a, b, c] = grids;
    if let (Some(a), Some(b), Some(c)) = (
        S::try_as_i16_cut_grid_mut(a),
        S::try_as_i16_cut_grid_mut(b),
        S::try_as_i16_cut_grid_mut(c),
    ) {
        do_i16(permutation, ty, [a, b, c]);
    } else if let (Some(a), Some(b), Some(c)) = (
        S::try_as_i32_cut_grid_mut(a),
        S::try_as_i32_cut_grid_mut(b),
        S::try_as_i32_cut_grid_mut(c),
    ) {
        do_i32(permutation, ty, [a, b, c]);
    }
}

#[inline]
fn do_i16(permutation: u32, ty: u32, grids: [&mut CutGrid<i16>; 3]) {
    let permutation = permutation as usize;

    let [a, b, c] = grids;
    let width = a.width();
    let height = a.height();
    assert_eq!(width, b.width());
    assert_eq!(width, c.width());
    assert_eq!(height, b.height());
    assert_eq!(height, c.height());

    for y in 0..height {
        let a = a.get_row_mut(y);
        let b = b.get_row_mut(y);
        let c = c.get_row_mut(y);
        for ((a, b), c) in a.iter_mut().zip(b).zip(c) {
            let samples = [a, b, c];

            let a = Wrapping(*samples[0]);
            let b = Wrapping(*samples[1]);
            let c = Wrapping(*samples[2]);
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
            *samples[permutation % 3] = d.0;
            *samples[(permutation + 1 + (permutation / 3)) % 3] = e.0;
            *samples[(permutation + 2 - (permutation / 3)) % 3] = f.0;
        }
    }
}

#[inline]
fn do_i32(permutation: u32, ty: u32, grids: [&mut CutGrid<i32>; 3]) {
    let permutation = permutation as usize;

    let [a, b, c] = grids;
    let width = a.width();
    let height = a.height();
    assert_eq!(width, b.width());
    assert_eq!(width, c.width());
    assert_eq!(height, b.height());
    assert_eq!(height, c.height());

    for y in 0..height {
        let a = a.get_row_mut(y);
        let b = b.get_row_mut(y);
        let c = c.get_row_mut(y);
        for ((a, b), c) in a.iter_mut().zip(b).zip(c) {
            let samples = [a, b, c];

            let a = Wrapping(*samples[0]);
            let b = Wrapping(*samples[1]);
            let c = Wrapping(*samples[2]);
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
            *samples[permutation % 3] = d.0;
            *samples[(permutation + 1 + (permutation / 3)) % 3] = e.0;
            *samples[(permutation + 2 - (permutation / 3)) % 3] = f.0;
        }
    }
}
