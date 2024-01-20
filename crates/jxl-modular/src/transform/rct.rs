use std::num::Wrapping;

use jxl_grid::CutGrid;

use crate::Sample;

pub fn inverse_rct<S: Sample>(permutation: u32, ty: u32, grids: [&mut CutGrid<S>; 3]) {
    let permutation = permutation as usize;

    let [a, b, c] = grids;
    let width = a.width();
    let height = a.height();
    assert_eq!(width, b.width());
    assert_eq!(width, c.width());
    assert_eq!(height, b.height());
    assert_eq!(height, c.height());

    for y in 0..height {
        for x in 0..width {
            let samples = [a.get_mut(x, y), b.get_mut(x, y), c.get_mut(x, y)];

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
            *samples[permutation % 3] = S::from_i32(d.0);
            *samples[(permutation + 1 + (permutation / 3)) % 3] = S::from_i32(e.0);
            *samples[(permutation + 2 - (permutation / 3)) % 3] = S::from_i32(f.0);
        }
    }
}
