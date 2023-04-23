use jxl_grid::SimpleGrid;

pub fn perform_inverse_ycbcr(fb_cbycr: [&mut SimpleGrid<f32>; 3]) {
    let [cb, y, cr] = fb_cbycr;
    let cb = cb.buf_mut();
    let y = y.buf_mut();
    let cr = cr.buf_mut();

    for ((r, g), b) in cb.iter_mut().zip(y).zip(cr) {
        let cb = *r;
        let y = *g + 0.5;
        let cr = *b;

        *r = y + 1.402 * cr;
        *g = y - 0.344016 * cb - 0.714136 * cr;
        *b = y + 1.772 * cb;
    }
}
