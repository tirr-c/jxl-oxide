use jxl_grid::SimpleGrid;

pub fn ycbcr_to_rgb(fb_cbycr: [&mut SimpleGrid<f32>; 3]) {
    let [cb, y, cr] = fb_cbycr;
    let cb = cb.buf_mut();
    let y = y.buf_mut();
    let cr = cr.buf_mut();

    for ((r, g), b) in cb.iter_mut().zip(y).zip(cr) {
        let cb = *r;
        let y = *g + 128.0 / 255.0;
        let cr = *b;

        *r = y + 1.402 * cr;
        *g = y + (-0.114 * 1.772 / 0.587) * cb + (-0.299 * 1.402 / 0.587) * cr;
        *b = y + 1.772 * cb;
    }
}
