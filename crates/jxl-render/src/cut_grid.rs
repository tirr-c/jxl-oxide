use jxl_grid::{CutGrid, Grid, SimpleGrid};
use jxl_modular::ChannelShift;

pub fn make_quant_cut_grid<'g>(
    buf: &'g mut SimpleGrid<f32>,
    left: usize,
    top: usize,
    shift: ChannelShift,
    channel_data: &Grid<i32>,
) -> CutGrid<'g, f32> {
    let left = left >> shift.hshift();
    let top = top >> shift.vshift();
    let width = channel_data.width();
    let height = channel_data.height();
    let stride = buf.width();
    let buf = &mut buf.buf_mut()[top * stride + left..];
    let mut grid = CutGrid::from_buf(buf, width, height, stride);
    if let Some(channel_data) = channel_data.as_simple() {
        let buf = channel_data.buf();
        for y in 0..height {
            let row = grid.get_row_mut(y);
            let quant = &buf[y * width..][..width];
            for (out, &q) in row.iter_mut().zip(quant) {
                *out = q as f32;
            }
        }
    } else {
        for y in 0..height {
            let row = grid.get_row_mut(y);
            for (x, out) in row.iter_mut().enumerate() {
                *out = *channel_data.get(x, y).unwrap() as f32;
            }
        }
    }
    grid
}
