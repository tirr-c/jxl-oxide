use jxl_frame::{
    data::{continuous_idct, erf, Spline},
    FrameHeader,
};
use jxl_grid::SimpleGrid;

pub fn render_spline(
    frame_header: &FrameHeader,
    base_grid: &mut [SimpleGrid<f32>],
    spline: Spline,
) -> crate::Result<()> {
    tracing::trace!("{}", spline);

    let all_samples = spline.get_samples();
    let arclength = all_samples.len() as f32 - 2.0 + all_samples.last().unwrap().length;
    for (i, arc) in all_samples.iter().enumerate() {
        let arclength_from_start = f32::min(1.0, (i as f32) / arclength);

        let t = 31.0 * arclength_from_start;
        let sigma = continuous_idct(&spline.sigma_dct, t);
        let inv_sigma = 1.0 / sigma;
        let values = [
            continuous_idct(&spline.xyb_dct[0], t) * arc.length,
            continuous_idct(&spline.xyb_dct[1], t) * arc.length,
            continuous_idct(&spline.xyb_dct[2], t) * arc.length,
        ];

        let max_color = f32::max(0.01, values.into_iter().reduce(f32::max).unwrap());
        let max_distance = f32::sqrt(2.0 * (f32::ln(10.0) * 3.0 + max_color)) * sigma.abs();

        let xbegin = i32::max(0, (arc.point.x - max_distance + 0.5).floor() as i32);
        let xend = i32::min(
            (frame_header.width) as i32,
            (arc.point.x + max_distance + 1.5).floor() as i32,
        );
        let ybegin = i32::max(0, (arc.point.y - max_distance + 0.5).floor() as i32);
        let yend = i32::min(
            (frame_header.height) as i32,
            (arc.point.y + max_distance + 1.5).floor() as i32,
        );

        #[allow(clippy::needless_range_loop)]
        for channel in 0..3 {
            let buffer = &mut base_grid[channel];
            for y in ybegin..yend {
                for x in xbegin..xend {
                    let sample = buffer.get_mut(x as usize, y as usize).unwrap();
                    let dx = (x as f32) - arc.point.x;
                    let dy = (y as f32) - arc.point.y;
                    let distance = f32::sqrt(dx * dx + dy * dy);
                    const SQRT_0125: f32 = 0.353_553_38;
                    let factor = erf((0.5 * distance + SQRT_0125) * inv_sigma)
                        - erf((0.5 * distance - SQRT_0125) * inv_sigma);
                    let extra = 0.25 * values[channel] * sigma * factor * factor;
                    *sample += extra;
                }
            }
        }
    }
    Ok(())
}
