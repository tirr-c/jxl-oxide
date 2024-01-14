use std::ops::{Add, Mul, Sub};

use jxl_frame::{
    data::{QuantSpline, Splines},
    FrameHeader,
};

use crate::region::ImageWithRegion;

/// Holds control point coordinates and dequantized DCT32 coefficients of XYB channels, σ parameter of the spline
#[derive(Debug)]
struct Spline {
    points: Vec<Point>,
    xyb_dct: [[f32; 32]; 3],
    sigma_dct: [f32; 32],
}

impl std::fmt::Display for Spline {
    /// Formats the value using the given formatter in jxl_from_tree syntax
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Spline")?;
        for i in self.xyb_dct.iter().chain(&[self.sigma_dct]) {
            for val in i {
                write!(f, "{} ", val)?;
            }
            writeln!(f)?;
        }
        for point in &self.points {
            writeln!(f, "{} {}", point.x as i32, point.y as i32)?;
        }
        writeln!(f, "EndSpline")
    }
}

struct SplineArc {
    point: Point,
    length: f32,
}

impl Spline {
    fn dequant(
        quant_spline: &QuantSpline,
        quant_adjust: i32,
        base_correlations_xb: Option<(f32, f32)>,
    ) -> Self {
        let points: Vec<_> = quant_spline
            .quant_points
            .iter()
            .map(|&(x, y)| Point::new(x as f32, y as f32))
            .collect();

        let mut xyb_dct = [[0f32; 32]; 3];
        let mut sigma_dct = [0f32; 32];

        let quant_adjust = quant_adjust as f32;
        let inverted_qa = if quant_adjust >= 0.0 {
            1.0 / (1.0 + quant_adjust / 8.0)
        } else {
            1.0 - quant_adjust / 8.0
        };

        const CHANNEL_WEIGHTS: [f32; 4] = [0.0042, 0.075, 0.07, 0.3333];
        for chan_idx in 0..3 {
            for i in 0..32 {
                xyb_dct[chan_idx][i] = quant_spline.xyb_dct[chan_idx][i] as f32
                    * CHANNEL_WEIGHTS[chan_idx]
                    * inverted_qa;
            }
        }
        let (corr_x, corr_b) = base_correlations_xb.unwrap_or((0.0, 1.0));
        for i in 0..32 {
            xyb_dct[0][i] += corr_x * xyb_dct[1][i];
            xyb_dct[2][i] += corr_b * xyb_dct[1][i];
        }

        for (sigma_dct, quant_sigma_dct) in sigma_dct.iter_mut().zip(quant_spline.sigma_dct) {
            *sigma_dct = quant_sigma_dct as f32 * CHANNEL_WEIGHTS[3] * inverted_qa;
        }

        Spline {
            points,
            xyb_dct,
            sigma_dct,
        }
    }

    fn get_samples(&self) -> Vec<SplineArc> {
        let upsampled_points = self.get_upsampled_points();

        let mut current = upsampled_points[0];
        let mut next_idx = 0;
        let mut all_samples = vec![SplineArc {
            point: current,
            length: 1f32,
        }];

        while next_idx < upsampled_points.len() {
            let mut prev = current;
            let mut arclength = 0f32;
            loop {
                if next_idx >= upsampled_points.len() {
                    all_samples.push(SplineArc {
                        point: prev,
                        length: arclength,
                    });
                    break;
                }
                let next = upsampled_points[next_idx];
                let arclength_to_next = (next - prev).norm();
                if arclength + arclength_to_next >= 1.0 {
                    current = prev
                        + ((upsampled_points[next_idx] - prev)
                            * ((1.0 - arclength) / arclength_to_next));
                    all_samples.push(SplineArc {
                        point: current,
                        length: 1.0,
                    });
                    break;
                }
                arclength += arclength_to_next;
                prev = next;
                next_idx += 1;
            }
        }
        all_samples
    }

    /// Returns the points for Cetripetal Catmull-Rom spline segments
    fn get_upsampled_points(&self) -> Vec<Point> {
        let s = &self.points;
        if s.len() == 1 {
            return vec![s[0]];
        }

        let mut extended = Vec::with_capacity(s.len() + 2);

        extended.push(s[1].mirror(&s[0]));
        extended.append(&mut s.clone());
        extended.push(s[s.len() - 2].mirror(&s[s.len() - 1]));

        let mut upsampled = Vec::with_capacity(16 * (extended.len() - 3) + 1);

        for i in 0..extended.len() - 3 {
            let mut p: [Point; 4] = Default::default();
            let mut t: [f32; 4] = Default::default();
            let mut a: [Point; 3] = Default::default();
            let mut b: [Point; 2] = Default::default();

            p.clone_from_slice(&extended[i..i + 4]);
            upsampled.push(p[1]);
            t[0] = 0f32;

            for k in 1..4 {
                // knot sequence with α = 0.25
                t[k] = t[k - 1] + (p[k] - p[k - 1]).norm_squared().powf(0.25);
            }

            for step in 1..16 {
                // knot t from t1 to t2
                let knot = t[1] + (step as f32 / 16.0) * (t[2] - t[1]);

                for k in 0..3 {
                    a[k] = p[k] + ((p[k + 1] - p[k]) * ((knot - t[k]) / (t[k + 1] - t[k])));
                }
                for k in 0..2 {
                    b[k] = a[k] + ((a[k + 1] - a[k]) * ((knot - t[k]) / (t[k + 2] - t[k])));
                }

                // C = ((t2 - t) * B1 + (t - t1) * B2) / (t2 - t1)
                upsampled.push(b[0] + ((b[1] - b[0]) * ((knot - t[1]) / (t[2] - t[1]))));
            }
        }
        upsampled.push(s[s.len() - 1]);
        upsampled
    }
}

pub fn render_spline(
    frame_header: &FrameHeader,
    base_grid: &mut ImageWithRegion,
    splines: &Splines,
    base_correlations_xb: Option<(f32, f32)>,
) -> crate::Result<()> {
    let region = base_grid.region();

    for quant_spline in &splines.quant_splines {
        let spline = Spline::dequant(quant_spline, splines.quant_adjust, base_correlations_xb);
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

            for (channel, buffer) in base_grid.buffer_mut()[..3].iter_mut().enumerate() {
                for y in ybegin..yend {
                    let fy = y - region.top;
                    if fy < 0 {
                        continue;
                    }

                    for x in xbegin..xend {
                        let fx = x - region.left;
                        if fx < 0 {
                            continue;
                        }

                        let Some(sample) = buffer.get_mut(fx as usize, fy as usize) else {
                            break;
                        };
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
    }

    Ok(())
}

/// 2D Point in f32 coordinates
#[derive(Debug, Default, Clone, Copy)]
struct Point {
    x: f32,
    y: f32,
}

impl Point {
    fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    fn mirror(&self, center: &Self) -> Self {
        Self {
            x: center.x + center.x - self.x,
            y: center.y + center.y - self.y,
        }
    }

    fn norm_squared(&self) -> f32 {
        self.x * self.x + self.y * self.y
    }

    fn norm(&self) -> f32 {
        f32::sqrt(self.norm_squared())
    }
}

impl Add for Point {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl Sub for Point {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl Mul<f32> for Point {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self::Output {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}

fn continuous_idct(dct: &[f32; 32], t: f32) -> f32 {
    let mut res = dct[0];
    for (i, &dct) in dct.iter().enumerate().skip(1) {
        let theta = (i as f32) * (std::f32::consts::PI / 32.0) * (t + 0.5);
        res += std::f32::consts::SQRT_2 * dct * theta.cos();
    }
    res
}

/// Computes the error function
// L1 error 7e-4.
#[allow(clippy::excessive_precision)]
fn erf(x: f32) -> f32 {
    let ax = x.abs();

    // Compute 1 - 1 / ((((x * a + b) * x + c) * x + d) * x + 1)**4
    let denom1 = ax * 7.77394369e-02 + 2.05260015e-04;
    let denom2 = denom1 * ax + 2.32120216e-01;
    let denom3 = denom2 * ax + 2.77820801e-01;
    let denom4 = denom3 * ax + 1.0;
    let denom5 = denom4 * denom4;
    let inv_denom5 = 1.0 / denom5;
    let result = -inv_denom5 * inv_denom5 + 1.0;

    // Change sign if needed.
    if x < 0.0 {
        -result
    } else {
        result
    }
}
