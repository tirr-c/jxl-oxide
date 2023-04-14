#![allow(clippy::needless_range_loop)]

use std::{fmt::Display, io::Read};

use jxl_bitstream::{unpack_signed, Bitstream, Bundle};
use jxl_coding::Decoder;

use crate::{FrameHeader, Result};

const MAX_NUM_SPLINES: usize = 1 << 24;
const MAX_NUM_CONTROL_POINTS: usize = 1 << 20;

/// Holds quantized splines
#[derive(Debug)]
pub struct Splines {
    pub quant_splines: Vec<QuantSpline>,
    pub quant_adjust: i32,
}

/// Holds control point coordinates and dequantized DCT32 coefficients of XYB channels, σ parameter of the spline
#[derive(Debug)]
pub struct Spline {
    pub points: Vec<(f32, f32)>,
    pub xyb_dct: [[f32; 32]; 3],
    pub sigma_dct: [f32; 32],
}

/// Holds delta-endcoded control points coordinates (without starting point) and quantized DCT32 coefficients
///
/// Use [`QuantSpline::dequant`] to get normal [Spline]
#[derive(Debug, Default, Clone)]
pub struct QuantSpline {
    start_point: (i32, i32),
    points_deltas: Vec<(i32, i32)>,
    xyb_dct: [[i32; 32]; 3],
    sigma_dct: [i32; 32],
}

impl Bundle<&FrameHeader> for Splines {
    type Error = crate::Error;

    fn parse<R: Read>(bitstream: &mut Bitstream<R>, header: &FrameHeader) -> Result<Self> {
        let mut decoder = jxl_coding::Decoder::parse(bitstream, 6)?;
        decoder.begin(bitstream)?;
        let num_pixels = (header.width * header.height) as usize;

        let num_splines = (decoder.read_varint(bitstream, 2)? + 1) as usize;

        let max_num_splines = usize::min(MAX_NUM_SPLINES, num_pixels / 4);
        if num_splines > max_num_splines {
            return Err(crate::Error::TooManySplines(num_splines));
        }

        let mut start_points = vec![(0i32, 0i32); num_splines];
        for i in 0..num_splines {
            let mut x = decoder.read_varint(bitstream, 1)? as i32;
            let mut y = decoder.read_varint(bitstream, 1)? as i32;
            if i != 0 {
                x = unpack_signed(x as u32) + start_points[i - 1].0;
                y = unpack_signed(y as u32) + start_points[i - 1].1;
            }
            start_points[i].0 = x;
            start_points[i].1 = y;
        }

        let quant_adjust = unpack_signed(decoder.read_varint(bitstream, 0)?);

        let mut splines: Vec<QuantSpline> = Vec::with_capacity(num_splines);
        for start_point in start_points {
            let mut spline = QuantSpline::new(start_point);
            spline.decode(&mut decoder, bitstream, num_pixels)?;
            spline.start_point = start_point;
            splines.push(spline);
        }

        Ok(Self {
            quant_adjust,
            quant_splines: splines,
        })
    }
}

impl QuantSpline {
    fn new(start_point: (i32, i32)) -> Self {
        Self {
            start_point,
            points_deltas: Vec::new(),
            xyb_dct: [[0; 32]; 3],
            sigma_dct: [0; 32],
        }
    }

    fn decode<R: Read>(
        &mut self,
        decoder: &mut Decoder,
        bitstream: &mut Bitstream<R>,
        num_pixels: usize,
    ) -> Result<()> {
        let num_points = decoder.read_varint(bitstream, 3)? as usize;

        let max_num_points = usize::min(MAX_NUM_CONTROL_POINTS, num_pixels / 2);
        if num_points > max_num_points {
            return Err(crate::Error::TooManySplinePoints(num_points));
        }

        self.points_deltas.resize(num_points, (0, 0));

        for delta in &mut self.points_deltas {
            delta.0 = unpack_signed(decoder.read_varint(bitstream, 4)?);
            delta.1 = unpack_signed(decoder.read_varint(bitstream, 4)?);
        }
        for color_dct in &mut self.xyb_dct {
            for i in color_dct {
                *i = unpack_signed(decoder.read_varint(bitstream, 5)?);
            }
        }
        for i in &mut self.sigma_dct {
            *i = unpack_signed(decoder.read_varint(bitstream, 5)?);
        }
        Ok(())
    }

    // TODO check Maximum total_estimated_area_reached
    pub fn dequant(
        &self,
        quant_adjust: i32,
        base_correlations_xb: Option<(f32, f32)>,
        estimated_area: &mut u64,
    ) -> Spline {
        let mut manhattan_distance = 0;
        let mut points = Vec::with_capacity(self.points_deltas.len() + 1);

        let mut cur_value = self.start_point;
        points.push((cur_value.0 as f32, cur_value.1 as f32));
        let mut cur_delta = (0, 0);
        for delta in &self.points_deltas {
            cur_delta.0 += delta.0;
            cur_delta.1 += delta.1;
            manhattan_distance += cur_delta.0.abs() + cur_delta.1.abs();
            cur_value.0 += cur_delta.0;
            cur_value.1 += cur_delta.1;
            points.push((cur_value.0 as f32, cur_value.1 as f32));
        }

        let mut xyb_dct = [[0f32; 32]; 3];
        let mut sigma_dct = [0f32; 32];
        let mut width_estimate = 0f32;

        let quant_adjust = quant_adjust as f32;
        let inverted_qa = if quant_adjust >= 0.0 {
            1.0 / (1.0 + quant_adjust / 8.0)
        } else {
            1.0 - quant_adjust / 8.0
        };

        const CHANNEL_WEIGHTS: [f32; 4] = [0.0042, 0.075, 0.07, 0.3333];
        for chan_idx in 0..2 {
            for i in 0..32 {
                xyb_dct[chan_idx][i] =
                    self.xyb_dct[chan_idx][i] as f32 * CHANNEL_WEIGHTS[chan_idx] * inverted_qa;
            }
        }
        if let Some((corr_x, corr_b)) = base_correlations_xb {
            for i in 0..32 {
                xyb_dct[0][i] += corr_x * xyb_dct[1][i];
                xyb_dct[2][i] += corr_b * xyb_dct[1][i];
            }
        }
        for i in 0..32 {
            sigma_dct[i] = self.sigma_dct[i] as f32 * CHANNEL_WEIGHTS[3] * inverted_qa;
            let weight = (self.sigma_dct[i]).abs() as f32 * (inverted_qa).ceil();
            width_estimate += weight * weight;
        }

        *estimated_area += (width_estimate * manhattan_distance as f32) as u64;

        Spline {
            points,
            xyb_dct,
            sigma_dct,
        }
    }
}

impl Display for Spline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Spline").unwrap();
        for i in self.xyb_dct.iter().chain(&[self.sigma_dct]) {
            for val in i {
                write!(f, "{} ", val).unwrap();
            }
            writeln!(f).unwrap();
        }
        for point in &self.points {
            writeln!(f, "{} {}", point.0 as i32, point.1 as i32).unwrap();
        }
        writeln!(f, "EndSpline").unwrap();
        Ok(())
    }
}
