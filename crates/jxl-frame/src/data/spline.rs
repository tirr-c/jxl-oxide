#![allow(unused_variables, unused_mut, dead_code, clippy::needless_range_loop)]

use std::io::Read;

use jxl_bitstream::{unpack_signed, Bitstream, Bundle};
use jxl_coding::Decoder;

use crate::Result;

#[derive(Debug)]
pub struct Splines {
    pub splines: Vec<QuantSpline>,
    start_points: Vec<(i32, i32)>,
    quant_adjust: i32,
}

#[derive(Debug, Default)]
pub struct Spline {
    points: Vec<(f32, f32)>,
    xyb_dct: [[f32; 32]; 3],
    sigma_dct: [f32; 32],
}

#[derive(Debug, Default, Clone)]
pub struct QuantSpline {
    points_deltas: Vec<(i32, i32)>,
    xyb_dct: [[i32; 32]; 3],
    sigma_dct: [i32; 32],
}

impl<Ctx> Bundle<Ctx> for Splines {
    type Error = crate::Error;

    fn parse<R: Read>(bitstream: &mut Bitstream<R>, _: Ctx) -> Result<Self> {
        let mut decoder = jxl_coding::Decoder::parse(bitstream, 6)?;
        decoder.begin(bitstream)?;

        let num_splines = (decoder.read_varint(bitstream, 2)? + 1) as usize;

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

        let mut splines = vec![QuantSpline::new(); num_splines];
        for spline in &mut splines {
            spline.decode(&mut decoder, bitstream)?;
        }

        Ok(Self {
            quant_adjust,
            splines,
            start_points,
        })
    }
}

impl QuantSpline {
    fn new() -> Self {
        Self::default()
    }

    fn decode<R: Read>(
        &mut self,
        decoder: &mut Decoder,
        bitstream: &mut Bitstream<R>,
    ) -> Result<()> {
        let num_points = decoder.read_varint(bitstream, 3)? as usize;
        self.points_deltas.resize(num_points, (0, 0));

        for cp in &mut self.points_deltas {
            cp.0 = unpack_signed(decoder.read_varint(bitstream, 4)?);
            cp.1 = unpack_signed(decoder.read_varint(bitstream, 4)?);
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

    fn dequant(
        &self,
        start_point: (i32, i32),
        quant_adjust: i32,
        base_correlation_x: f32,
        base_correlation_b: f32,
    ) -> Spline {
        let mut manhattan_distance = 0;
        let mut points = Vec::with_capacity(self.points_deltas.len() + 1);

        let mut cur_value = start_point;
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
        for i in 0..32 {
            xyb_dct[0][i] += base_correlation_x * xyb_dct[1][i];
            xyb_dct[2][i] += base_correlation_b * xyb_dct[1][i];
        }
        for i in 0..32 {
            sigma_dct[i] = self.sigma_dct[i] as f32 * CHANNEL_WEIGHTS[3] * inverted_qa;
            let weight = (self.sigma_dct[i]).abs() as f32 * (inverted_qa).ceil();
            width_estimate += weight * weight;
        }

        let estimated_area_reached = width_estimate * manhattan_distance as f32;

        Spline {
            points,
            xyb_dct,
            sigma_dct,
        }
    }
}

