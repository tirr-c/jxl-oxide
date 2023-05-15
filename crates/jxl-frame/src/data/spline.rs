use std::io::Read;

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
            let spline = QuantSpline::parse(
                bitstream,
                QuantSplineParams::new(start_point, num_pixels, &mut decoder),
            )?;
            splines.push(spline);
        }

        Ok(Self {
            quant_adjust,
            quant_splines: splines,
        })
    }
}

struct QuantSplineParams<'d> {
    start_point: (i32, i32),
    num_pixels: usize,
    decoder: &'d mut Decoder,
}

impl<'d> QuantSplineParams<'d> {
    fn new(start_point: (i32, i32), num_pixels: usize, decoder: &'d mut Decoder) -> Self {
        Self { start_point, num_pixels, decoder }
    }
}

/// Holds delta-endcoded control points coordinates (without starting point) and quantized DCT32 coefficients
#[derive(Debug, Default, Clone)]
pub struct QuantSpline {
    pub quant_points: Vec<(i32, i32)>,
    pub manhattan_distance: u64,
    pub xyb_dct: [[i32; 32]; 3],
    pub sigma_dct: [i32; 32],
}

impl Bundle<QuantSplineParams<'_>> for QuantSpline {
    type Error = crate::Error;

    fn parse<R: Read>(bitstream: &mut Bitstream<R>, params: QuantSplineParams<'_>) -> std::result::Result<Self, Self::Error> {
        let QuantSplineParams { start_point, num_pixels, decoder } = params;

        let num_points = decoder.read_varint(bitstream, 3)? as usize;

        let max_num_points = usize::min(MAX_NUM_CONTROL_POINTS, num_pixels / 2);
        if num_points > max_num_points {
            return Err(crate::Error::TooManySplinePoints(num_points));
        }

        let mut quant_points = Vec::with_capacity(1 + num_points);
        let mut cur_value = start_point;
        let mut cur_delta = (0, 0);
        let mut manhattan_distance = 0u64;
        quant_points.push(cur_value);
        for _ in 0..num_points {
            let delta_x = unpack_signed(decoder.read_varint(bitstream, 4)?);
            let delta_y = unpack_signed(decoder.read_varint(bitstream, 4)?);

            cur_delta.0 += delta_x;
            cur_delta.1 += delta_y;
            manhattan_distance += (cur_delta.0.abs() + cur_delta.1.abs()) as u64;
            cur_value.0 += cur_delta.0;
            cur_value.1 += cur_delta.1;
            quant_points.push(cur_value);
        }

        let mut xyb_dct = [[0; 32]; 3];
        for color_dct in &mut xyb_dct {
            for i in color_dct {
                *i = unpack_signed(decoder.read_varint(bitstream, 5)?);
            }
        }

        let mut sigma_dct = [0; 32];
        for i in &mut sigma_dct {
            *i = unpack_signed(decoder.read_varint(bitstream, 5)?);
        }

        Ok(Self {
            quant_points,
            manhattan_distance,
            xyb_dct,
            sigma_dct,
        })
    }
}
