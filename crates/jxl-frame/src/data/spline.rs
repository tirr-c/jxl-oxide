use jxl_bitstream::{unpack_signed, Bitstream};
use jxl_coding::Decoder;
use jxl_oxide_common::Bundle;

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

    fn parse(bitstream: &mut Bitstream, header: &FrameHeader) -> Result<Self> {
        let mut decoder = jxl_coding::Decoder::parse(bitstream, 6)?;
        decoder.begin(bitstream)?;

        let num_splines = decoder.read_varint(bitstream, 2)? as usize;
        let num_pixels = (header.width * header.height) as usize;
        let max_num_splines = usize::min(MAX_NUM_SPLINES, num_pixels / 4);
        if num_splines >= max_num_splines {
            tracing::error!(num_splines, max_num_splines, "Too many splines");
            return Err(jxl_bitstream::Error::ProfileConformance("too many splines").into());
        }
        let num_splines = num_splines + 1;

        let mut start_points = vec![(0i64, 0i64); num_splines];
        let mut prev_point = (
            decoder.read_varint(bitstream, 1)? as i64,
            decoder.read_varint(bitstream, 1)? as i64,
        );
        start_points[0] = prev_point;
        for next_point in &mut start_points[1..] {
            let x = decoder.read_varint(bitstream, 1)?;
            let y = decoder.read_varint(bitstream, 1)?;
            prev_point.0 += unpack_signed(x) as i64;
            prev_point.1 += unpack_signed(y) as i64;
            *next_point = prev_point;
        }

        let quant_adjust = unpack_signed(decoder.read_varint(bitstream, 0)?);

        let mut splines: Vec<QuantSpline> = Vec::with_capacity(num_splines);
        let mut acc_control_points = 0usize;
        for start_point in start_points {
            let spline = QuantSpline::parse(
                bitstream,
                QuantSplineParams::new(start_point, num_pixels, &mut decoder, acc_control_points),
            )?;

            acc_control_points += spline.quant_points.len();
            splines.push(spline);
        }

        decoder.finalize()?;

        Ok(Self {
            quant_adjust,
            quant_splines: splines,
        })
    }
}

impl Splines {
    pub(crate) fn estimate_area(&self, base_correlation_xb: Option<(f32, f32)>) -> u64 {
        let base_correlation_xb = base_correlation_xb.unwrap_or((0.0, 1.0));
        let corr_x = base_correlation_xb.0.abs().ceil() as u64;
        let corr_b = base_correlation_xb.1.abs().ceil() as u64;
        let quant_adjust = self.quant_adjust;
        let mut total_area = 0u64;

        for quant_spline in &self.quant_splines {
            let log_color = {
                let mut color_xyb = quant_spline.xyb_dct.map(|quant_color_dct| -> u64 {
                    quant_color_dct
                        .into_iter()
                        .map(|q| div_ceil_qa(q.unsigned_abs(), quant_adjust))
                        .sum()
                });

                color_xyb[0] += corr_x * color_xyb[1];
                color_xyb[2] += corr_b * color_xyb[1];
                log2_ceil(1u64 + color_xyb.into_iter().max().unwrap()) as u64
            };

            let mut width_estimate = 0u64;
            for quant_sigma_dct in quant_spline.sigma_dct {
                let quant_sigma_dct = quant_sigma_dct.unsigned_abs();
                let weight = 1 + div_ceil_qa(quant_sigma_dct, quant_adjust);
                width_estimate += weight * weight * log_color;
            }

            total_area += width_estimate * quant_spline.manhattan_distance;
        }

        total_area
    }
}

#[inline]
fn log2_ceil(x: u64) -> u32 {
    x.next_power_of_two().trailing_zeros()
}

#[inline]
fn div_ceil_qa(dividend: u32, quant_adjust: i32) -> u64 {
    let dividend = dividend as u64;
    if quant_adjust >= 0 {
        let quant_adjust = quant_adjust as u64;
        (8 * dividend + 7 + quant_adjust) / (8 + quant_adjust)
    } else {
        let abs_quant_adjust = (-quant_adjust) as u64;
        dividend + (dividend * abs_quant_adjust).div_ceil(8)
    }
}

struct QuantSplineParams<'d> {
    start_point: (i64, i64),
    num_pixels: usize,
    decoder: &'d mut Decoder,
    acc_control_points: usize,
}

impl<'d> QuantSplineParams<'d> {
    fn new(
        start_point: (i64, i64),
        num_pixels: usize,
        decoder: &'d mut Decoder,
        acc_control_points: usize,
    ) -> Self {
        Self {
            start_point,
            num_pixels,
            decoder,
            acc_control_points,
        }
    }
}

/// Holds delta-endcoded control points coordinates (without starting point) and quantized DCT32 coefficients
#[derive(Debug, Default, Clone)]
pub struct QuantSpline {
    pub quant_points: Vec<(i64, i64)>,
    pub manhattan_distance: u64,
    pub xyb_dct: [[i32; 32]; 3],
    pub sigma_dct: [i32; 32],
}

impl Bundle<QuantSplineParams<'_>> for QuantSpline {
    type Error = crate::Error;

    fn parse(
        bitstream: &mut Bitstream,
        params: QuantSplineParams<'_>,
    ) -> std::result::Result<Self, Self::Error> {
        let QuantSplineParams {
            start_point,
            num_pixels,
            decoder,
            acc_control_points,
        } = params;

        let num_points = decoder.read_varint(bitstream, 3)? as usize;
        let acc_num_points = acc_control_points + num_points;
        let max_num_points = usize::min(MAX_NUM_CONTROL_POINTS, num_pixels / 2);
        if acc_num_points > max_num_points {
            tracing::error!(num_points, max_num_points, "Too many spline points");
            return Err(jxl_bitstream::Error::ProfileConformance("too many spline points").into());
        }

        let mut quant_points = Vec::with_capacity(1 + num_points);
        let mut cur_value = start_point;
        let mut cur_delta = (0, 0);
        let mut manhattan_distance = 0u64;
        quant_points.push(cur_value);
        for _ in 0..num_points {
            let prev_value = cur_value;
            let delta_x = unpack_signed(decoder.read_varint(bitstream, 4)?) as i64;
            let delta_y = unpack_signed(decoder.read_varint(bitstream, 4)?) as i64;

            cur_delta.0 += delta_x;
            cur_delta.1 += delta_y;
            manhattan_distance += (cur_delta.0.abs() + cur_delta.1.abs()) as u64;
            cur_value.0 = cur_value.0.checked_add(cur_delta.0).ok_or(
                jxl_bitstream::Error::ValidationFailed("control point overflowed"),
            )?;
            cur_value.1 = cur_value.1.checked_add(cur_delta.1).ok_or(
                jxl_bitstream::Error::ValidationFailed("control point overflowed"),
            )?;
            if cur_value == prev_value {
                return Err(jxl_bitstream::Error::ValidationFailed(
                    "two consecutive control points have the same value",
                )
                .into());
            }
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
