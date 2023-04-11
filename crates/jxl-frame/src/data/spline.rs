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

}

}

