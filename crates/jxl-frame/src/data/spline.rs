#![allow(unused_variables, unused_mut, dead_code)]

use std::io::Read;

use jxl_bitstream::{Bitstream, Bundle};

use crate::Result;

#[derive(Debug)]
pub struct Splines {
    coords: Vec<(i32, i32)>,
    quant_adjust: i32,
}

impl<Ctx> Bundle<Ctx> for Splines {
    type Error = crate::Error;

    fn parse<R: Read>(bitstream: &mut Bitstream<R>, _: Ctx) -> Result<Self> {
        let mut decoder = jxl_coding::Decoder::parse(bitstream, 6)?;
        decoder.begin(bitstream)?;

        let num_splines = decoder.read_varint(bitstream, 2)? + 1;
        let mut last = (0i32, 0i32);
        todo!()
    }
}
