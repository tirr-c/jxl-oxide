#![allow(unused_variables, unused_mut, dead_code)]

#[derive(Debug)]
pub struct NoiseParameters {
    lut: [f32; 8],
}

impl<Ctx> jxl_bitstream::Bundle<Ctx> for NoiseParameters {
    type Error = crate::Error;

    fn parse<R: std::io::Read>(bitstream: &mut jxl_bitstream::Bitstream<R>, _: Ctx) -> crate::Result<Self> {
        let mut lut = [0.0f32; 8];
        for slot in &mut lut {
            *slot = bitstream.read_bits(10)? as f32 / (1 << 10) as f32;
        }

        Ok(Self { lut })
    }
}
