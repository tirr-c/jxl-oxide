#[derive(Debug)]
pub struct NoiseParameters {
    pub lut: [f32; 8],
}

impl<Ctx> jxl_bitstream::Bundle<Ctx> for NoiseParameters {
    type Error = crate::Error;

    fn parse(bitstream: &mut jxl_bitstream::Bitstream, _: Ctx) -> crate::Result<Self> {
        let mut lut = [0.0f32; 8];
        for slot in &mut lut {
            *slot = bitstream.read_bits(10)? as f32 / (1 << 10) as f32;
        }

        Ok(Self { lut })
    }
}
