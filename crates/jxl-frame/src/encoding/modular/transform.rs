use jxl_bitstream::{define_bundle, read_bits, Bitstream, Bundle};

#[derive(Debug)]
pub enum TransformInfo {
    Rct(Rct),
    Palette(Palette),
    Squeeze(Squeeze),
}

impl<Ctx> Bundle<Ctx> for TransformInfo {
    type Error = crate::Error;

    fn parse<R: std::io::Read>(bitstream: &mut Bitstream<R>, _: Ctx) -> crate::Result<Self> {
        let tr = bitstream.read_bits(2)?;
        match tr {
            0 => read_bits!(bitstream, Bundle(Rct)).map(Self::Rct),
            1 => read_bits!(bitstream, Bundle(Palette)).map(Self::Palette),
            2 => read_bits!(bitstream, Bundle(Squeeze)).map(Self::Squeeze),
            value => Err(crate::Error::Bitstream(jxl_bitstream::Error::InvalidEnum {
                name: "TransformId",
                value,
            }))
        }
    }
}

define_bundle! {
    #[derive(Debug)]
    pub struct Rct error(crate::Error) {
        begin_c: ty(U32(u(3), 8 + u(6), 72 + u(10), 1096 + u(13))),
        rct_type: ty(U32(6, u(2), 2 + u(4), 10 + u(6))),
    }

    #[derive(Debug)]
    pub struct Palette error(crate::Error) {
        begin_c: ty(U32(u(3), 8 + u(6), 72 + u(10), 1096 + u(13))),
        num_c: ty(U32(1, 3, 4, 1 + u(13))),
        nb_colours: ty(U32(u(8), 256 + u(10), 1280 + u(12), 5376 + u(16))),
        nb_deltas: ty(U32(0, 1 + u(8), 257 + u(10), 1281 + u(16))),
        d_pred: ty(u(4)),
    }

    #[derive(Debug)]
    pub struct Squeeze error(crate::Error) {
        num_sq: ty(U32(0, 1 + u(4), 9 + u(6), 41 + u(8))),
        sp: ty(Vec[Bundle(SqueezeParams)]; num_sq),
    }

    #[derive(Debug)]
    struct SqueezeParams error(crate::Error) {
        horizontal: ty(Bool),
        in_place: ty(Bool),
        begin_c: ty(U32(u(3), 8 + u(6), 72 + u(10), 1096 + u(13))),
        num_c: ty(U32(1, 2, 3, 4 + u(4))),
    }
}
