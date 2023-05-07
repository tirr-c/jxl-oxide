#![allow(clippy::excessive_precision)]
use std::io::Read;

use jxl_bitstream::{
    Bitstream,
    Bundle,
    Error,
    Result,
    define_bundle,
    read_bits,
};

define_bundle! {
    #[derive(Debug)]
    pub struct ColourEncoding {
        all_default: ty(Bool) default(true),
        pub want_icc: ty(Bool) cond(!all_default) default(false),
        pub colour_space: ty(Enum(ColourSpace)) cond(!all_default) default(ColourSpace::Rgb),
        pub white_point: ty(Bundle(WhitePoint)) cond(!all_default && !want_icc && colour_space != ColourSpace::Xyb) default(WhitePoint::D65),
        pub primaries: ty(Bundle(Primaries)) cond(!all_default && !want_icc && colour_space != ColourSpace::Xyb && colour_space != ColourSpace::Grey) default(Primaries::Srgb),
        pub tf: ty(Bundle(TransferFunction)) cond(!all_default && !want_icc),
        pub rendering_intent: ty(Enum(RenderingIntent)) cond(!all_default && !want_icc) default(RenderingIntent::Relative),
    }

    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    pub struct Customxy {
        pub x: ty(U32(u(19), 524288 + u(19), 1048576 + u(20), 2097152 + u(21)); UnpackSigned),
        pub y: ty(U32(u(19), 524288 + u(19), 1048576 + u(20), 2097152 + u(21)); UnpackSigned),
    }

    #[derive(Debug)]
    pub struct ToneMapping {
        all_default: ty(Bool) default(true),
        pub intensity_target: ty(F16) cond(!all_default) default(255.0),
        pub min_nits: ty(F16) cond(!all_default) default(0.0),
        pub relative_to_max_display: ty(Bool) cond(!all_default) default(false),
        pub linear_below: ty(F16) cond(!all_default) default(0.0),
    }
}

impl ColourEncoding {
    /// Returns whether this `ColourEncoding` represents the sRGB colorspace.
    #[inline]
    pub fn is_srgb(&self) -> bool {
        self.is_srgb_gamut() && self.tf == TransferFunction::Srgb
    }

    #[inline]
    pub fn is_srgb_gamut(&self) -> bool {
        self.colour_space == ColourSpace::Rgb &&
            self.white_point == WhitePoint::D65 &&
            self.primaries == Primaries::Srgb
    }

    /// Returns the CICP tag which represents this `ColourEncoding`.
    pub fn cicp(&self) -> Option<[u8; 4]> {
        let primaries_cicp = self.primaries.cicp();
        let tf_cicp = self.tf.cicp();
        if let (Some(primaries), Some(tf)) = (primaries_cicp, tf_cicp) {
            Some([primaries, tf, 0, 1])
        } else {
            None
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u8)]
pub enum ColourSpace {
    Rgb = 0,
    Grey = 1,
    Xyb = 2,
    Unknown = 3,
}

impl TryFrom<u32> for ColourSpace {
    type Error = ();

    fn try_from(value: u32) -> std::result::Result<Self, Self::Error> {
        Ok(match value {
            0 => Self::Rgb,
            1 => Self::Grey,
            2 => Self::Xyb,
            3 => Self::Unknown,
            _ => return Err(()),
        })
    }
}

#[derive(Debug, PartialEq, Eq)]
#[repr(u8)]
enum WhitePointDiscriminator {
    D65 = 1,
    Custom = 2,
    E = 10,
    Dci = 11,
}

impl TryFrom<u32> for WhitePointDiscriminator {
    type Error = ();

    fn try_from(value: u32) -> std::result::Result<Self, Self::Error> {
        Ok(match value {
            1 => Self::D65,
            2 => Self::Custom,
            10 => Self::E,
            11 => Self::Dci,
            _ => return Err(()),
        })
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum WhitePoint {
    #[default]
    D65 = 1,
    Custom(Customxy) = 2,
    E = 10,
    Dci = 11,
}

impl<Ctx> Bundle<Ctx> for WhitePoint {
    type Error = Error;

    fn parse<R: Read>(bitstream: &mut Bitstream<R>, _ctx: Ctx) -> Result<Self> {
        let d = read_bits!(bitstream, Enum(WhitePointDiscriminator))?;
        Ok(match d {
            WhitePointDiscriminator::D65 => Self::D65,
            WhitePointDiscriminator::E => Self::E,
            WhitePointDiscriminator::Dci => Self::Dci,
            WhitePointDiscriminator::Custom => {
                let white = read_bits!(bitstream, Bundle(Customxy))?;
                Self::Custom(white)
            },
        })
    }
}

#[derive(Debug, PartialEq, Eq, Default)]
#[repr(u8)]
enum PrimariesDiscriminator {
    #[default]
    Srgb = 1,
    Custom = 2,
    Bt2100 = 9,
    P3 = 11,
}

impl TryFrom<u32> for PrimariesDiscriminator {
    type Error = ();

    fn try_from(value: u32) -> std::result::Result<Self, Self::Error> {
        Ok(match value {
            1 => Self::Srgb,
            2 => Self::Custom,
            9 => Self::Bt2100,
            11 => Self::P3,
            _ => return Err(()),
        })
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum Primaries {
    #[default]
    Srgb = 1,
    Custom {
        red: Customxy,
        green: Customxy,
        blue: Customxy,
    } = 2,
    Bt2100 = 9,
    P3 = 11,
}

impl<Ctx> Bundle<Ctx> for Primaries {
    type Error = Error;

    fn parse<R: Read>(bitstream: &mut Bitstream<R>, _ctx: Ctx) -> Result<Self> {
        let d = read_bits!(bitstream, Enum(PrimariesDiscriminator))?;
        Ok(match d {
            PrimariesDiscriminator::Srgb => Self::Srgb,
            PrimariesDiscriminator::Bt2100 => Self::Bt2100,
            PrimariesDiscriminator::P3 => Self::P3,
            PrimariesDiscriminator::Custom => {
                let red = read_bits!(bitstream, Bundle(Customxy))?;
                let green = read_bits!(bitstream, Bundle(Customxy))?;
                let blue = read_bits!(bitstream, Bundle(Customxy))?;
                Self::Custom { red, green, blue }
            },
        })
    }
}

impl Primaries {
    pub fn cicp(&self) -> Option<u8> {
        match self {
            Primaries::Srgb => Some(1),
            Primaries::Custom { .. } => None,
            Primaries::Bt2100 => Some(9),
            Primaries::P3 => Some(11),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum RenderingIntent {
    Perceptual = 0,
    #[default]
    Relative = 1,
    Saturation = 2,
    Absolute = 3,
}

impl TryFrom<u32> for RenderingIntent {
    type Error = ();

    fn try_from(value: u32) -> std::result::Result<Self, Self::Error> {
        Ok(match value {
            0 => Self::Perceptual,
            1 => Self::Relative,
            2 => Self::Saturation,
            3 => Self::Absolute,
            _ => return Err(()),
        })
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum TransferFunction {
    Gamma(u32) = 0,
    Bt709 = 1,
    Unknown = 2,
    Linear = 8,
    #[default]
    Srgb = 13,
    Pq = 16,
    Dci = 17,
    Hlg = 18,
}

impl TryFrom<u32> for TransferFunction {
    type Error = ();

    fn try_from(value: u32) -> std::result::Result<Self, Self::Error> {
        Ok(match value {
            1 => Self::Bt709,
            2 => Self::Unknown,
            8 => Self::Linear,
            13 => Self::Srgb,
            16 => Self::Pq,
            17 => Self::Dci,
            18 => Self::Hlg,
            _ => return Err(()),
        })
    }
}

impl<Ctx> Bundle<Ctx> for TransferFunction {
    type Error = Error;

    fn parse<R: Read>(bitstream: &mut Bitstream<R>, _ctx: Ctx) -> Result<Self> {
        let has_gamma = bitstream.read_bool()?;
        if has_gamma {
            let gamma = bitstream.read_bits(24)?;
            Ok(Self::Gamma(gamma))
        } else {
            read_bits!(bitstream, Enum(TransferFunction)).map_err(From::from)
        }
    }
}

impl TransferFunction {
    pub fn cicp(&self) -> Option<u8> {
        match self {
            TransferFunction::Gamma(_) => None,
            TransferFunction::Bt709 => Some(1),
            TransferFunction::Unknown => None,
            TransferFunction::Linear => Some(8),
            TransferFunction::Srgb => Some(13),
            TransferFunction::Pq => Some(16),
            TransferFunction::Dci => Some(17),
            TransferFunction::Hlg => Some(18),
        }
    }
}

define_bundle! {
    #[derive(Debug)]
    pub struct OpsinInverseMatrix {
        all_default: ty(Bool) default(true),
        pub inv_mat: ty(Array[Array[F16]; 3]; 3) cond(!all_default) default([
            [11.031566901960783, -9.866943921568629, -0.16462299647058826],
            [-3.254147380392157, 4.418770392156863, -0.16462299647058826],
            [-3.6588512862745097, 2.7129230470588235, 1.9459282392156863],
        ]),
        pub opsin_bias: ty(Array[F16]; 3) cond(!all_default) default([-0.0037930732552754493; 3]),
        pub quant_bias: ty(Array[F16]; 3) cond(!all_default) default([
            1.0 - 0.05465007330715401,
            1.0 - 0.07005449891748593,
            1.0 - 0.049935103337343655,
        ]),
        pub quant_bias_numerator: ty(F16) cond(!all_default) default(0.145),
    }
}
