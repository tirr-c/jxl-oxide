use jxl_bitstream::Bitstream;
use jxl_image::ColourEncoding;

use crate::{Error, Result};

pub fn read_icc<R: std::io::Read>(bitstream: &mut Bitstream<R>) -> Result<Vec<u8>> {
    let enc_size = jxl_bitstream::read_bits!(bitstream, U64)?;
    let mut decoder = jxl_coding::Decoder::parse(bitstream, 41)?;

    let mut encoded_icc = vec![0u8; enc_size as usize];
    let mut b1 = 0u8;
    let mut b2 = 0u8;
    decoder.begin(bitstream).unwrap();
    for (idx, b) in encoded_icc.iter_mut().enumerate() {
        let sym = decoder.read_varint(bitstream, get_icc_ctx(idx, b1, b2))?;
        if sym >= 256 {
            return Err(Error::InvalidIccStream("decoded value out of range"));
        }
        *b = sym as u8;

        b2 = b1;
        b1 = *b;
    }

    Ok(encoded_icc)
}

fn get_icc_ctx(idx: usize, b1: u8, b2: u8) -> u32 {
    if idx <= 128 {
        return 0;
    }

    let p1 = match b1 {
        | b'a'..=b'z'
        | b'A'..=b'Z' => 0,
        | b'0'..=b'9'
        | b'.'
        | b',' => 1,
        | 0..=1 => 2 + b1 as u32,
        | 2..=15 => 4,
        | 241..=254 => 5,
        | 255 => 6,
        | _ => 7,
    };
    let p2 = match b2 {
        | b'a'..=b'z'
        | b'A'..=b'Z' => 0,
        | b'0'..=b'9'
        | b'.'
        | b',' => 1,
        | 0..=15 => 2,
        | 241..=255 => 3,
        | _ => 4,
    };

    1 + p1 + 8 * p2
}

#[cfg(feature = "icc")]
pub fn colour_encoding_to_icc(colour_encoding: &ColourEncoding) -> Result<Vec<u8>> {
    use jxl_image::{ColourSpace, Primaries, TransferFunction, WhitePoint};
    use lcms2::{CIExyY, ToneCurve};
    use crate::{consts::{illuminant, primaries}, tf};

    if colour_encoding.want_icc {
        return Err(Error::IccProfileEmbedded);
    }

    if colour_encoding.is_srgb() {
        return Ok(lcms2::Profile::new_srgb().icc()?);
    }

    let wp = match &colour_encoding.white_point {
        WhitePoint::D65 => illuminant::D65_LCMS,
        WhitePoint::Custom(xy) => CIExyY { x: xy.x as f64 / 1e6, y: xy.y as f64 / 1e6, Y: 1.0 },
        WhitePoint::E => illuminant::E_LCMS,
        WhitePoint::Dci => illuminant::DCI_LCMS,
    };

    let tf = match colour_encoding.tf {
        TransferFunction::Gamma(g) => ToneCurve::new(1e7 / g as f64),
        TransferFunction::Bt709 => ToneCurve::new_parametric(4, &[20.0 / 9.0, 1.0 / 1.099, 0.099 / 1.099, 1.0 / 4.5, 0.081])?,
        TransferFunction::Unknown => return Err(Error::InvalidEnumColorspace),
        TransferFunction::Linear => ToneCurve::new(1.0),
        TransferFunction::Srgb => ToneCurve::new_parametric(4, &[2.4, 1.0 / 1.055, 0.055 / 1.055, 1.0 / 12.92, 0.04045])?,
        TransferFunction::Pq => {
            let table = tf::pq_table(4096);
            ToneCurve::new_tabulated(&table)
        },
        TransferFunction::Dci => ToneCurve::new(2.6),
        TransferFunction::Hlg => {
            let table = tf::hlg_table(4096);
            ToneCurve::new_tabulated(&table)
        },
    };

    let mut profile = match colour_encoding.colour_space {
        ColourSpace::Rgb => {
            let primaries = match &colour_encoding.primaries {
                Primaries::Srgb => primaries::SRGB_64,
                Primaries::Custom { red, green, blue } => [
                    [red.x as f64 / 1e6, red.y as f64 / 1e6],
                    [green.x as f64 / 1e6, green.y as f64 / 1e6],
                    [blue.x as f64 / 1e6, blue.y as f64 / 1e6],
                ],
                Primaries::Bt2100 => primaries::BT2100_64,
                Primaries::P3 => primaries::P3_64,
            };
            let primaries = util::primaries_to_xyy(primaries, wp);

            lcms2::Profile::new_rgb(&wp, &primaries, &[&tf, &tf, &tf])?
        },
        ColourSpace::Grey => {
            lcms2::Profile::new_gray(&wp, &tf)?
        },
        ColourSpace::Xyb => {
            todo!()
        },
        ColourSpace::Unknown => return Err(Error::InvalidEnumColorspace),
    };

    profile.set_header_rendering_intent(match colour_encoding.rendering_intent {
        jxl_image::RenderingIntent::Perceptual => lcms2::Intent::Perceptual,
        jxl_image::RenderingIntent::Relative => lcms2::Intent::RelativeColorimetric,
        jxl_image::RenderingIntent::Saturation => lcms2::Intent::Saturation,
        jxl_image::RenderingIntent::Absolute => lcms2::Intent::AbsoluteColorimetric,
    });
    Ok(profile.icc()?)
}

#[cfg(feature = "icc")]
mod util {
    use lcms2::CIExyY;

    pub fn primaries_to_xyy(primaries: [[f64; 2]; 3], wp: CIExyY) -> lcms2::CIExyYTRIPLE {
        let primaries = [
            primaries[0][0], primaries[1][0], primaries[2][0],
            primaries[0][1], primaries[1][1], primaries[2][1],
            (1.0 - primaries[0][0] - primaries[0][1]),
            (1.0 - primaries[1][0] - primaries[1][1]),
            (1.0 - primaries[2][0] - primaries[2][1]),
        ];
        let primaries_inv = matinv_64(&primaries);

        let w_xyz = [wp.x / wp.y, 1.0, (1.0 - wp.x) / wp.y - 1.0];
        let mul = matmul3vec_64(&primaries_inv, &w_xyz);

        lcms2::CIExyYTRIPLE {
            Red: CIExyY { x: primaries[0], y: primaries[3], Y: primaries[3] * mul[0] },
            Green: CIExyY { x: primaries[1], y: primaries[4], Y: primaries[4] * mul[1] },
            Blue: CIExyY { x: primaries[2], y: primaries[5], Y: primaries[5] * mul[2] },
        }
    }

    #[inline]
    fn matmul3vec_64(a: &[f64; 9], b: &[f64; 3]) -> [f64; 3] {
        [
            a[0] * b[0] + a[1] * b[1] * a[2] * b[2],
            a[3] * b[0] + a[4] * b[1] * a[5] * b[2],
            a[6] * b[0] + a[7] * b[1] * a[8] * b[2],
        ]
    }

    #[inline]
    fn matinv_64(mat: &[f64; 9]) -> [f64; 9] {
        let det = mat[0] * (mat[4] * mat[8] - mat[5] * mat[7]) +
            mat[1] * (mat[5] * mat[6] - mat[3] * mat[8]) +
            mat[2] * (mat[3] * mat[7] - mat[4] * mat[6]);
        [
            (mat[4] * mat[8] - mat[5] * mat[7]) / det,
            (mat[7] * mat[2] - mat[8] * mat[1]) / det,
            (mat[1] * mat[5] - mat[2] * mat[4]) / det,
            (mat[5] * mat[6] - mat[3] * mat[8]) / det,
            (mat[8] * mat[0] - mat[6] * mat[2]) / det,
            (mat[2] * mat[3] - mat[0] * mat[5]) / det,
            (mat[3] * mat[7] - mat[4] * mat[6]) / det,
            (mat[6] * mat[1] - mat[7] * mat[0]) / det,
            (mat[0] * mat[4] - mat[1] * mat[3]) / det,
        ]
    }
}
