use std::io::prelude::*;
use std::io::Cursor;

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

fn varint(stream: &mut Cursor<&[u8]>) -> Result<u64> {
    let mut value = 0u64;
    let mut shift = 0;
    let mut b = 0;
    while shift < 63 {
        stream
            .read_exact(std::slice::from_mut(&mut b))
            .map_err(|_| Error::InvalidIccStream("stream is too short"))?;
        value |= ((b & 0x7f) as u64) << shift;
        if b & 0x80 == 0 {
            break;
        }
        shift += 7;
    }
    Ok(value)
}

fn predict_header(idx: usize, output_size: u32, header: &[u8]) -> u8 {
    match idx {
        0..=3 => output_size.to_be_bytes()[idx],
        8 => 4,
        12..=23 => b"mntrRGB XYZ "[idx - 12],
        36..=39 => b"acsp"[idx - 36],
        // APPL
        41 | 42 if header[40] == b'A' => b'P',
        43 if header[40] == b'A' => b'L',
        // MSFT
        41 if header[40] == b'M' => b'S',
        42 if header[40] == b'M' => b'F',
        43 if header[40] == b'M' => b'T',
        // SGI_
        42 if header[40] == b'S' && header[41] == b'G' => b'I',
        43 if header[40] == b'S' && header[41] == b'G' => b' ',
        // SUNW
        42 if header[40] == b'S' && header[41] == b'U' => b'N',
        43 if header[40] == b'S' && header[41] == b'U' => b'W',
        70 => 246,
        71 => 214,
        73 => 1,
        78 => 211,
        79 => 45,
        80..=83 => header[4 + idx - 80],
        _ => 0,
    }
}

fn shuffle2(bytes: &[u8]) -> Vec<u8> {
    let len = bytes.len();
    let mut out = Vec::with_capacity(bytes.len());
    let height = len / 2;
    let odd = len % 2;
    for idx in 0..height {
        out.push(bytes[idx]);
        out.push(bytes[idx + height + odd]);
    }
    if odd != 0 {
        out.push(bytes[height]);
    }
    out
}

fn shuffle4(bytes: &[u8]) -> Vec<u8> {
    let len = bytes.len();
    let mut out = Vec::with_capacity(bytes.len());
    let step = len / 4;
    let wide_count = len % 4;
    for idx in 0..step {
        let mut base = idx;
        for _ in 0..wide_count {
            out.push(bytes[base]);
            base += step + 1;
        }
        for _ in wide_count..4 {
            out.push(bytes[base]);
            base += step;
        }
    }
    for idx in 0..wide_count {
        out.push(bytes[(step + 1) * idx - 1]);
    }
    out
}

pub fn decode_icc(stream: &[u8]) -> Result<Vec<u8>> {
    const COMMON_TAGS: [&[u8]; 19] = [
        b"rTRC", b"rXYZ",
        b"cprt", b"wtpt", b"bkpt",
        b"rXYZ", b"gXYZ", b"bXYZ", b"kXYZ",
        b"rTRC", b"gTRC", b"bTRC", b"kTRC",
        b"chad", b"desc", b"chrm", b"dmnd", b"dmdd", b"lumi",
    ];

    const COMMON_DATA: [&[u8]; 8] = [
        b"XYZ ", b"desc", b"text", b"mluc",
        b"para", b"curv", b"sf32", b"gbd ",
    ];

    let mut tmp_cursor = Cursor::new(stream);
    let output_size = varint(&mut tmp_cursor)?;
    let commands_size = varint(&mut tmp_cursor)?;
    let stream_offset = tmp_cursor.position();

    let mut out = Vec::with_capacity(output_size as usize);
    let (commands, data) = stream[stream_offset as usize..].split_at(commands_size as usize);
    let header_size = output_size.min(128) as usize;
    let (header_data, mut data) = data.split_at(header_size);
    let mut commands_stream = Cursor::new(commands);

    // Header
    for (idx, &e) in header_data.iter().enumerate() {
        let p = predict_header(idx, output_size as u32, header_data);
        out.push(p.wrapping_add(e));
    }
    if output_size <= 128 {
        return Ok(out);
    }

    // Tag
    let v = varint(&mut commands_stream)?;
    if let Some(num_tags) = v.checked_sub(1) {
        let num_tags = num_tags as u32;
        out.extend_from_slice(&num_tags.to_be_bytes());

        let mut prev_tagstart = num_tags * 12 + 128;
        let mut prev_tagsize = 0u32;

        loop {
            let mut command = 0u8;
            if commands_stream.read_exact(std::slice::from_mut(&mut command)).is_err() {
                return Ok(out);
            }
            let tagcode = command & 63;
            let tag = match tagcode {
                0 => break,
                1 => {
                    let (tag, next_data) = data.split_at(4);
                    data = next_data;
                    tag
                },
                2..=20 => COMMON_TAGS[(tagcode - 2) as usize],
                _ => return Err(Error::InvalidIccStream("invalid tagcode")),
            };

            let tagstart = if command & 64 == 0 {
                prev_tagstart + prev_tagsize
            } else {
                varint(&mut commands_stream)? as u32
            };
            let tagsize = match tag {
                _ if command & 128 != 0 => varint(&mut commands_stream)? as u32,
                b"rXYZ" | b"gXYZ" | b"bXYZ" | b"kXYZ" | b"wtpt" | b"bkpt" | b"lumi" => 20,
                _ => prev_tagsize,
            };

            prev_tagstart = tagstart;
            prev_tagsize = tagsize;

            out.extend_from_slice(tag);
            out.extend_from_slice(&tagstart.to_be_bytes());
            out.extend_from_slice(&tagsize.to_be_bytes());
            if tagcode == 2 {
                out.extend_from_slice(b"gTRC");
                out.extend_from_slice(&tagstart.to_be_bytes());
                out.extend_from_slice(&tagsize.to_be_bytes());
                out.extend_from_slice(b"bTRC");
                out.extend_from_slice(&tagstart.to_be_bytes());
                out.extend_from_slice(&tagsize.to_be_bytes());
            } else if tagcode == 3 {
                out.extend_from_slice(b"gXYZ");
                out.extend_from_slice(&(tagstart + tagsize).to_be_bytes());
                out.extend_from_slice(&tagsize.to_be_bytes());
                out.extend_from_slice(b"bXYZ");
                out.extend_from_slice(&(tagstart + tagsize * 2).to_be_bytes());
                out.extend_from_slice(&tagsize.to_be_bytes());
            }
        }
    }

    // Main
    let mut command = 0u8;
    while commands_stream.read_exact(std::slice::from_mut(&mut command)).is_ok() {
        match command {
            1 => {
                let num = varint(&mut commands_stream)? as usize;
                let (bytes, next_data) = data.split_at(num);
                data = next_data;
                out.extend_from_slice(bytes);
            },
            2 | 3 => {
                let num = varint(&mut commands_stream)? as usize;
                let (bytes, next_data) = data.split_at(num);
                data = next_data;
                let bytes = if command == 2 {
                    shuffle2(bytes)
                } else {
                    shuffle4(bytes)
                };
                out.extend_from_slice(&bytes);
            },
            4 => {
                let mut flags = 0u8;
                commands_stream.read_exact(std::slice::from_mut(&mut flags))
                    .map_err(|_| Error::InvalidIccStream("stream is too short"))?;
                let width = ((flags & 3) + 1) as usize;
                let order = (flags >> 2) & 3;
                if width == 3 || order == 3 {
                    return Err(Error::InvalidIccStream("width == 3 || order == 3"));
                }

                let stride = if (flags & 16) == 0 {
                    width
                } else {
                    let stride = varint(&mut commands_stream)? as usize;
                    if stride < width {
                        return Err(Error::InvalidIccStream("stride < width"));
                    }
                    stride
                };
                if stride * 4 >= out.len() {
                    return Err(Error::InvalidIccStream("stride * 4 >= out.len()"));
                }

                let num = varint(&mut commands_stream)? as usize;
                if data.len() < num {
                    return Err(Error::InvalidIccStream("stream is too short"));
                }
                let (bytes, next_data) = data.split_at(num);
                data = next_data;
                let shuffled;
                let bytes = match width {
                    1 => bytes,
                    2 => {
                        shuffled = shuffle2(bytes);
                        &shuffled
                    },
                    4 => {
                        shuffled = shuffle4(bytes);
                        &shuffled
                    },
                    _ => unreachable!(),
                };

                for i in (0..num).step_by(width) {
                    let mut prev = [0u32; 3];
                    for (j, p) in prev[..=order as usize].iter_mut().enumerate() {
                        let offset = out.len() - stride * (j + 1);
                        let mut bytes = [0u8; 4];
                        bytes[(4 - width)..].copy_from_slice(&out[offset..][..width]);
                        *p = u32::from_be_bytes(bytes);
                    }
                    let p = match order {
                        0 => prev[0],
                        1 => 2 * prev[0] - prev[1],
                        2 => 3 * prev[0] - 3 * prev[1] + prev[2],
                        _ => unreachable!(),
                    };

                    for j in 0..width.min(num - i) {
                        let val = (bytes[i + j] as u32 + (p >> (8 * (width - 1 - j)))) as u8;
                        out.push(val);
                    }
                }
            },
            10 => {
                if data.len() < 12 {
                    return Err(Error::InvalidIccStream("stream is too short"));
                }
                out.extend_from_slice(&[b'X', b'Y', b'Z', b' ', 0, 0, 0, 0]);
                let (bytes, next_data) = data.split_at(12);
                data = next_data;
                out.extend_from_slice(bytes);
            },
            16..=23 => {
                out.extend_from_slice(COMMON_DATA[command as usize - 16]);
                out.extend_from_slice(&[0, 0, 0, 0]);
            },
            _ => {
                return Err(Error::InvalidIccStream("invalid command"));
            },
        }
    }
    Ok(out)
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
