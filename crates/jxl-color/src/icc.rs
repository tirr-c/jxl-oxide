//! Functions related to ICC profiles.
//!
//! - [`read_icc`] and [`decode_icc`] can be used to read embedded ICC profile from the bitstream.
//! - [`colour_encoding_to_icc`] can be used to create an ICC profile to embed into the decoded
//!   image file, or to be used by the color management system for various purposes.

use std::io::prelude::*;
use std::io::Cursor;

use jxl_bitstream::Bitstream;

use crate::{
    ciexyz::*,
    consts::*,
    tf,
    ColourEncoding,
    ColourSpace,
    Error,
    Primaries,
    RenderingIntent,
    Result,
    TransferFunction,
    WhitePoint,
};

/// Reads the encoded ICC profile stream from the given bitstream.
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

/// Decodes the given ICC profile stream.
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
                        1 => (2 * prev[0]).wrapping_sub(prev[1]),
                        2 => (3 * prev[0]).wrapping_sub(3 * prev[1]).wrapping_add(prev[2]),
                        _ => unreachable!(),
                    };

                    for j in 0..width.min(num - i) {
                        let val = (bytes[i + j] as u32).wrapping_add(p >> (8 * (width - 1 - j))) as u8;
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

#[derive(Debug)]
struct IccTag {
    tag: [u8; 4],
    data_offset: u32,
    len: u32,
}

fn append_tag_with_data(tags_out: &mut Vec<IccTag>, data_out: &mut Vec<u8>, tag: [u8; 4], data: &[u8]) {
    append_multiple_tags_with_data(tags_out, data_out, &[tag], data)
}

fn append_multiple_tags_with_data(
    tags_out: &mut Vec<IccTag>,
    data_out: &mut Vec<u8>,
    tags: &[[u8; 4]],
    data: &[u8],
) {
    let data_offset = data_out.len() as u32;
    let len = data.len() as u32;
    for &tag in tags {
        tags_out.push(IccTag {
            tag,
            data_offset,
            len,
        });
    }
    data_out.extend_from_slice(data);
    // Align to 4 bytes
    data_out.resize((data_out.len() + 3) & (!3), 0);
}

fn create_mluc(locale: [u8; 4], strings: &[&str]) -> Vec<u8> {
    let mut out = vec![b'm', b'l', b'u', b'c', 0, 0, 0, 0];
    let mut data = Vec::new();
    out.extend_from_slice(&(strings.len() as u32).to_be_bytes());
    out.extend_from_slice(&[0, 0, 0, 0xc]);
    for s in strings {
        let offset = data.len() as u32;
        data.extend(s.encode_utf16());
        out.extend_from_slice(&locale);
        out.extend_from_slice(&((data.len() as u32 - offset) * 2).to_be_bytes());
        out.extend_from_slice(&(0x14 + strings.len() as u32 * 12 + offset * 2).to_be_bytes());
    }
    for c in data {
        let b = c.to_be_bytes();
        out.push(b[0]);
        out.push(b[1]);
    }
    out
}

fn create_xyz([x, y, z]: [i32; 3]) -> [u8; 20] {
    let mut out = [0u8; 20];
    out[..4].copy_from_slice(b"XYZ ");
    out[8..][..4].copy_from_slice(&x.to_be_bytes());
    out[12..][..4].copy_from_slice(&y.to_be_bytes());
    out[16..][..4].copy_from_slice(&z.to_be_bytes());
    out
}

fn create_curv_lut(lut: &[u16]) -> Vec<u8> {
    let len = lut.len() as u32;
    assert!(len >= 2);
    let mut trc = vec![b'c', b'u', b'r', b'v', 0, 0, 0, 0, 0, 0, 0, 0];
    trc[8..12].copy_from_slice(&len.to_be_bytes());
    trc.resize(trc.len() + lut.len() * 2, 0);
    for (buf, &v) in trc[12..].chunks_exact_mut(2).zip(lut) {
        buf.copy_from_slice(&v.to_be_bytes());
    }
    trc
}

fn create_para(ty: u16, params: &[u32]) -> Vec<u8> {
    let mut out = vec![b'p', b'a', b'r', b'a', 0, 0, 0, 0, 0, 0, 0, 0];
    out[8..10].copy_from_slice(&ty.to_be_bytes());
    for &p in params {
        out.extend_from_slice(&p.to_be_bytes());
    }
    out
}

/// Creates an ICCv4 profile from the given [`ColourEncoding`].
pub fn colour_encoding_to_icc(colour_encoding: &ColourEncoding) -> Result<Vec<u8>> {
    let ColourEncoding {
        want_icc,
        mut colour_space,
        mut white_point,
        mut primaries,
        mut tf,
        mut rendering_intent,
        ..
    } = *colour_encoding;

    if want_icc {
        // Create absolute linear sRGB profile
        colour_space = ColourSpace::Rgb;
        white_point = WhitePoint::D65;
        primaries = Primaries::Srgb;
        tf = TransferFunction::Linear;
        rendering_intent = RenderingIntent::Absolute;
    }

    if colour_space == ColourSpace::Xyb {
        todo!("ICC profile for XYB color space is not supported yet");
    }

    let mut header = vec![
        0, 0, 0, 0, // profile size
        b'j', b'x', b'l', b' ',
        4, 0x40, 0, 0,
        b'm', b'n', b't', b'r',
        // 0x10
        0, 0, 0, 0, // device space
        b'X', b'Y', b'Z', b' ',
        7, 0xe7, 0, 4, 0, 22, 0, 0, 0, 0, 0, 0, // datetime
        b'a', b'c', b's', b'p',
        b'A', b'P', b'P', b'L',
        0, 0, 0, 0,
        // 0x30
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        // 0x40
        0, 0, 0, 0, // rendering intent
        0, 0, 0xf6, 0xd6, 0, 1, 0, 0, 0, 0, 0xd3, 0x2d, // D50 (X=.9642, Y=1., Z=.8249)
        // 0x50
        b'j', b'x', b'l', b' ',
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // MD5
    ];
    header.resize(128, 0);

    header[16..20].copy_from_slice(match colour_space {
        ColourSpace::Rgb => b"RGB ",
        ColourSpace::Grey => b"GRAY",
        ColourSpace::Xyb => b"3CLR",
        ColourSpace::Unknown => b"3CLR",
    });
    header[0x43] = match rendering_intent {
        RenderingIntent::Perceptual => 0,
        RenderingIntent::Relative => 1,
        RenderingIntent::Saturation => 2,
        RenderingIntent::Absolute => 3,
    };

    let mut tags = Vec::new();
    let mut data = Vec::new();
    let desc = format!(
        "{:?}_{:?}_{:?}_{:?}_{:?}",
        colour_space,
        rendering_intent,
        white_point,
        primaries,
        tf,
    );
    append_tag_with_data(
        &mut tags,
        &mut data,
        *b"desc",
        &create_mluc(*b"enUS", &[&desc]),
    );
    append_tag_with_data(
        &mut tags,
        &mut data,
        *b"cprt",
        &create_mluc(*b"enUS", &["(C) 2023 Wonwoo Choi, CC-BY-SA 3.0"]),
    );
    append_tag_with_data(
        &mut tags,
        &mut data,
        *b"wtpt",
        &create_xyz([0xf6d6, 0x10000, 0xd32d]),
    );

    let from_illuminant = match white_point {
        WhitePoint::D65 => ILLUMINANT_D65,
        WhitePoint::Custom(xy) => [xy.x as f32 / 1e6, xy.y as f32 / 1e6],
        WhitePoint::E => ILLUMINANT_E,
        WhitePoint::Dci => ILLUMINANT_DCI,
    };
    let chad = adapt_mat(from_illuminant, ILLUMINANT_D50);
    let chad_q = chad.map(|f| (f * 65536.0 + 0.5) as i32);
    let mut chad_data = vec![b's', b'f', b'3', b'2', 0, 0, 0, 0];
    for val in chad_q {
        chad_data.extend_from_slice(&val.to_be_bytes());
    }
    append_tag_with_data(&mut tags, &mut data, *b"chad", &chad_data);

    let trc = match tf {
        TransferFunction::Gamma(g) => {
            let g = g as u64;
            let adj = g / 2;
            let gamma = ((65536u64 * 10000000u64 + adj) / g) as u32;
            create_para(0, &[gamma])
        },
        TransferFunction::Bt709 => create_para(3, &[
            (65536 * 20 + 4) / 9,
            (65536 * 1000 + 549) / 1099,
            (65536 * 99 + 549) / 1099,
            (65536 * 10 + 22) / 45,
            (65536 * 81 + 500) / 1000,
        ]),
        TransferFunction::Unknown => panic!(),
        TransferFunction::Linear => vec![b'c', b'u', b'r', b'v', 0, 0, 0, 0, 0, 0, 0, 0],
        TransferFunction::Srgb => create_para(3, &[
            (65536 * 24 + 5) / 10,
            (65536 * 1000 + 527) / 1055,
            (65536 * 55 + 527) / 1055,
            (65536 * 100 + 646) / 1292,
            (65536 * 4045 + 50000) / 100000,
        ]),
        TransferFunction::Pq => create_curv_lut(&tf::pq_table(4096)),
        TransferFunction::Dci => create_para(0, &[(65536 * 26 + 5) / 10]),
        TransferFunction::Hlg => create_curv_lut(&tf::hlg_table(4096)),
    };

    let primaries = match primaries {
        Primaries::Srgb => PRIMARIES_SRGB,
        Primaries::Custom { red, green, blue } => [
            [red.x as f32 / 1e6, red.y as f32 / 1e6],
            [green.x as f32 / 1e6, green.y as f32 / 1e6],
            [blue.x as f32 / 1e6, blue.y as f32 / 1e6],
        ],
        Primaries::Bt2100 => PRIMARIES_BT2100,
        Primaries::P3 => PRIMARIES_P3,
    };

    if matches!(tf, TransferFunction::Pq | TransferFunction::Hlg) {
        if let Some(cicp) = colour_encoding.cicp() {
            append_tag_with_data(&mut tags, &mut data, *b"cicp", &cicp);
        }
    }

    match colour_space {
        ColourSpace::Rgb => {
            append_multiple_tags_with_data(&mut tags, &mut data, &[*b"rTRC", *b"gTRC", *b"bTRC"], &trc);
            let p_xyz = primaries_to_xyz_mat(primaries, from_illuminant);
            let p_pcs = matmul3(&chad, &p_xyz);
            let p_pcs_q = p_pcs.map(|f| (f * 65536.0 + 0.5) as i32);
            let p_data = [
                [p_pcs_q[0], p_pcs_q[3], p_pcs_q[6]],
                [p_pcs_q[1], p_pcs_q[4], p_pcs_q[7]],
                [p_pcs_q[2], p_pcs_q[5], p_pcs_q[8]],
            ];
            append_tag_with_data(&mut tags, &mut data, *b"rXYZ", &create_xyz(p_data[0]));
            append_tag_with_data(&mut tags, &mut data, *b"gXYZ", &create_xyz(p_data[1]));
            append_tag_with_data(&mut tags, &mut data, *b"bXYZ", &create_xyz(p_data[2]));
        },
        ColourSpace::Grey => {
            append_tag_with_data(&mut tags, &mut data, *b"kTRC", &trc);
        },
        ColourSpace::Xyb => todo!(),
        ColourSpace::Unknown => panic!("Unknown color space, ICC profile not embedded?"),
    }

    let data_offset = 128 + 4 + tags.len() as u32 * 12;
    let mut out = header;
    out.reserve(4 + tags.len() * 12 + data.len());

    out.extend_from_slice(&(tags.len() as u32).to_be_bytes());
    for tag in tags {
        out.extend_from_slice(&tag.tag);
        out.extend_from_slice(&(tag.data_offset + data_offset).to_be_bytes());
        out.extend_from_slice(&tag.len.to_be_bytes());
    }

    out.extend(data);
    let total_len = out.len() as u32;
    out[..4].copy_from_slice(&total_len.to_be_bytes());
    Ok(out)
}
