use crate::{
    ColourSpace, EnumColourEncoding, Primaries, RenderingIntent, TransferFunction, WhitePoint,
    ciexyz::*, consts::*, tf,
};

use super::IccTag;

fn append_tag_with_data(
    tags_out: &mut Vec<IccTag>,
    data_out: &mut Vec<u8>,
    tag: [u8; 4],
    data: &[u8],
) {
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
        out.extend_from_slice(&(0x10 + strings.len() as u32 * 12 + offset * 2).to_be_bytes());
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

/// Creates an ICCv4 profile from the given [`EnumColourEncoding`].
pub fn colour_encoding_to_icc(colour_encoding: &EnumColourEncoding) -> Vec<u8> {
    let &EnumColourEncoding {
        colour_space,
        white_point,
        primaries,
        tf,
        rendering_intent,
    } = colour_encoding;

    if colour_space == ColourSpace::Xyb {
        todo!("ICC profile for XYB color space is not supported yet");
    }

    #[rustfmt::skip]
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
    let desc =
        format!("{colour_space:?}_{rendering_intent:?}_{white_point:?}_{primaries:?}_{tf:?}",);
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
        &create_mluc(*b"enUS", &["CC0, generated by jxl-oxide"]),
    );

    let mut chad = [1f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let from_illuminant = match white_point {
        WhitePoint::D65 => ILLUMINANT_D65,
        WhitePoint::Custom(xy) => [xy.x as f32 / 1e6, xy.y as f32 / 1e6],
        WhitePoint::E => ILLUMINANT_E,
        WhitePoint::Dci => ILLUMINANT_DCI,
    };

    if colour_space == ColourSpace::Rgb {
        append_tag_with_data(
            &mut tags,
            &mut data,
            *b"wtpt",
            &create_xyz([0xf6d6, 0x10000, 0xd32d]),
        );
        chad = adapt_mat(from_illuminant, ILLUMINANT_D50);
        let chad_q = chad.map(|f| (f * 65536.0 + 0.5) as i32);
        let mut chad_data = vec![b's', b'f', b'3', b'2', 0, 0, 0, 0];
        for val in chad_q {
            chad_data.extend_from_slice(&val.to_be_bytes());
        }
        append_tag_with_data(&mut tags, &mut data, *b"chad", &chad_data);
    } else {
        let xyz = illuminant_to_xyz(from_illuminant);
        append_tag_with_data(
            &mut tags,
            &mut data,
            *b"wtpt",
            &create_xyz(xyz.map(|v| (v * 65536.0 + 0.5) as i32)),
        );
    }

    let trc = match tf {
        TransferFunction::Gamma { g, inverted: false } => {
            let g = g as u64;
            let gamma = ((g * 65536 + 5000000) / 10000000) as u32;
            create_para(0, &[gamma])
        }
        TransferFunction::Gamma { g, inverted: true } => {
            let g = g as u64;
            let adj = g / 2;
            let gamma = ((65536u64 * 10000000u64 + adj) / g) as u32;
            create_para(0, &[gamma])
        }
        TransferFunction::Bt709 => create_para(
            3,
            &[
                (65536 * 20 + 4) / 9,
                (65536 * 1000 + 549) / 1099,
                (65536 * 99 + 549) / 1099,
                (65536 * 10 + 22) / 45,
                (65536 * 81 + 500) / 1000,
            ],
        ),
        TransferFunction::Unknown => panic!(),
        TransferFunction::Linear => vec![b'c', b'u', b'r', b'v', 0, 0, 0, 0, 0, 0, 0, 0],
        TransferFunction::Srgb => create_para(
            3,
            &[
                (65536 * 24 + 5) / 10,
                (65536 * 1000 + 527) / 1055,
                (65536 * 55 + 527) / 1055,
                (65536 * 100 + 646) / 1292,
                (65536 * 4045 + 50000) / 100000,
            ],
        ),
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
            let mut cicp_tag = [0u8; 12];
            cicp_tag[..4].copy_from_slice(b"cicp");
            cicp_tag[8..].copy_from_slice(&cicp);
            append_tag_with_data(&mut tags, &mut data, *b"cicp", &cicp_tag);
        }
    }

    match colour_space {
        ColourSpace::Rgb => {
            append_multiple_tags_with_data(
                &mut tags,
                &mut data,
                &[*b"rTRC", *b"gTRC", *b"bTRC"],
                &trc,
            );
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
        }
        ColourSpace::Grey => {
            append_tag_with_data(&mut tags, &mut data, *b"kTRC", &trc);
        }
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
    out
}
