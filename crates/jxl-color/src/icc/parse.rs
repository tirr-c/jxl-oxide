use crate::{
    ColourSpace, Customxy, EnumColourEncoding, Error, Primaries, RenderingIntent, Result,
    TransferFunction, WhitePoint,
};

#[derive(Debug)]
pub struct IccProfileInfo {
    color_space: [u8; 4],
    rendering_intent: RenderingIntent,
    chad: [i32; 9],
    wtpt: [i32; 3],
    trc_k: Option<KnownIccTrc>,
    trc_rgb: Option<[KnownIccTrc; 3]>,
    xyz_rgb: Option<[[i32; 3]; 3]>,
}

impl IccProfileInfo {
    #[inline]
    pub fn is_rgb(&self) -> bool {
        &self.color_space == b"RGB "
    }

    #[inline]
    pub fn is_cmyk(&self) -> bool {
        &self.color_space == b"CMYK"
    }

    #[inline]
    pub fn is_grayscale(&self) -> bool {
        &self.color_space == b"GRAY"
    }

    #[inline]
    pub fn rendering_intent(&self) -> RenderingIntent {
        self.rendering_intent
    }

    #[inline]
    pub fn trc_color(&self) -> Option<TransferFunction> {
        let [r, g, b] = self.trc_rgb?;
        if r == g && g == b {
            Some(r.into())
        } else {
            None
        }
    }

    #[inline]
    pub fn trc_gray(&self) -> Option<TransferFunction> {
        self.trc_k.map(From::from)
    }

    #[inline]
    fn chad_f32(&self) -> [f32; 9] {
        self.chad.map(|x| x as f32 / 65536f32)
    }

    #[inline]
    fn chad_inv(&self) -> [f32; 9] {
        let chad_mat = self.chad_f32();
        crate::ciexyz::matinv(&chad_mat)
    }

    pub fn primaries(&self) -> Option<Primaries> {
        const PRIMARIES_TO_ENUM: [([[f32; 2]; 3], Primaries); 3] = [
            (crate::consts::PRIMARIES_SRGB, Primaries::Srgb),
            (crate::consts::PRIMARIES_P3, Primaries::P3),
            (crate::consts::PRIMARIES_BT2100, Primaries::Bt2100),
        ];

        let xyz_rgb = self.xyz_rgb?;

        let xyz_rgb_mat: [f32; 9] = std::array::from_fn(|idx| {
            let v = xyz_rgb[idx % 3][idx / 3];
            v as f32 / 65536f32
        });
        let chad_inv = self.chad_inv();
        let xyz_rgb_mat_adapted = crate::ciexyz::matmul3(&chad_inv, &xyz_rgb_mat);

        let xyz_sum = [
            xyz_rgb_mat_adapted[0] + xyz_rgb_mat_adapted[3] + xyz_rgb_mat_adapted[6],
            xyz_rgb_mat_adapted[1] + xyz_rgb_mat_adapted[4] + xyz_rgb_mat_adapted[7],
            xyz_rgb_mat_adapted[2] + xyz_rgb_mat_adapted[5] + xyz_rgb_mat_adapted[8],
        ];
        let primaries = [
            [
                xyz_rgb_mat_adapted[0] / xyz_sum[0],
                xyz_rgb_mat_adapted[3] / xyz_sum[0],
            ],
            [
                xyz_rgb_mat_adapted[1] / xyz_sum[1],
                xyz_rgb_mat_adapted[4] / xyz_sum[1],
            ],
            [
                xyz_rgb_mat_adapted[2] / xyz_sum[2],
                xyz_rgb_mat_adapted[5] / xyz_sum[2],
            ],
        ];

        'outer: for (known_primaries, ret) in PRIMARIES_TO_ENUM {
            for y in 0..3 {
                for x in 0..2 {
                    let diff = (primaries[y][x] - known_primaries[y][x]).abs();
                    if diff >= 1e-4 {
                        continue 'outer;
                    }
                }
            }
            return Some(ret);
        }

        Some(Primaries::Custom {
            red: Customxy {
                x: (primaries[0][0] * 1e6 + 0.5) as i32,
                y: (primaries[0][1] * 1e6 + 0.5) as i32,
            },
            green: Customxy {
                x: (primaries[1][0] * 1e6 + 0.5) as i32,
                y: (primaries[1][1] * 1e6 + 0.5) as i32,
            },
            blue: Customxy {
                x: (primaries[2][0] * 1e6 + 0.5) as i32,
                y: (primaries[2][1] * 1e6 + 0.5) as i32,
            },
        })
    }

    #[inline]
    fn media_white(&self) -> [f32; 3] {
        self.wtpt.map(|v| v as f32 / 65536f32)
    }

    pub fn white_point(&self) -> WhitePoint {
        const WP_TO_ENUM: [([f32; 2], WhitePoint); 3] = [
            (crate::consts::ILLUMINANT_D65, WhitePoint::D65),
            (crate::consts::ILLUMINANT_DCI, WhitePoint::Dci),
            (crate::consts::ILLUMINANT_E, WhitePoint::E),
        ];

        let chad_inv = self.chad_inv();
        let ill_xyz = crate::ciexyz::matmul3vec(&chad_inv, &self.media_white());
        let xyz_sum = ill_xyz[0] + ill_xyz[1] + ill_xyz[2];
        let illuminant = [ill_xyz[0] / xyz_sum, ill_xyz[1] / xyz_sum];

        'outer: for (known_wp, ret) in WP_TO_ENUM {
            for x in 0..2 {
                let diff = (illuminant[x] - known_wp[x]).abs();
                if diff >= 1e-4 {
                    continue 'outer;
                }
            }
            return ret;
        }

        WhitePoint::Custom(Customxy {
            x: (illuminant[0] * 1e6 + 0.5) as i32,
            y: (illuminant[1] * 1e6 + 0.5) as i32,
        })
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
enum KnownIccTrc {
    ParametricGamma(u32),
    Linear,
    Srgb,
    Bt709,
    Pq,
    Hlg,
}

impl From<KnownIccTrc> for TransferFunction {
    fn from(value: KnownIccTrc) -> Self {
        match value {
            KnownIccTrc::ParametricGamma(g) => {
                let g = g as u64;
                let g_1e7 = (g * 10000000 + 32768) / 65536;
                TransferFunction::Gamma {
                    g: g_1e7 as u32,
                    inverted: false,
                }
            }
            KnownIccTrc::Linear => TransferFunction::Linear,
            KnownIccTrc::Srgb => TransferFunction::Srgb,
            KnownIccTrc::Bt709 => TransferFunction::Bt709,
            KnownIccTrc::Pq => TransferFunction::Pq,
            KnownIccTrc::Hlg => TransferFunction::Hlg,
        }
    }
}

impl KnownIccTrc {
    fn from_gamma(g_s15fixed16: i32) -> Option<Self> {
        match g_s15fixed16 {
            ..=65535 => None,
            65536 => Some(Self::Linear),
            g => Some(Self::ParametricGamma(g as u32)),
        }
    }
}

pub(crate) struct IccProfile<'a> {
    pub(crate) header: super::IccHeader,
    tags: Vec<RawTag<'a>>,
}

impl IccProfile<'_> {
    pub(crate) fn color_space(&self) -> ColourSpace {
        let color_space = &self.header.color_space;
        if color_space == b"RGB " || color_space == b"CMYK" {
            ColourSpace::Rgb
        } else if color_space == b"GRAY" {
            ColourSpace::Grey
        } else {
            ColourSpace::Unknown
        }
    }

    pub(crate) fn is_cmyk(&self) -> bool {
        &self.header.color_space == b"CMYK"
    }
}

struct RawTag<'a> {
    tag: [u8; 4],
    data: &'a [u8],
}

pub(crate) fn parse_icc_raw(profile: &[u8]) -> Result<IccProfile> {
    if profile.len() < 128 {
        return Err(Error::IccParseFailure("profile is too short"));
    }

    let size = u32::from_be_bytes([profile[0], profile[1], profile[2], profile[3]]);
    if profile.len() != size as usize {
        return Err(Error::IccParseFailure("profile size mismatch"));
    }

    let color_space = [profile[0x10], profile[0x11], profile[0x12], profile[0x13]];
    let rendering_intent_raw = profile[0x43];
    let rendering_intent = match rendering_intent_raw {
        0 => RenderingIntent::Perceptual,
        1 => RenderingIntent::Relative,
        2 => RenderingIntent::Saturation,
        3 => RenderingIntent::Absolute,
        _ => return Err(Error::IccParseFailure("invalid rendering intent")),
    };

    let header = super::IccHeader {
        color_space,
        rendering_intent,
    };

    if size < 0x84 {
        return Ok(IccProfile {
            header,
            tags: Vec::new(),
        });
    }

    let tag_count =
        u32::from_be_bytes([profile[0x80], profile[0x81], profile[0x82], profile[0x83]]);
    if size < 0x84 + 12 * tag_count {
        return Err(Error::IccParseFailure(
            "unexpected end of profile while reading tag list",
        ));
    }

    let mut tags = Vec::new();
    let tag_bytes = &profile[0x84..][..12 * tag_count as usize];
    for raw_tag in tag_bytes.chunks_exact(12) {
        let tag = [raw_tag[0], raw_tag[1], raw_tag[2], raw_tag[3]];
        let offset = u32::from_be_bytes([raw_tag[4], raw_tag[5], raw_tag[6], raw_tag[7]]);
        let tag_size = u32::from_be_bytes([raw_tag[8], raw_tag[9], raw_tag[10], raw_tag[11]]);
        let tag_end = offset + tag_size;
        if size < tag_end {
            return Err(Error::IccParseFailure(
                "unexpected end of profile while reading tag data",
            ));
        }

        tags.push(RawTag {
            tag,
            data: &profile[offset as usize..tag_end as usize],
        });
    }

    Ok(IccProfile { header, tags })
}

pub fn detect_profile_info(profile: &[u8]) -> Result<IccProfileInfo> {
    let profile = parse_icc_raw(profile)?;

    let color_space = profile.header.color_space;
    let rendering_intent = profile.header.rendering_intent;

    let mut wtpt = [0xf6d6, 0x10000, 0xd32d]; // D50
    let mut chad: Option<[i32; 9]> = None;
    let mut trcs: [Option<KnownIccTrc>; 4] = [None; 4];
    let mut xyzs: [Option<[i32; 3]>; 3] = [None; 3];
    let mut cicp: Option<[u8; 4]> = None;
    for tag in profile.tags {
        let data = tag.data;
        let tag = tag.tag;
        if data.len() < 4 {
            continue;
        }

        match tag {
            [color, b'T', b'R', b'C'] => {
                let index = match color {
                    b'r' => 0,
                    b'g' => 1,
                    b'b' => 2,
                    b'k' => 3,
                    _ => continue,
                };

                let tf = match data {
                    [b'p', b'a', b'r', b'a', ..] => {
                        if data.len() < 12 {
                            continue;
                        }
                        let curve_type = u16::from_be_bytes([data[8], data[9]]);
                        let parameters = (data.len() - 12) / 4;
                        match curve_type {
                            0 => {
                                if parameters != 1 {
                                    return Err(Error::IccParseFailure(
                                        "invalid parametricCurveType",
                                    ));
                                }

                                let [gamma]: [i32; 1] = std::array::from_fn(|idx| {
                                    let mut bytes = [0u8; 4];
                                    bytes.copy_from_slice(&data[12 + 4 * idx..][..4]);
                                    i32::from_be_bytes(bytes)
                                });
                                if let Some(trc) = KnownIccTrc::from_gamma(gamma) {
                                    trc
                                } else {
                                    continue;
                                }
                            }
                            3 => {
                                if parameters != 5 {
                                    return Err(Error::IccParseFailure(
                                        "invalid parametricCurveType",
                                    ));
                                }

                                let params: [i32; 5] = std::array::from_fn(|idx| {
                                    let mut bytes = [0u8; 4];
                                    bytes.copy_from_slice(&data[12 + 4 * idx..][..4]);
                                    i32::from_be_bytes(bytes)
                                });

                                if params
                                    == [
                                        (65536 * 20 + 4) / 9,
                                        (65536 * 1000 + 549) / 1099,
                                        (65536 * 99 + 549) / 1099,
                                        (65536 * 10 + 22) / 45,
                                        (65536 * 81 + 500) / 1000,
                                    ]
                                {
                                    KnownIccTrc::Bt709
                                } else if params
                                    == [
                                        (65536 * 24 + 5) / 10,
                                        (65536 * 1000 + 527) / 1055,
                                        (65536 * 55 + 527) / 1055,
                                        (65536 * 100 + 646) / 1292,
                                        (65536 * 4045 + 50000) / 100000,
                                    ]
                                {
                                    KnownIccTrc::Srgb
                                } else if let [gamma, 65536, 0, 65536, 0] = params {
                                    if let Some(trc) = KnownIccTrc::from_gamma(gamma) {
                                        trc
                                    } else {
                                        continue;
                                    }
                                } else {
                                    continue;
                                }
                            }
                            _ => continue,
                        }
                    }
                    [b'c', b'u', b'r', b'v', 0, 0, 0, 0, 0, 0, 0, 0] => KnownIccTrc::Linear,
                    [b'c', b'u', b'r', b'v', 0, 0, 0, 0, 0, 0, 0, 1, a, b] => {
                        KnownIccTrc::ParametricGamma(u32::from_be_bytes([0, *a, *b, 0]))
                    }
                    _ => continue,
                };

                trcs[index] = Some(tf);
            }
            [color, b'X', b'Y', b'Z'] => {
                let index = match color {
                    b'r' => 0,
                    b'g' => 1,
                    b'b' => 2,
                    _ => continue,
                };

                if &data[..4] != b"XYZ " || data.len() < 20 {
                    return Err(Error::IccParseFailure("invalid colorant tag"));
                }

                let xyz = [
                    i32::from_be_bytes([data[8], data[9], data[10], data[11]]),
                    i32::from_be_bytes([data[12], data[13], data[14], data[15]]),
                    i32::from_be_bytes([data[16], data[17], data[18], data[19]]),
                ];
                validate_xyz(xyz)?;
                xyzs[index] = Some(xyz);
            }
            [b'c', b'h', b'a', b'd'] => {
                if &data[..4] != b"sf32" || data.len() < 44 {
                    return Err(Error::IccParseFailure("invalid chad tag"));
                }

                let mat = std::array::from_fn(|idx| {
                    let mut bytes = [0u8; 4];
                    bytes.copy_from_slice(&data[8 + 4 * idx..][..4]);
                    i32::from_be_bytes(bytes)
                });
                validate_chad(mat)?;
                chad = Some(mat);
            }
            [b'w', b't', b'p', b't'] => {
                if &data[..4] != b"XYZ " || data.len() < 20 {
                    return Err(Error::IccParseFailure("invalid wtpt tag"));
                }

                wtpt = [
                    i32::from_be_bytes([data[8], data[9], data[10], data[11]]),
                    i32::from_be_bytes([data[12], data[13], data[14], data[15]]),
                    i32::from_be_bytes([data[16], data[17], data[18], data[19]]),
                ];
                validate_xyz(wtpt)?;
            }
            [b'A' | b'D', b'2', b'B', b'0'..=b'3']
            | [b'B', b'2', b'A' | b'D', b'0'..=b'3']
            | [b'p', b'r', b'e', b'0'..=b'2'] => {
                return Err(Error::UnsupportedIccProfile);
            }
            ref x
                if x == b"chrm"
                    || x == b"clro"
                    || x == b"clrt"
                    || x == b"clot"
                    || x == b"ciis"
                    || x == b"lumi"
                    || x == b"meas"
                    || x == b"ncl2"
                    || x == b"resp"
                    || x == b"view" =>
            {
                return Err(Error::UnsupportedIccProfile);
            }
            [b'c', b'i', b'c', b'p'] => {
                cicp = data[..4].try_into().ok();
            }
            _ => {}
        }
    }

    let override_trc = if let Some([_, 16, _, _]) = cicp {
        Some(KnownIccTrc::Pq)
    } else if let Some([_, 18, _, _]) = cicp {
        Some(KnownIccTrc::Hlg)
    } else {
        None
    };
    let trc_rgb = if let [Some(r), Some(g), Some(b), _] = trcs {
        if let Some(trc) = override_trc {
            Some([trc; 3])
        } else {
            Some([r, g, b])
        }
    } else {
        None
    };
    let trc_k = trcs[3].map(|k| override_trc.unwrap_or(k));
    let xyz_rgb = if let [Some(r), Some(g), Some(b)] = xyzs {
        Some([r, g, b])
    } else {
        None
    };

    Ok(IccProfileInfo {
        color_space,
        rendering_intent,
        chad: chad.unwrap_or([65536, 0, 0, 0, 65536, 0, 0, 0, 65536]),
        wtpt,
        trc_k,
        trc_rgb,
        xyz_rgb,
    })
}

fn validate_xyz(xyz_s15fixed16: [i32; 3]) -> Result<()> {
    let flt = xyz_s15fixed16.map(|v| v as f32 / 65536.0);
    let sum = flt[0] + flt[1] + flt[2];
    for v in flt {
        if !(v / sum).is_finite() {
            return Err(Error::IccParseFailure("invalid XYZType"));
        }
    }
    Ok(())
}

fn validate_chad(chad_s15fixed16: [i32; 9]) -> Result<()> {
    let mat_float = chad_s15fixed16.map(|v| v as f32 / 65536.0);
    let chad_inv = crate::ciexyz::matinv(&mat_float);
    for v in chad_inv {
        if !v.is_finite() {
            return Err(Error::IccParseFailure("invalid chad tag"));
        }
    }
    Ok(())
}

pub(crate) fn parse_icc(profile: &[u8]) -> Result<EnumColourEncoding> {
    let info = detect_profile_info(profile)?;
    let rendering_intent = info.rendering_intent();

    if info.is_cmyk() {
        Err(Error::UnsupportedIccProfile)
    } else if info.is_grayscale() {
        let Some(tf) = info.trc_gray() else {
            return Err(Error::UnsupportedIccProfile);
        };
        let wp = info.white_point();
        Ok(EnumColourEncoding {
            colour_space: ColourSpace::Grey,
            white_point: wp,
            primaries: Primaries::Srgb,
            tf,
            rendering_intent,
        })
    } else if info.is_rgb() {
        let (Some(tf), Some(primaries)) = (info.trc_color(), info.primaries()) else {
            return Err(Error::UnsupportedIccProfile);
        };
        let wp = info.white_point();
        Ok(EnumColourEncoding {
            colour_space: ColourSpace::Rgb,
            white_point: wp,
            primaries,
            tf,
            rendering_intent,
        })
    } else {
        Err(Error::UnsupportedIccProfile)
    }
}

#[inline]
pub fn icc_tf(profile: &[u8]) -> Option<TransferFunction> {
    parse_icc(profile).ok().map(|profile| profile.tf)
}

#[cfg(test)]
mod tests {
    use super::parse_icc;
    use crate::*;

    #[test]
    fn srgb_rel() {
        let profile = parse_icc(include_bytes!("./test-profiles/srgb-rel.icc")).unwrap();
        dbg!(&profile);
        assert!(matches!(
            profile,
            EnumColourEncoding {
                colour_space: ColourSpace::Rgb,
                white_point: WhitePoint::D65,
                primaries: Primaries::Srgb,
                tf: TransferFunction::Srgb,
                rendering_intent: RenderingIntent::Relative,
            },
        ));
    }

    #[test]
    fn srgb_bt709_per() {
        let profile = parse_icc(include_bytes!("./test-profiles/srgb-bt709-per.icc")).unwrap();
        dbg!(&profile);
        assert!(matches!(
            profile,
            EnumColourEncoding {
                colour_space: ColourSpace::Rgb,
                white_point: WhitePoint::D65,
                primaries: Primaries::Srgb,
                tf: TransferFunction::Bt709,
                rendering_intent: RenderingIntent::Perceptual,
            },
        ));
    }

    #[test]
    fn srgb_gamma22_per() {
        // gamma of the profile is 0x23332 (0.4545500), not 0x23333 (0.4545455)
        let profile = parse_icc(include_bytes!("./test-profiles/srgb-gamma22-rel.icc")).unwrap();
        dbg!(&profile);
        assert!(matches!(
            profile,
            EnumColourEncoding {
                colour_space: ColourSpace::Rgb,
                white_point: WhitePoint::D65,
                primaries: Primaries::Srgb,
                tf: TransferFunction::Gamma {
                    g: 21999000..=22001000,
                    inverted: false
                },
                rendering_intent: RenderingIntent::Relative,
            },
        ));
    }

    #[test]
    fn srgb_linear_per() {
        let profile = parse_icc(include_bytes!("./test-profiles/srgb-linear-rel.icc")).unwrap();
        dbg!(&profile);
        assert!(matches!(
            profile,
            EnumColourEncoding {
                colour_space: ColourSpace::Rgb,
                white_point: WhitePoint::D65,
                primaries: Primaries::Srgb,
                tf: TransferFunction::Linear,
                rendering_intent: RenderingIntent::Relative,
            },
        ));
    }

    #[test]
    fn gray_d65_srgb_rel() {
        let profile = parse_icc(include_bytes!("./test-profiles/gray-d65-srgb-rel.icc")).unwrap();
        dbg!(&profile);
        assert!(matches!(
            profile,
            EnumColourEncoding {
                colour_space: ColourSpace::Grey,
                white_point: WhitePoint::D65,
                tf: TransferFunction::Srgb,
                rendering_intent: RenderingIntent::Relative,
                ..
            },
        ));
    }

    #[test]
    fn gray_d65_linear_rel() {
        let profile = parse_icc(include_bytes!("./test-profiles/gray-d65-linear-rel.icc")).unwrap();
        dbg!(&profile);
        assert!(matches!(
            profile,
            EnumColourEncoding {
                colour_space: ColourSpace::Grey,
                white_point: WhitePoint::D65,
                tf: TransferFunction::Linear,
                rendering_intent: RenderingIntent::Relative,
                ..
            },
        ));
    }
}
