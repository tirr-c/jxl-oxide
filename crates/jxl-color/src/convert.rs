use crate::{
    ciexyz::*, consts::*, icc::colour_encoding_to_icc, tf, ColorManagementSystem, ColourEncoding,
    ColourSpace, EnumColourEncoding, Error, OpsinInverseMatrix, RenderingIntent, Result,
    ToneMapping, TransferFunction,
};

mod gamut_map;
mod tone_map;

#[derive(Clone)]
pub struct ColorEncodingWithProfile {
    encoding: ColourEncoding,
    icc_profile: Vec<u8>,
}

impl std::fmt::Debug for ColorEncodingWithProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ColorEncodingWithProfile")
            .field("encoding", &self.encoding)
            .field(
                "icc_profile",
                &format_args!("({} byte(s))", self.icc_profile.len()),
            )
            .finish()
    }
}

impl ColorEncodingWithProfile {
    pub fn new(encoding: ColourEncoding) -> Self {
        assert!(!matches!(encoding, ColourEncoding::IccProfile(_)));
        Self {
            encoding,
            icc_profile: Vec::new(),
        }
    }

    pub fn with_icc(encoding: ColourEncoding, icc_profile: Vec<u8>) -> Self {
        assert!(matches!(encoding, ColourEncoding::IccProfile(_)));
        Self {
            encoding,
            icc_profile,
        }
    }

    #[inline]
    pub fn encoding(&self) -> &ColourEncoding {
        &self.encoding
    }

    #[inline]
    pub fn icc_profile(&self) -> &[u8] {
        &self.icc_profile
    }

    #[inline]
    pub fn is_grayscale(&self) -> bool {
        match &self.encoding {
            ColourEncoding::Enum(encoding) => encoding.colour_space == ColourSpace::Grey,
            ColourEncoding::IccProfile(_) => {
                self.icc_profile.len() > 0x14 && self.icc_profile[0x10..0x14] == *b"GRAY"
            }
        }
    }
}

impl ColorEncodingWithProfile {
    fn is_equivalent(&self, other: &Self) -> bool {
        if self.encoding.want_icc() != other.encoding.want_icc() {
            return false;
        }

        if self.encoding.want_icc() {
            return self.icc_profile == other.icc_profile;
        }

        match (&self.encoding, &other.encoding) {
            (ColourEncoding::Enum(me), ColourEncoding::Enum(other)) => {
                if me.colour_space != other.colour_space {
                    return false;
                }

                if me.colour_space == ColourSpace::Xyb {
                    return true;
                }

                if me.rendering_intent != other.rendering_intent
                    || me.white_point != other.white_point
                    || me.tf != other.tf
                {
                    return false;
                }

                me.colour_space == ColourSpace::Grey || me.primaries == other.primaries
            }
            _ => false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ColorTransform {
    begin_channels: usize,
    ops: Vec<ColorTransformOp>,
}

impl ColorTransform {
    pub fn new(
        from: &ColorEncodingWithProfile,
        to: &ColorEncodingWithProfile,
        oim: &OpsinInverseMatrix,
        tone_mapping: &ToneMapping,
    ) -> Result<Self> {
        let intensity_target = tone_mapping.intensity_target;
        let min_nits = tone_mapping.min_nits;

        let from_parsed;
        let from = match &from.encoding {
            ColourEncoding::IccProfile(_) => {
                if let Ok(encoding) = crate::icc::parse_icc(&from.icc_profile) {
                    from_parsed = encoding;
                    &from_parsed
                } else {
                    from
                }
            }
            _ => from,
        };

        let to_parsed;
        let to = match &to.encoding {
            ColourEncoding::IccProfile(_) => {
                if let Ok(encoding) = crate::icc::parse_icc(&to.icc_profile) {
                    to_parsed = encoding;
                    &to_parsed
                } else {
                    to
                }
            }
            _ => to,
        };

        let begin_channels = match &from.encoding {
            ColourEncoding::Enum(EnumColourEncoding {
                colour_space: ColourSpace::Grey,
                ..
            }) => 1,
            ColourEncoding::Enum(_) => 3,
            ColourEncoding::IccProfile(_) => {
                let profile = from.icc_profile();
                if profile.len() < 0x14 {
                    3
                } else {
                    match &profile[0x10..0x14] {
                        [b'G', b'R', b'A', b'Y'] => 1,
                        [b'C', b'M', b'Y', b'K'] => 4,
                        _ => 3,
                    }
                }
            }
        };

        if from.is_equivalent(to) {
            return Ok(Self {
                begin_channels,
                ops: Vec::new(),
            });
        }

        let mut ops = Vec::new();

        let current_encoding = match from.encoding {
            ColourEncoding::IccProfile(_) => {
                let rendering_intent = crate::icc::parse_icc_raw(&from.icc_profile)?
                    .header
                    .rendering_intent;
                match &to.encoding {
                    ColourEncoding::IccProfile(_) => {
                        return Ok(Self {
                            begin_channels,
                            ops: vec![ColorTransformOp::IccToIcc {
                                inputs: 0,
                                outputs: 0,
                                from: from.icc_profile.clone(),
                                to: to.icc_profile.clone(),
                                rendering_intent,
                            }],
                        });
                    }
                    ColourEncoding::Enum(encoding) => {
                        let mut target_encoding = encoding.clone();
                        target_encoding.tf = TransferFunction::Linear;
                        ops.push(ColorTransformOp::IccToIcc {
                            inputs: 0,
                            outputs: 0,
                            from: from.icc_profile.clone(),
                            to: colour_encoding_to_icc(&target_encoding),
                            rendering_intent,
                        });
                        target_encoding
                    }
                }
            }
            ColourEncoding::Enum(EnumColourEncoding {
                colour_space: ColourSpace::Xyb,
                rendering_intent,
                ..
            }) => {
                let inv_mat = oim.inv_mat;
                #[rustfmt::skip]
                let matrix = [
                    inv_mat[0][0], inv_mat[0][1], inv_mat[0][2],
                    inv_mat[1][0], inv_mat[1][1], inv_mat[1][2],
                    inv_mat[2][0], inv_mat[2][1], inv_mat[2][2],
                ];
                ops.push(ColorTransformOp::XybToMixedLms {
                    opsin_bias: oim.opsin_bias,
                    intensity_target,
                });
                ops.push(ColorTransformOp::Matrix(matrix));
                // result: RGB; D65 illuminant, sRGB primaries, linear tf
                EnumColourEncoding::srgb_linear(rendering_intent)
            }
            ColourEncoding::Enum(
                ref encoding @ EnumColourEncoding {
                    colour_space: ColourSpace::Rgb,
                    white_point,
                    primaries,
                    tf,
                    ..
                },
            ) => {
                let illuminant = white_point.as_chromaticity();
                let mat =
                    crate::ciexyz::primaries_to_xyz_mat(primaries.as_chromaticity(), illuminant);
                let luminances = [mat[3], mat[4], mat[5]];
                ops.push(ColorTransformOp::TransferFunction {
                    tf,
                    hdr_params: HdrParams {
                        luminances,
                        intensity_target,
                        min_nits,
                    },
                    inverse: true,
                });

                let mut current_encoding = encoding.clone();
                current_encoding.tf = TransferFunction::Linear;
                current_encoding
            }
            ColourEncoding::Enum(
                ref encoding @ EnumColourEncoding {
                    colour_space: ColourSpace::Grey,
                    white_point,
                    tf,
                    ..
                },
            ) => {
                let illuminant = white_point.as_chromaticity();
                // primaries don't matter for grayscale
                let mat = crate::ciexyz::primaries_to_xyz_mat(PRIMARIES_SRGB, illuminant);
                let luminances = [mat[3], mat[4], mat[5]];

                ops.push(ColorTransformOp::TransferFunction {
                    tf,
                    hdr_params: HdrParams {
                        luminances,
                        intensity_target,
                        min_nits,
                    },
                    inverse: true,
                });

                let mut current_encoding = encoding.clone();
                current_encoding.tf = TransferFunction::Linear;
                current_encoding
            }
            ColourEncoding::Enum(_) => {
                return Err(Error::UnsupportedColorEncoding);
            }
        };

        let target_encoding = match &to.encoding {
            ColourEncoding::Enum(encoding) => encoding,
            ColourEncoding::IccProfile(_) => {
                ops.push(ColorTransformOp::IccToIcc {
                    inputs: 0,
                    outputs: 0,
                    from: colour_encoding_to_icc(&current_encoding),
                    to: to.icc_profile.clone(),
                    rendering_intent: current_encoding.rendering_intent,
                });
                return Ok(Self {
                    begin_channels,
                    ops,
                });
            }
        };

        debug_assert_eq!(current_encoding.tf, TransferFunction::Linear);

        if current_encoding.colour_space != target_encoding.colour_space
            || current_encoding.white_point != target_encoding.white_point
            || (current_encoding.colour_space == ColourSpace::Rgb
                && current_encoding.primaries != target_encoding.primaries)
        {
            match current_encoding.colour_space {
                ColourSpace::Rgb => {
                    // RGB to XYZ
                    let illuminant = current_encoding.white_point.as_chromaticity();
                    let mat = crate::ciexyz::primaries_to_xyz_mat(
                        current_encoding.primaries.as_chromaticity(),
                        illuminant,
                    );
                    ops.push(ColorTransformOp::Matrix(mat));
                }
                ColourSpace::Grey => {
                    // Yxy to XYZ
                    let illuminant = current_encoding.white_point.as_chromaticity();
                    ops.push(ColorTransformOp::LumaToXyz { illuminant });
                }
                ColourSpace::Xyb | ColourSpace::Unknown => {
                    unreachable!()
                }
            }

            if current_encoding.rendering_intent != RenderingIntent::Absolute {
                // Chromatic adaptation: XYZ to XYZ
                let adapt = adapt_mat(
                    current_encoding.white_point.as_chromaticity(),
                    target_encoding.white_point.as_chromaticity(),
                );
                ops.push(ColorTransformOp::Matrix(adapt));
            }

            match target_encoding.colour_space {
                ColourSpace::Rgb => {
                    // XYZ to RGB
                    let illuminant = target_encoding.white_point.as_chromaticity();
                    let mat = crate::ciexyz::xyz_to_primaries_mat(
                        target_encoding.primaries.as_chromaticity(),
                        illuminant,
                    );
                    ops.push(ColorTransformOp::Matrix(mat));
                }
                ColourSpace::Grey => {
                    // XYZ to Yxy
                    ops.push(ColorTransformOp::XyzToLuma);
                }
                ColourSpace::Xyb | ColourSpace::Unknown => {
                    return Err(Error::UnsupportedColorEncoding);
                }
            }
        }

        let illuminant = target_encoding.white_point.as_chromaticity();
        let mat = crate::ciexyz::primaries_to_xyz_mat(
            target_encoding.primaries.as_chromaticity(),
            illuminant,
        );
        let luminances = [mat[3], mat[4], mat[5]];

        let hdr_params = HdrParams {
            luminances,
            intensity_target,
            min_nits,
        };

        if intensity_target > 255.0 && !target_encoding.is_hdr() {
            if target_encoding.colour_space == ColourSpace::Grey {
                ops.push(ColorTransformOp::ToneMapLumaRec2408 {
                    hdr_params,
                    target_display_luminance: 255.0,
                });
            } else {
                ops.push(ColorTransformOp::ToneMapRec2408 {
                    hdr_params,
                    target_display_luminance: 255.0,
                });

                if current_encoding.rendering_intent == RenderingIntent::Perceptual {
                    ops.push(ColorTransformOp::GamutMap {
                        luminances,
                        saturation_factor: 0.1,
                    });
                }
            }
        }

        if target_encoding.tf != TransferFunction::Linear {
            ops.push(ColorTransformOp::TransferFunction {
                tf: target_encoding.tf,
                hdr_params,
                inverse: false,
            });
        }

        let mut ret = Self {
            begin_channels,
            ops,
        };
        ret.optimize();
        Ok(ret)
    }

    pub fn xyb_to_enum(
        encoding: &EnumColourEncoding,
        oim: &OpsinInverseMatrix,
        tone_mapping: &ToneMapping,
    ) -> Result<Self> {
        Self::new(
            &ColorEncodingWithProfile::new(ColourEncoding::Enum(EnumColourEncoding::xyb())),
            &ColorEncodingWithProfile::new(ColourEncoding::Enum(encoding.clone())),
            oim,
            tone_mapping,
        )
    }

    pub fn run<Cms: ColorManagementSystem + ?Sized>(
        &self,
        channels: &mut [&mut [f32]],
        cms: &Cms,
    ) -> Result<usize> {
        let _gurad = tracing::trace_span!("Run color transform ops").entered();

        let mut num_channels = self.begin_channels;
        for op in &self.ops {
            tracing::trace!(?op);
            num_channels = op.run(channels, num_channels, cms)?;
        }
        Ok(num_channels)
    }

    pub fn run_with_threads<Cms: ColorManagementSystem + Sync + ?Sized>(
        &self,
        channels: &mut [&mut [f32]],
        cms: &Cms,
        pool: &jxl_threadpool::JxlThreadPool,
    ) -> Result<usize> {
        let _gurad = tracing::trace_span!("Run color transform ops").entered();
        tracing::trace!(ops = ?self.ops);

        let mut chunks = Vec::new();
        let mut it = channels
            .iter_mut()
            .map(|ch| ch.chunks_mut(65536))
            .collect::<Vec<_>>();
        loop {
            let Some(chunk) = it
                .iter_mut()
                .map(|it| it.next())
                .collect::<Option<Vec<_>>>()
            else {
                break;
            };
            chunks.push(chunk);
        }

        let ret = std::sync::Mutex::new(Ok(self.begin_channels));
        pool.for_each_vec(chunks, |mut channels| {
            let mut num_channels = self.begin_channels;
            for op in &self.ops {
                match op.run(&mut channels, num_channels, cms) {
                    Ok(x) => {
                        num_channels = x;
                    }
                    err => {
                        *ret.lock().unwrap() = err;
                        return;
                    }
                }
            }
            *ret.lock().unwrap() = Ok(num_channels);
        });
        ret.into_inner().unwrap()
    }

    fn optimize(&mut self) {
        let mut matrix_op_from = None;
        let mut matrix = [0f32; 9];
        let mut idx = 0usize;
        let mut len = self.ops.len();
        while idx < len {
            let op = &self.ops[idx];
            if let ColorTransformOp::Matrix(mat) = op {
                if matrix_op_from.is_none() {
                    matrix_op_from = Some(idx);
                    matrix = *mat;
                } else {
                    matrix = matmul3(mat, &matrix);
                }
            } else if let Some(from) = matrix_op_from {
                self.ops[from] = ColorTransformOp::Matrix(matrix);
                self.ops.drain((from + 1)..idx);
                matrix_op_from = None;
                idx = from + 1;
                len = self.ops.len();
                continue;
            }
            idx += 1;
        }

        if let Some(from) = matrix_op_from {
            self.ops[from] = ColorTransformOp::Matrix(matrix);
            self.ops.drain((from + 1)..len);
        }
    }
}

#[derive(Clone)]
enum ColorTransformOp {
    XybToMixedLms {
        opsin_bias: [f32; 3],
        intensity_target: f32,
    },
    LumaToXyz {
        illuminant: [f32; 2],
    },
    XyzToLuma,
    Matrix([f32; 9]),
    TransferFunction {
        tf: TransferFunction,
        hdr_params: HdrParams,
        inverse: bool,
    },
    ToneMapRec2408 {
        hdr_params: HdrParams,
        target_display_luminance: f32,
    },
    ToneMapLumaRec2408 {
        hdr_params: HdrParams,
        target_display_luminance: f32,
    },
    GamutMap {
        luminances: [f32; 3],
        saturation_factor: f32,
    },
    IccToIcc {
        inputs: usize,
        outputs: usize,
        from: Vec<u8>,
        to: Vec<u8>,
        rendering_intent: RenderingIntent,
    },
}

impl std::fmt::Debug for ColorTransformOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::XybToMixedLms {
                opsin_bias,
                intensity_target,
            } => f
                .debug_struct("XybToMixedLms")
                .field("opsin_bias", opsin_bias)
                .field("intensity_target", intensity_target)
                .finish(),
            Self::LumaToXyz { illuminant } => f
                .debug_struct("LumaToXyz")
                .field("illuminant", illuminant)
                .finish(),
            Self::XyzToLuma => f.debug_struct("XyzToLuma").finish(),
            Self::Matrix(arg0) => f.debug_tuple("Matrix").field(arg0).finish(),
            Self::TransferFunction {
                tf,
                hdr_params,
                inverse,
            } => f
                .debug_struct("TransferFunction")
                .field("tf", tf)
                .field("hdr_params", hdr_params)
                .field("inverse", inverse)
                .finish(),
            Self::ToneMapRec2408 {
                hdr_params,
                target_display_luminance,
            } => f
                .debug_struct("ToneMapRec2408")
                .field("hdr_params", hdr_params)
                .field("target_display_luminance", target_display_luminance)
                .finish(),
            Self::ToneMapLumaRec2408 {
                hdr_params,
                target_display_luminance,
            } => f
                .debug_struct("ToneMapLumaRec2408")
                .field("hdr_params", hdr_params)
                .field("target_display_luminance", target_display_luminance)
                .finish(),
            Self::GamutMap {
                luminances,
                saturation_factor,
            } => f
                .debug_struct("GamutMap")
                .field("luminances", luminances)
                .field("saturation_factor", saturation_factor)
                .finish(),
            Self::IccToIcc {
                inputs,
                outputs,
                from,
                to,
                rendering_intent,
            } => f
                .debug_struct("IccToIcc")
                .field("inputs", inputs)
                .field("outputs", outputs)
                .field("from", &format_args!("({} byte(s))", from.len()))
                .field("to", &format_args!("({} byte(s))", to.len()))
                .field("rendering_intent", rendering_intent)
                .finish(),
        }
    }
}

impl ColorTransformOp {
    #[inline]
    fn inputs(&self) -> Option<usize> {
        match *self {
            ColorTransformOp::XybToMixedLms { .. } | ColorTransformOp::Matrix(_) => Some(3),
            ColorTransformOp::LumaToXyz { .. } => Some(1),
            ColorTransformOp::XyzToLuma => Some(3),
            ColorTransformOp::TransferFunction {
                tf: TransferFunction::Hlg,
                ..
            } => Some(3),
            ColorTransformOp::TransferFunction { .. } => None,
            ColorTransformOp::ToneMapRec2408 { .. } => Some(3),
            ColorTransformOp::ToneMapLumaRec2408 { .. } => Some(1),
            ColorTransformOp::GamutMap { .. } => Some(3),
            ColorTransformOp::IccToIcc { inputs: 0, .. } => None,
            ColorTransformOp::IccToIcc { inputs, .. } => Some(inputs),
        }
    }

    #[inline]
    fn outputs(&self) -> Option<usize> {
        match *self {
            ColorTransformOp::XybToMixedLms { .. } | ColorTransformOp::Matrix(_) => Some(3),
            ColorTransformOp::LumaToXyz { .. } => Some(3),
            ColorTransformOp::XyzToLuma => Some(1),
            ColorTransformOp::TransferFunction {
                tf: TransferFunction::Hlg,
                ..
            } => Some(3),
            ColorTransformOp::TransferFunction { .. } => None,
            ColorTransformOp::ToneMapRec2408 { .. } => Some(3),
            ColorTransformOp::ToneMapLumaRec2408 { .. } => Some(1),
            ColorTransformOp::GamutMap { .. } => Some(3),
            ColorTransformOp::IccToIcc { outputs: 0, .. } => None,
            ColorTransformOp::IccToIcc { outputs, .. } => Some(outputs),
        }
    }

    fn run<Cms: ColorManagementSystem + ?Sized>(
        &self,
        channels: &mut [&mut [f32]],
        num_input_channels: usize,
        cms: &Cms,
    ) -> Result<usize> {
        let channel_count = channels.len();
        if let Some(inputs) = self.inputs() {
            assert!(channel_count >= inputs);
        }
        if let Some(outputs) = self.outputs() {
            assert!(channel_count >= outputs);
        }

        Ok(match self {
            Self::XybToMixedLms {
                opsin_bias,
                intensity_target,
            } => {
                let [x, y, b, ..] = channels else {
                    unreachable!()
                };
                let xyb = [&mut **x, &mut **y, &mut **b];
                crate::xyb::run(xyb, *opsin_bias, *intensity_target);
                3
            }
            Self::LumaToXyz { illuminant } => {
                let [a, b, c, ..] = channels else {
                    unreachable!()
                };
                let [x, y] = illuminant;
                for ((a, b), c) in a.iter_mut().zip(&mut **b).zip(&mut **c) {
                    let luma_div_y = *a / y;
                    *b = *a;
                    *a = x * luma_div_y;
                    *c = (1.0 - x - y) * luma_div_y;
                }
                3
            }
            Self::XyzToLuma => {
                let [x, y, ..] = channels else {
                    unreachable!();
                };
                x.copy_from_slice(y);
                1
            }
            Self::Matrix(matrix) => {
                let [a, b, c, ..] = channels else {
                    unreachable!()
                };
                for ((a, b), c) in a.iter_mut().zip(&mut **b).zip(&mut **c) {
                    let result = crate::ciexyz::matmul3vec(matrix, &[*a, *b, *c]);
                    *a = result[0];
                    *b = result[1];
                    *c = result[2];
                }
                3
            }
            Self::TransferFunction {
                tf,
                hdr_params,
                inverse: false,
            } => {
                apply_transfer_function(&mut channels[..num_input_channels], *tf, *hdr_params);
                channels.len()
            }
            Self::TransferFunction {
                tf,
                hdr_params,
                inverse: true,
            } => {
                apply_inverse_transfer_function(
                    &mut channels[..num_input_channels],
                    *tf,
                    *hdr_params,
                );
                channels.len()
            }
            Self::ToneMapRec2408 {
                hdr_params,
                target_display_luminance,
            } => {
                let [r, g, b, ..] = channels else {
                    unreachable!()
                };
                tone_map::tone_map(r, g, b, hdr_params, *target_display_luminance);
                3
            }
            Self::ToneMapLumaRec2408 {
                hdr_params,
                target_display_luminance,
            } => {
                let [y, ..] = channels else { unreachable!() };
                tone_map::tone_map_luma(y, hdr_params, *target_display_luminance);
                1
            }
            Self::GamutMap {
                luminances,
                saturation_factor,
            } => {
                let [r, g, b, ..] = channels else {
                    unreachable!()
                };
                gamut_map::gamut_map(r, g, b, *luminances, *saturation_factor);
                3
            }
            Self::IccToIcc { from, to, .. } => {
                cms.transform(from, to, RenderingIntent::Relative, channels)?
            }
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct HdrParams {
    luminances: [f32; 3],
    intensity_target: f32,
    min_nits: f32,
}

fn apply_transfer_function(
    channels: &mut [&mut [f32]],
    tf: TransferFunction,
    hdr_params: HdrParams,
) {
    match tf {
        TransferFunction::Gamma {
            g: gamma,
            inverted: false,
        } => {
            let gamma = 1e7 / gamma as f32;
            for ch in channels {
                tf::apply_gamma(ch, gamma);
            }
        }
        TransferFunction::Gamma {
            g: gamma,
            inverted: true,
        } => {
            let gamma = gamma as f32 / 1e7;
            for ch in channels {
                tf::apply_gamma(ch, gamma);
            }
        }
        TransferFunction::Bt709 => {
            for ch in channels {
                tf::linear_to_bt709(ch);
            }
        }
        TransferFunction::Unknown => {}
        TransferFunction::Linear => {}
        TransferFunction::Srgb => {
            for ch in channels {
                tf::linear_to_srgb(ch);
            }
        }
        TransferFunction::Pq => {
            let intensity_target = hdr_params.intensity_target;
            for ch in channels {
                tf::linear_to_pq(ch, intensity_target);
            }
        }
        TransferFunction::Dci => {
            let gamma = 1.0 / 2.6;
            for ch in channels {
                tf::apply_gamma(ch, gamma);
            }
        }
        TransferFunction::Hlg => {
            let [r, g, b, ..] = channels else { panic!() };
            let r = &mut **r;
            let g = &mut **g;
            let b = &mut **b;
            let luminances = hdr_params.luminances;
            let intensity_target = hdr_params.intensity_target;

            tf::hlg_inverse_oo([r, g, b], luminances, intensity_target);
            tf::linear_to_hlg(r);
            tf::linear_to_hlg(g);
            tf::linear_to_hlg(b);
        }
    }
}

fn apply_inverse_transfer_function(
    channels: &mut [&mut [f32]],
    tf: TransferFunction,
    hdr_params: HdrParams,
) {
    match tf {
        TransferFunction::Gamma {
            g: gamma,
            inverted: false,
        } => {
            let gamma = gamma as f32 / 1e7;
            for ch in channels {
                tf::apply_gamma(ch, gamma);
            }
        }
        TransferFunction::Gamma {
            g: gamma,
            inverted: true,
        } => {
            let gamma = 1e7 / gamma as f32;
            for ch in channels {
                tf::apply_gamma(ch, gamma);
            }
        }
        TransferFunction::Bt709 => {
            for ch in channels {
                tf::bt709_to_linear(ch);
            }
        }
        TransferFunction::Unknown => {}
        TransferFunction::Linear => {}
        TransferFunction::Srgb => {
            for ch in channels {
                tf::srgb_to_linear(ch);
            }
        }
        TransferFunction::Pq => {
            let intensity_target = hdr_params.intensity_target;
            for ch in channels {
                tf::pq_to_linear(ch, intensity_target);
            }
        }
        TransferFunction::Dci => {
            let gamma = 2.6;
            for ch in channels {
                tf::apply_gamma(ch, gamma);
            }
        }
        TransferFunction::Hlg => {
            let [r, g, b, ..] = channels else { panic!() };
            let r = &mut **r;
            let g = &mut **g;
            let b = &mut **b;
            let luminances = hdr_params.luminances;
            let intensity_target = hdr_params.intensity_target;

            tf::hlg_to_linear(r);
            tf::hlg_to_linear(g);
            tf::hlg_to_linear(b);
            tf::hlg_oo([r, g, b], luminances, intensity_target);
        }
    }
}
