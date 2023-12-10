use crate::{ciexyz::*, consts::*, tf, TransferFunction, EnumColourEncoding, ColourEncoding, OpsinInverseMatrix, ColourSpace, icc::nciexyz_icc_profile, RenderingIntent, ColorManagementSystem};

#[derive(Clone)]
pub struct ColorEncodingWithProfile {
    encoding: ColourEncoding,
    icc_profile: Vec<u8>,
}

impl std::fmt::Debug for ColorEncodingWithProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f
            .debug_struct("ColorEncodingWithProfile")
            .field("encoding", &self.encoding)
            .field("icc_profile", &format_args!("({} byte(s))", self.icc_profile.len()))
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
        Self { encoding, icc_profile }
    }

    #[inline]
    pub fn encoding(&self) -> &ColourEncoding {
        &self.encoding
    }

    #[inline]
    pub fn icc_profile(&self) -> &[u8] {
        &self.icc_profile
    }
}

impl ColorEncodingWithProfile {
    fn is_equivalent(&self, other: &Self) -> bool {
        if self.encoding.want_icc() != other.encoding.want_icc() {
            // TODO: parse profile and check
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

                if me.rendering_intent != other.rendering_intent || me.white_point != other.white_point || me.tf != other.tf {
                    return false;
                }

                me.colour_space == ColourSpace::Grey || me.primaries == other.primaries
            },
            (ColourEncoding::Enum(_), ColourEncoding::PcsXyz) => false,
            (ColourEncoding::PcsXyz, ColourEncoding::Enum(_)) => false,
            (ColourEncoding::PcsXyz, ColourEncoding::PcsXyz) => true,
            _ => false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ColorTransform {
    ops: Vec<ColorTransformOp>,
}

impl ColorTransform {
    pub fn new(
        from: &ColorEncodingWithProfile,
        to: &ColorEncodingWithProfile,
        oim: &OpsinInverseMatrix,
        intensity_target: f32,
    ) -> Self {
        if from.is_equivalent(to) {
            return Self { ops: Vec::new() };
        }

        let mut ops = Vec::new();

        let connecting_illuminant = match from.encoding {
            ColourEncoding::Enum(EnumColourEncoding { colour_space: ColourSpace::Xyb, .. }) => {
                let inv_mat = oim.inv_mat;
                #[rustfmt::skip]
                let matrix = [
                    inv_mat[0][0], inv_mat[0][1], inv_mat[0][2],
                    inv_mat[1][0], inv_mat[1][1], inv_mat[1][2],
                    inv_mat[2][0], inv_mat[2][1], inv_mat[2][2],
                ];
                ops.push(ColorTransformOp::XybToMixedLms { opsin_bias: oim.opsin_bias, intensity_target });
                ops.push(ColorTransformOp::Matrix(matrix));
                // result: RGB; D65 illuminant, sRGB primaries, linear tf
                ops.push(ColorTransformOp::Matrix(srgb_to_xyz_mat()));
                ILLUMINANT_D65
            },
            ColourEncoding::Enum(EnumColourEncoding { colour_space: ColourSpace::Rgb, white_point, primaries, tf, .. }) => {
                // TODO: address rendering intent
                let illuminant = white_point.as_chromaticity();
                let mat = crate::ciexyz::primaries_to_xyz_mat(primaries.as_chromaticity(), illuminant);
                let luminances = [mat[3], mat[4], mat[5]];
                ops.push(ColorTransformOp::TransferFunction {
                    tf,
                    hdr_params: HdrParams {
                        luminances,
                        intensity_target,
                    },
                    inverse: true,
                });
                ops.push(ColorTransformOp::Matrix(mat));
                illuminant
            },
            ColourEncoding::Enum(EnumColourEncoding { colour_space: ColourSpace::Grey, white_point, tf, .. }) => {
                // TODO: address rendering intent
                let illuminant = white_point.as_chromaticity();
                // primaries don't matter for grayscale
                let mat = crate::ciexyz::primaries_to_xyz_mat(PRIMARIES_SRGB, illuminant);
                let luminances = [mat[3], mat[4], mat[5]];
                ops.push(ColorTransformOp::TransferFunction {
                    tf,
                    hdr_params: HdrParams {
                        luminances,
                        intensity_target,
                    },
                    inverse: true,
                });
                ops.push(ColorTransformOp::Matrix(mat));
                illuminant
            },
            ColourEncoding::Enum(ref e) => todo!("Unsupported input enum colorspace {:?}", e),
            ColourEncoding::IccProfile(_) => {
                ops.push(ColorTransformOp::IccToIcc {
                    inputs: 0,
                    outputs: 3,
                    from: from.icc_profile.clone(),
                    to: nciexyz_icc_profile(ILLUMINANT_D50),
                });
                ILLUMINANT_D50
            },
            ColourEncoding::PcsXyz => {
                ILLUMINANT_D50
            },
        };

        // CIEXYZ adapted to `connecting_illuminant`

        match to.encoding {
            ColourEncoding::Enum(EnumColourEncoding { colour_space: ColourSpace::Rgb, white_point, primaries, tf, .. }) => {
                // TODO: address rendering intent
                let illuminant = white_point.as_chromaticity();
                let mat = crate::ciexyz::primaries_to_xyz_mat(primaries.as_chromaticity(), illuminant);
                let luminances = [mat[3], mat[4], mat[5]];
                let mat = crate::ciexyz::xyz_to_primaries_mat(primaries.as_chromaticity(), illuminant);
                ops.push(ColorTransformOp::Matrix(adapt_mat(connecting_illuminant, illuminant)));
                ops.push(ColorTransformOp::Matrix(mat));
                ops.push(ColorTransformOp::TransferFunction {
                    tf,
                    hdr_params: HdrParams {
                        luminances,
                        intensity_target,
                    },
                    inverse: false,
                });
            },
            ColourEncoding::Enum(EnumColourEncoding { colour_space: ColourSpace::Grey, white_point, tf, .. }) => {
                // TODO: address rendering intent
                let illuminant = white_point.as_chromaticity();
                // primaries don't matter for grayscale
                let mat = crate::ciexyz::primaries_to_xyz_mat(PRIMARIES_SRGB, illuminant);
                let luminances = [mat[3], mat[4], mat[5]];
                let mat = crate::ciexyz::xyz_to_primaries_mat(PRIMARIES_SRGB, illuminant);
                ops.push(ColorTransformOp::Matrix(adapt_mat(connecting_illuminant, illuminant)));
                ops.push(ColorTransformOp::Matrix(mat));
                ops.push(ColorTransformOp::TransferFunction {
                    tf,
                    hdr_params: HdrParams {
                        luminances,
                        intensity_target,
                    },
                    inverse: true,
                });
            },
            ColourEncoding::Enum(ref e) => todo!("Unsupported output enum colorspace {:?}", e),
            ColourEncoding::IccProfile(_) => {
                ops.push(ColorTransformOp::IccToIcc {
                    inputs: 3,
                    outputs: 0,
                    from: nciexyz_icc_profile(connecting_illuminant),
                    to: to.icc_profile.clone(),
                });
            },
            ColourEncoding::PcsXyz => {
                if connecting_illuminant != ILLUMINANT_D50 {
                    ops.push(ColorTransformOp::Matrix(
                        adapt_mat(connecting_illuminant, ILLUMINANT_D50)
                    ));
                }
            },
        }

        Self { ops }
    }

    pub fn xyb_to_enum(encoding: &EnumColourEncoding, oim: &OpsinInverseMatrix, intensity_target: f32) -> Self {
        Self::new(
            &ColorEncodingWithProfile::new(ColourEncoding::Enum(EnumColourEncoding::xyb())),
            &ColorEncodingWithProfile::new(ColourEncoding::Enum(encoding.clone())),
            oim,
            intensity_target,
        )
    }

    pub fn run<Cms: ColorManagementSystem + ?Sized>(&self, channels: &mut [&mut [f32]], cms: &Cms) -> Result<usize, crate::Error> {
        let mut num_channels = channels.len();
        for op in &self.ops {
            num_channels = op.run(channels, cms)?;
        }
        Ok(num_channels)
    }
}

#[derive(Debug, Clone)]
enum ColorTransformOp {
    XybToMixedLms {
        opsin_bias: [f32; 3],
        intensity_target: f32,
    },
    Matrix([f32; 9]),
    Bias([f32; 3]),
    TransferFunction {
        tf: TransferFunction,
        hdr_params: HdrParams,
        inverse: bool,
    },
    IccToIcc {
        inputs: usize,
        outputs: usize,
        from: Vec<u8>,
        to: Vec<u8>,
    },
}

impl ColorTransformOp {
    #[inline]
    fn inputs(&self) -> Option<usize> {
        match *self {
            ColorTransformOp::XybToMixedLms { .. } | ColorTransformOp::Matrix(_) | ColorTransformOp::Bias(_) => Some(3),
            ColorTransformOp::TransferFunction { tf: TransferFunction::Hlg, .. } => Some(3),
            ColorTransformOp::TransferFunction { .. }  => None,
            ColorTransformOp::IccToIcc { inputs: 0, .. } => None,
            ColorTransformOp::IccToIcc { inputs, .. } => Some(inputs),
        }
    }

    #[inline]
    fn outputs(&self) -> Option<usize> {
        match *self {
            ColorTransformOp::XybToMixedLms { .. } | ColorTransformOp::Matrix(_) | ColorTransformOp::Bias(_) => Some(3),
            ColorTransformOp::TransferFunction { tf: TransferFunction::Hlg, .. } => Some(3),
            ColorTransformOp::TransferFunction { .. }  => None,
            ColorTransformOp::IccToIcc { outputs: 0, .. } => None,
            ColorTransformOp::IccToIcc { outputs, .. } => Some(outputs),
        }
    }

    fn run<Cms: ColorManagementSystem + ?Sized>(
        &self,
        channels: &mut [&mut [f32]],
        cms: &Cms,
    ) -> Result<usize, crate::Error> {
        let channel_count = channels.len();
        if let Some(inputs) = self.inputs() {
            assert!(channel_count >= inputs);
        }
        if let Some(outputs) = self.outputs() {
            assert!(channel_count >= outputs);
        }

        Ok(match self {
            Self::XybToMixedLms { opsin_bias, intensity_target } => {
                let [x, y, b, ..] = channels else { unreachable!() };
                let xyb = [&mut **x, &mut **y, &mut **b];
                crate::xyb::run(xyb, *opsin_bias, *intensity_target);
                3
            },
            Self::Matrix(matrix) => {
                let [a, b, c, ..] = channels else { unreachable!() };
                for ((a, b), c) in a.iter_mut().zip(&mut **b).zip(&mut **c) {
                    let result = crate::ciexyz::matmul3vec(matrix, &[*a, *b, *c]);
                    *a = result[0];
                    *b = result[1];
                    *c = result[2];
                }
                3
            },
            Self::Bias(bias) => {
                let [a, b, c, ..] = channels else { unreachable!() };
                for ((a, b), c) in a.iter_mut().zip(&mut **b).zip(&mut **c) {
                    *a += bias[0];
                    *b += bias[1];
                    *c += bias[2];
                }
                3
            },
            Self::TransferFunction { tf, hdr_params, inverse: false } => {
                apply_transfer_function(channels, *tf, *hdr_params);
                channels.len()
            },
            Self::TransferFunction { tf, hdr_params, inverse: true } => {
                apply_inverse_transfer_function(channels, *tf, *hdr_params);
                channels.len()
            },
            Self::IccToIcc { from, to, .. } => {
                cms.transform(from, to, RenderingIntent::Relative, channels)?
            },
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct HdrParams {
    luminances: [f32; 3],
    intensity_target: f32,
}

#[inline]
pub(crate) fn srgb_to_xyz_mat() -> [f32; 9] {
    primaries_to_xyz_mat(PRIMARIES_SRGB, ILLUMINANT_D65)
}

fn apply_transfer_function(
    channels: &mut [&mut [f32]],
    tf: TransferFunction,
    hdr_params: HdrParams,
) {
    match tf {
        TransferFunction::Gamma(gamma) => {
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
        TransferFunction::Gamma(gamma) => {
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
            todo!()
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

            tf::hlg_oo([r, g, b], luminances, intensity_target);
            tf::hlg_to_linear(r);
            tf::hlg_to_linear(g);
            tf::hlg_to_linear(b);
        }
    }
}
