use jxl_oxide::color::{
    ColourSpace, Customxy, EnumColourEncoding, Primaries, RenderingIntent, TransferFunction,
    WhitePoint,
};

#[derive(Debug, Clone, Default)]
struct ColorspaceSpec {
    ty: Option<ColourSpace>,
    white_point: Option<WhitePoint>,
    gamut: Option<Primaries>,
    tf: Option<TransferFunction>,
    intent: Option<RenderingIntent>,
}

#[derive(Debug)]
pub struct ColorspaceSpecParseError(std::borrow::Cow<'static, str>);

impl From<&'static str> for ColorspaceSpecParseError {
    fn from(value: &'static str) -> Self {
        Self(value.into())
    }
}

impl From<String> for ColorspaceSpecParseError {
    fn from(value: String) -> Self {
        Self(value.into())
    }
}

impl std::fmt::Display for ColorspaceSpecParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for ColorspaceSpecParseError {}

impl ColorspaceSpec {
    fn new() -> Self {
        Self::default()
    }

    fn from_preset(preset: &str) -> Result<Self, ColorspaceSpecParseError> {
        let preset_lowercase = preset.to_lowercase();
        Ok(match &*preset_lowercase {
            "srgb" => ColorspaceSpec {
                ty: Some(ColourSpace::Rgb),
                white_point: Some(WhitePoint::D65),
                gamut: Some(Primaries::Srgb),
                tf: Some(TransferFunction::Srgb),
                intent: Some(RenderingIntent::Relative),
            },
            "display_p3" => ColorspaceSpec {
                ty: Some(ColourSpace::Rgb),
                white_point: Some(WhitePoint::D65),
                gamut: Some(Primaries::P3),
                tf: Some(TransferFunction::Srgb),
                intent: Some(RenderingIntent::Relative),
            },
            "rec2020" | "rec.2020" => ColorspaceSpec {
                ty: Some(ColourSpace::Rgb),
                white_point: Some(WhitePoint::D65),
                gamut: Some(Primaries::Bt2100),
                tf: Some(TransferFunction::Bt709),
                intent: Some(RenderingIntent::Relative),
            },
            "rec2100" | "rec.2100" => ColorspaceSpec {
                ty: Some(ColourSpace::Rgb),
                white_point: Some(WhitePoint::D65),
                gamut: Some(Primaries::Bt2100),
                tf: None,
                intent: Some(RenderingIntent::Relative),
            },
            _ => {
                return Err(format!("unknown preset `{preset}`").into());
            }
        })
    }

    fn add_param(&mut self, param: &str) -> Result<(), ColorspaceSpecParseError> {
        let (name, value) = param
            .split_once('=')
            .ok_or_else(|| format!("`{param}` is not a parameter spec"))?;
        let name_lowercase = name.to_ascii_lowercase();
        let value_lowercase = value.to_ascii_lowercase();
        match &*name_lowercase {
            "type" | "color_space" => match &*value_lowercase {
                "rgb" => self.ty = Some(ColourSpace::Rgb),
                "xyb" => {
                    let mut invalid_option = None;
                    if self.white_point.is_some() {
                        invalid_option = Some("white point");
                    } else if self.gamut.is_some() {
                        invalid_option = Some("color gamut");
                    } else if self.tf.is_some() {
                        invalid_option = Some("transfer function");
                    }
                    if let Some(invalid_option) = invalid_option {
                        return Err(format!(
                            "cannot set {invalid_option} when color space type is XYB"
                        )
                        .into());
                    }

                    self.ty = Some(ColourSpace::Xyb);
                }
                "gray" | "grey" | "grayscale" | "greyscale" => {
                    if self.gamut.is_some() {
                        return Err(
                            "cannot set color gamut when color space type is Grayscale".into()
                        );
                    }

                    self.ty = Some(ColourSpace::Grey);
                }
                _ => return Err(format!("unknown color space type `{value}`").into()),
            },
            "white_point" | "wp" => {
                if let Some(ColourSpace::Xyb) = self.ty {
                    return Err("cannot set white point if color space type is XYB".into());
                }

                let wp = match &*value_lowercase {
                    "d65" => WhitePoint::D65,
                    "dci" => WhitePoint::Dci,
                    "e" => WhitePoint::E,
                    "d50" => WhitePoint::Custom(Customxy {
                        x: 345669,
                        y: 358496,
                    }),
                    _ => return Err(format!("invalid white point `{value}`").into()),
                };
                self.white_point = Some(wp);
            }
            "gamut" | "primaries" => {
                if let Some(ColourSpace::Xyb) = self.ty {
                    return Err("cannot set white point if color space type is XYB".into());
                }
                if let Some(ColourSpace::Grey) = self.ty {
                    return Err("cannot set white point if color space type is Grayscale".into());
                }

                let gamut = match &*value_lowercase {
                    "srgb" | "bt709" => Primaries::Srgb,
                    "p3" | "dci" => Primaries::P3,
                    "2020" | "bt2020" | "bt.2020" | "rec2020" | "rec.2020" | "2100" | "bt2100"
                    | "bt.2100" | "rec2100" | "rec.2100" => Primaries::Bt2100,
                    _ => return Err(format!("invalid gamut `{value}`").into()),
                };
                self.gamut = Some(gamut);
            }
            "tf" | "transfer_function" | "curve" | "tone_curve" => {
                if let Some(ColourSpace::Xyb) = self.ty {
                    return Err("cannot set transfer function if color space type is XYB".into());
                }

                let tf = match &*value_lowercase {
                    "srgb" => TransferFunction::Srgb,
                    "bt709" | "bt.709" | "709" => TransferFunction::Bt709,
                    "dci" => TransferFunction::Dci,
                    "pq" | "perceptual_quantizer" => TransferFunction::Pq,
                    "hlg" | "hybrid_log_gamma" => TransferFunction::Hlg,
                    "linear" => TransferFunction::Linear,
                    gamma => {
                        let gamma = gamma
                            .parse::<f32>()
                            .map_err(|_| format!("invalid transfer function `{value}`"))?;
                        if !gamma.is_finite() || gamma < 1f32 {
                            return Err(format!("gamma of {gamma} is invalid").into());
                        }

                        TransferFunction::Gamma {
                            g: (gamma * 1e7 + 0.5) as u32,
                            inverted: false,
                        }
                    }
                };
                self.tf = Some(tf);
            }
            "intent" | "rendering_intent" => {
                let intent = match &*value_lowercase {
                    "relative" | "rel" | "relative_colorimetric" => RenderingIntent::Relative,
                    "perceptual" | "per" => RenderingIntent::Perceptual,
                    "saturation" | "sat" => RenderingIntent::Saturation,
                    "absolute" | "abs" | "absolute_colorimetric" => RenderingIntent::Absolute,
                    _ => return Err(format!("invalid rendering intent `{value}`").into()),
                };
                self.intent = Some(intent);
            }
            _ => return Err(format!("invalid parameter `{name}`").into()),
        }

        Ok(())
    }
}

pub fn parse_color_encoding(val: &str) -> Result<EnumColourEncoding, ColorspaceSpecParseError> {
    let mut params = val.split(',');

    let first = params.next().ok_or("parameters are required")?;
    let mut spec = ColorspaceSpec::from_preset(first).or_else(|preset_err| {
        let mut spec = ColorspaceSpec::new();
        spec.add_param(first).map(|_| spec).map_err(|param_err| {
            if first.contains('=') {
                param_err
            } else {
                preset_err
            }
        })
    })?;

    for param in params {
        spec.add_param(param)?;
    }

    let ty = if let Some(ty) = spec.ty {
        ty
    } else if spec.white_point.is_some() && spec.gamut.is_some() && spec.tf.is_some() {
        ColourSpace::Rgb
    } else {
        return Err("color space type is required".into());
    };

    Ok(match ty {
        ColourSpace::Rgb => {
            let white_point = spec.white_point.ok_or("white point is required")?;
            let primaries = spec.gamut.ok_or("color gamut is required")?;
            let tf = spec.tf.ok_or("transfer function is required")?;
            let rendering_intent = spec.intent.ok_or("rendering intent is required")?;
            EnumColourEncoding {
                colour_space: ColourSpace::Rgb,
                white_point,
                primaries,
                tf,
                rendering_intent,
            }
        }
        ColourSpace::Grey => {
            let white_point = spec.white_point.ok_or("white point is required")?;
            let tf = spec.tf.ok_or("transfer function is required")?;
            let rendering_intent = spec.intent.ok_or("rendering intent is required")?;
            EnumColourEncoding {
                colour_space: ColourSpace::Grey,
                white_point,
                primaries: Primaries::Srgb,
                tf,
                rendering_intent,
            }
        }
        ColourSpace::Xyb => {
            let rendering_intent = spec.intent.ok_or("rendering intent is required")?;
            EnumColourEncoding {
                colour_space: ColourSpace::Xyb,
                white_point: WhitePoint::D65,
                primaries: Primaries::Srgb,
                tf: TransferFunction::Srgb,
                rendering_intent,
            }
        }
        _ => unreachable!(),
    })
}
