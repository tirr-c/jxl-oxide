use crate::{header::Encoding, Result};
use jxl_bitstream::{Bitstream, Bundle};

#[derive(Debug, Clone)]
pub enum Gabor {
    Disabled,
    Enabled([[f32; 2]; 3]),
}

impl Default for Gabor {
    fn default() -> Self {
        Self::Enabled([[0.115169525, 0.061248592]; 3])
    }
}

impl<Ctx> Bundle<Ctx> for Gabor {
    type Error = crate::Error;

    fn parse(bitstream: &mut Bitstream, _ctx: Ctx) -> Result<Self> {
        let gab_enabled = bitstream.read_bool()?;
        if !gab_enabled {
            return Ok(Self::Disabled);
        }

        let custom = bitstream.read_bool()?;
        if !custom {
            return Ok(Self::default());
        }

        let mut weights = [[0.0f32; 2]; 3];
        for chan_weight in &mut weights {
            for weight in &mut *chan_weight {
                *weight = bitstream.read_f16_as_f32()?;
            }
            if f32::abs(1.0 + (chan_weight[0] + chan_weight[1]) * 4.0) < f32::EPSILON {
                return Err(jxl_bitstream::Error::ValidationFailed(
                    "Gaborish weights lead to near 0 unnormalized kernel",
                )
                .into());
            }
        }
        Ok(Self::Enabled(weights))
    }
}

impl Gabor {
    pub fn enabled(&self) -> bool {
        matches!(self, Self::Enabled(_))
    }
}

#[derive(Debug, Clone)]
pub enum EdgePreservingFilter {
    Disabled,
    Enabled(EpfParams),
}

#[derive(Debug, Clone)]
pub struct EpfParams {
    pub iters: u32,
    pub sharp_lut: [f32; 8],
    pub channel_scale: [f32; 3],
    pub sigma: EpfSigma,
    pub sigma_for_modular: f32,
}

impl EdgePreservingFilter {
    pub fn enabled(&self) -> bool {
        matches!(self, Self::Enabled { .. })
    }
}

impl Default for EdgePreservingFilter {
    fn default() -> Self {
        Self::Enabled(EpfParams::default())
    }
}

const EPF_SHARP_LUT_DEFAULT: [f32; 8] = [
    0.0,
    1.0 / 7.0,
    2.0 / 7.0,
    3.0 / 7.0,
    4.0 / 7.0,
    5.0 / 7.0,
    6.0 / 7.0,
    1.0,
];
const EPF_CHANNEL_SCALE_DEFAULT: [f32; 3] = [40.0, 5.0, 3.5];

impl Default for EpfParams {
    fn default() -> Self {
        Self {
            iters: 2,
            sharp_lut: EPF_SHARP_LUT_DEFAULT,
            channel_scale: EPF_CHANNEL_SCALE_DEFAULT,
            sigma: Default::default(),
            sigma_for_modular: 1.0,
        }
    }
}

impl Bundle<Encoding> for EdgePreservingFilter {
    type Error = crate::Error;

    fn parse(bitstream: &mut Bitstream, encoding: Encoding) -> Result<Self> {
        let iters = bitstream.read_bits(2)?;
        if iters == 0 {
            return Ok(Self::Disabled);
        }

        let sharp_custom = if encoding == Encoding::VarDct {
            bitstream.read_bool()?
        } else {
            false
        };
        let sharp_lut = if sharp_custom {
            let mut ret = [0.0; 8];
            for out in &mut ret {
                *out = bitstream.read_f16_as_f32()?;
            }
            ret
        } else {
            EPF_SHARP_LUT_DEFAULT
        };

        let weight_custom = bitstream.read_bool()?;
        let channel_scale = if weight_custom {
            let mut ret = [0.0; 3];
            for out in &mut ret {
                *out = bitstream.read_f16_as_f32()?;
            }
            bitstream.read_bits(32)?; // ignored
            ret
        } else {
            EPF_CHANNEL_SCALE_DEFAULT
        };

        let sigma_custom = bitstream.read_bool()?;
        let sigma = if sigma_custom {
            EpfSigma::parse(bitstream, encoding)?
        } else {
            EpfSigma::default()
        };

        let sigma_for_modular = if encoding == Encoding::Modular {
            let out = bitstream.read_f16_as_f32()?;
            if out < f32::EPSILON {
                tracing::warn!("EPF: sigma for modular is too small");
            }
            out
        } else {
            1.0
        };

        Ok(Self::Enabled(EpfParams {
            iters,
            sharp_lut,
            channel_scale,
            sigma,
            sigma_for_modular,
        }))
    }
}

#[derive(Debug, Clone)]
pub struct EpfSigma {
    pub quant_mul: f32,
    pub pass0_sigma_scale: f32,
    pub pass2_sigma_scale: f32,
    pub border_sad_mul: f32,
}

impl Default for EpfSigma {
    fn default() -> Self {
        Self {
            quant_mul: 0.46,
            pass0_sigma_scale: 0.9,
            pass2_sigma_scale: 6.5,
            border_sad_mul: 2.0 / 3.0,
        }
    }
}

impl Bundle<Encoding> for EpfSigma {
    type Error = crate::Error;

    fn parse(bitstream: &mut Bitstream, encoding: Encoding) -> Result<Self> {
        Ok(Self {
            quant_mul: if encoding == Encoding::VarDct {
                bitstream.read_f16_as_f32()?
            } else {
                0.46
            },
            pass0_sigma_scale: bitstream.read_f16_as_f32()?,
            pass2_sigma_scale: bitstream.read_f16_as_f32()?,
            border_sad_mul: bitstream.read_f16_as_f32()?,
        })
    }
}
