use std::io::Read;
use jxl_bitstream::{Bitstream, Bundle, Result as BitstreamResult};
use crate::header::Encoding;

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
    fn parse<R: Read>(bitstream: &mut Bitstream<R>, _ctx: Ctx) -> BitstreamResult<Self> {
        let custom = bitstream.read_bool()?;
        if !custom {
            return Ok(Self::default());
        }

        let mut weights = [[0.0f32; 2]; 3];
        for chan_weight in &mut weights {
            for weight in chan_weight {
                *weight = bitstream.read_f16_as_f32()?;
            }
        }
        Ok(Self::Enabled(weights))
    }
}

#[derive(Debug)]
pub enum EdgePreservingFilter {
    Disabled,
    Enabled {
        iters: u32,
        sharp_lut: [f32; 8],
        channel_scale: [f32; 3],
        sigma: EpfSigma,
        sigma_for_modular: f32,
    },
}

impl EdgePreservingFilter {
    const SHARP_LUT_DEFAULT: [f32; 8] = [0.0, 1.0 / 7.0, 2.0 / 7.0, 3.0 / 7.0, 4.0 / 7.0, 5.0 / 7.0, 6.0 / 7.0, 1.0];
    const CHANNEL_SCALE_DEFAULT: [f32; 3] = [40.0, 5.0, 3.5];
}

impl Default for EdgePreservingFilter {
    fn default() -> Self {
        Self::Enabled {
            iters: 2,
            sharp_lut: Self::SHARP_LUT_DEFAULT,
            channel_scale: Self::CHANNEL_SCALE_DEFAULT,
            sigma: Default::default(),
            sigma_for_modular: 1.0,
        }
    }
}

impl Bundle<Encoding> for EdgePreservingFilter {
    fn parse<R: Read>(bitstream: &mut Bitstream<R>, encoding: Encoding) -> BitstreamResult<Self> {
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
            Self::SHARP_LUT_DEFAULT
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
            Self::CHANNEL_SCALE_DEFAULT
        };

        let sigma_custom = bitstream.read_bool()?;
        let sigma = if sigma_custom {
            EpfSigma::parse(bitstream, encoding)?
        } else {
            EpfSigma::default()
        };

        let sigma_for_modular = if encoding == Encoding::Modular {
            bitstream.read_f16_as_f32()?
        } else {
            1.0
        };

        Ok(Self::Enabled {
            iters,
            sharp_lut,
            channel_scale,
            sigma,
            sigma_for_modular,
        })
    }
}

#[derive(Debug)]
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
    fn parse<R: Read>(bitstream: &mut Bitstream<R>, encoding: Encoding) -> BitstreamResult<Self> {
        Ok(Self {
            quant_mul: if encoding == Encoding::VarDct { bitstream.read_f16_as_f32()? } else { 0.46 },
            pass0_sigma_scale: bitstream.read_f16_as_f32()?,
            pass2_sigma_scale: bitstream.read_f16_as_f32()?,
            border_sad_mul: bitstream.read_f16_as_f32()?,
        })
    }
}
