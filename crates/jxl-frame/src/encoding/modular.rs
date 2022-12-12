use std::io::Read;

use jxl_bitstream::{define_bundle, read_bits, Bitstream, Bundle};

use crate::{Result, frame_data::GlobalModular};

mod ma;
mod predictor;
mod transform;
pub use ma::{MaConfig, MaContext};

#[derive(Debug)]
pub struct Modular {
    inner: Option<ModularData>,
}

#[derive(Debug)]
struct ModularData {
    header: ModularHeader,
    ma_ctx: ma::MaContext,
    channels: ModularChannels,
}

#[derive(Debug, Clone)]
pub struct ModularParams<'a> {
    pub width: u32,
    pub height: u32,
    pub channel_shifts: Vec<ChannelShift>,
    pub ma_config: Option<&'a MaConfig>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelShift {
    JpegUpsampling(bool, bool),
    Shifts(u32),
    NoShift,
}

impl ChannelShift {
    pub fn from_upsampling_factor(upsampling: u32) -> ChannelShift {
        Self::Shifts(upsampling.next_power_of_two().trailing_zeros())
    }

    pub fn from_upsampling_factor_and_shift(upsampling: u32, dim_shift: u32) -> ChannelShift {
        Self::Shifts(upsampling.next_power_of_two().trailing_zeros() + dim_shift)
    }

    pub fn from_jpeg_upsampling(jpeg_upsampling: u32) -> Self {
        let (h, v) = match jpeg_upsampling {
            0 => (false, false),
            1 => (true, true),
            2 => (true, false),
            3 => (false, true),
            _ => panic!("Invalid jpeg_upsampling value of {}", jpeg_upsampling),
        };
        Self::JpegUpsampling(h, v)
    }

    fn hshift(&self) -> i32 {
        match self {
            Self::JpegUpsampling(h, _) => *h as i32,
            Self::Shifts(s) => *s as i32,
            Self::NoShift => -1,
        }
    }

    fn vshift(&self) -> i32 {
        match self {
            Self::JpegUpsampling(_, v) => *v as i32,
            Self::Shifts(s) => *s as i32,
            Self::NoShift => -1,
        }
    }
}

impl<'a> ModularParams<'a> {
    pub fn new(
        width: u32,
        height: u32,
        channel_shifts: Vec<ChannelShift>,
        ma_config: Option<&'a MaConfig>,
    ) -> Self {
        Self { width, height, channel_shifts, ma_config }
    }
}

impl Bundle<ModularParams<'_>> for Modular {
    type Error = crate::Error;

    fn parse<R: Read>(
        bitstream: &mut Bitstream<R>,
        params: ModularParams<'_>,
    ) -> Result<Self> {
        let inner = if params.channel_shifts.is_empty() {
            None
        } else {
            Some(read_bits!(bitstream, Bundle(ModularData), params)?)
        };
        Ok(Self { inner })
    }
}

impl Bundle<ModularParams<'_>> for ModularData {
    type Error = crate::Error;

    fn parse<R: Read>(
        bitstream: &mut Bitstream<R>,
        params: ModularParams<'_>,
    ) -> Result<Self> {
        let mut header = read_bits!(bitstream, Bundle(ModularHeader))?;
        let ma_ctx = if header.use_global_tree {
            params.ma_config.ok_or(crate::Error::GlobalMaTreeNotAvailable)?.make_context()
        } else {
            read_bits!(bitstream, Bundle(ma::MaConfig))?.into()
        };

        let mut channels = ModularChannels::from_params(&params);
        for transform in &mut header.transform {
            transform.or_default(&mut channels);
            transform.transform_channel_info(&mut channels)?;
        }

        Ok(Self {
            header,
            ma_ctx,
            channels,
        })
    }
}

define_bundle! {
    #[derive(Debug)]
    pub struct ModularHeader error(crate::Error) {
        use_global_tree: ty(Bool),
        wp_params: ty(Bundle(predictor::WpHeader)),
        nb_transforms: ty(U32(0, 1, 2 + u(4), 18 + u(8))),
        transform: ty(Vec[Bundle(transform::TransformInfo)]; nb_transforms),
    }
}

#[derive(Debug)]
struct ModularChannels {
    info: Vec<ModularChannelInfo>,
    nb_meta_channels: u32,
}

impl ModularChannels {
    fn from_params(params: &ModularParams<'_>) -> Self {
        let width = params.width;
        let height = params.height;
        let info = params.channel_shifts.iter()
            .map(|&shift| ModularChannelInfo::new(width, height, shift))
            .collect();
        Self {
            info,
            nb_meta_channels: 0,
        }
    }
}

#[derive(Debug, Clone)]
struct ModularChannelInfo {
    width: u32,
    height: u32,
    hshift: i32,
    vshift: i32,
}

impl ModularChannelInfo {
    fn new(width: u32, height: u32, shift: ChannelShift) -> Self {
        Self {
            width,
            height,
            hshift: shift.hshift(),
            vshift: shift.vshift(),
        }
    }
}
