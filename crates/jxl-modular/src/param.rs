use super::MaConfig;

#[derive(Debug, Clone)]
pub struct ModularParams<'a> {
    pub group_dim: u32,
    pub bit_depth: u32,
    pub channels: Vec<ModularChannelParams>,
    pub channel_mapping: Option<Vec<SubimageChannelInfo>>,
    pub ma_config: Option<&'a MaConfig>,
}

impl<'a> ModularParams<'a> {
    pub fn new(
        width: u32,
        height: u32,
        group_dim: u32,
        bit_depth: u32,
        channel_shifts: Vec<ChannelShift>,
        ma_config: Option<&'a MaConfig>,
    ) -> Self {
        let channels = channel_shifts
            .into_iter()
            .map(|shift| ModularChannelParams { width, height, shift })
            .collect();
        Self::with_channels(group_dim, bit_depth, channels, ma_config)
    }

    pub fn with_channels(
        group_dim: u32,
        bit_depth: u32,
        channels: Vec<ModularChannelParams>,
        ma_config: Option<&'a MaConfig>,
    ) -> Self {
        Self {
            group_dim,
            bit_depth,
            channels,
            channel_mapping: None,
            ma_config,
        }
    }

    pub fn base_size(&self) -> Option<(u32, u32)> {
        if self.channels.is_empty() {
            return Some((0, 0));
        }

        let bw = self.channels[0].width;
        let bh = self.channels[0].height;
        for &ModularChannelParams { width, height, .. } in &self.channels {
            if bw != width || bh != height {
                return None;
            }
        }

        Some((bw, bh))
    }
}

#[derive(Debug, Clone)]
pub struct ModularChannelParams {
    pub width: u32,
    pub height: u32,
    pub shift: ChannelShift,
}

impl ModularChannelParams {
    pub fn new(width: u32, height: u32, group_dim: u32) -> Self {
        Self {
            width,
            height,
            shift: ChannelShift::from_shift(0),
        }
    }

    pub fn jpeg(width: u32, height: u32, group_dim: u32, jpeg_upsampling: u32) -> Self {
        Self {
            width,
            height,
            shift: ChannelShift::from_jpeg_upsampling(jpeg_upsampling),
        }
    }

    pub fn with_shift(width: u32, height: u32, group_dim: u32, shift: ChannelShift) -> Self {
        Self {
            width,
            height,
            shift,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SubimageChannelInfo {
    pub channel_id: usize,
    pub base_x: u32,
    pub base_y: u32,
}

impl SubimageChannelInfo {
    pub fn new(channel_id: usize, base_x: u32, base_y: u32) -> Self {
        SubimageChannelInfo { channel_id, base_x, base_y }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelShift {
    JpegUpsampling(bool, bool),
    Shifts(u32),
    Raw(i32, i32),
}

impl ChannelShift {
    pub fn from_shift(shift: u32) -> ChannelShift {
        Self::Shifts(shift)
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

    pub fn hshift(&self) -> i32 {
        match self {
            Self::JpegUpsampling(h, _) => *h as i32,
            Self::Shifts(s) => *s as i32,
            Self::Raw(h, _) => *h,
        }
    }

    pub fn vshift(&self) -> i32 {
        match self {
            Self::JpegUpsampling(_, v) => *v as i32,
            Self::Shifts(s) => *s as i32,
            Self::Raw(_, v) => *v,
        }
    }
}
