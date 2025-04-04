use jxl_grid::AllocTracker;

use super::MaConfig;

#[derive(Debug, Clone)]
pub struct ModularParams<'a, 'b> {
    pub group_dim: u32,
    pub bit_depth: u32,
    pub channels: Vec<ModularChannelParams>,
    pub ma_config: Option<&'a MaConfig>,
    pub tracker: Option<&'b AllocTracker>,
    pub narrow_buffer: bool,
}

impl<'a, 'b> ModularParams<'a, 'b> {
    pub fn new(
        width: u32,
        height: u32,
        group_dim: u32,
        bit_depth: u32,
        channel_shifts: Vec<ChannelShift>,
        ma_config: Option<&'a MaConfig>,
        tracker: Option<&'b AllocTracker>,
    ) -> Self {
        let channels = channel_shifts
            .into_iter()
            .map(|shift| ModularChannelParams {
                width,
                height,
                shift,
            })
            .collect();
        Self::with_channels(group_dim, bit_depth, channels, ma_config, tracker)
    }

    pub fn with_channels(
        group_dim: u32,
        bit_depth: u32,
        channels: Vec<ModularChannelParams>,
        ma_config: Option<&'a MaConfig>,
        tracker: Option<&'b AllocTracker>,
    ) -> Self {
        Self {
            group_dim,
            bit_depth,
            channels,
            ma_config,
            tracker,
            narrow_buffer: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModularChannelParams {
    pub width: u32,
    pub height: u32,
    pub shift: ChannelShift,
}

impl ModularChannelParams {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            shift: ChannelShift::from_shift(0),
        }
    }

    pub fn jpeg(width: u32, height: u32, jpeg_upsampling: [u32; 3], idx: usize) -> Self {
        Self {
            width,
            height,
            shift: ChannelShift::from_jpeg_upsampling(jpeg_upsampling, idx),
        }
    }

    pub fn with_shift(width: u32, height: u32, shift: ChannelShift) -> Self {
        Self {
            width,
            height,
            shift,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelShift {
    JpegUpsampling {
        has_h_subsample: bool,
        h_subsample: bool,
        has_v_subsample: bool,
        v_subsample: bool,
    },
    Shifts(u32),
    Raw(i32, i32),
}

impl ChannelShift {
    #[inline]
    pub fn from_shift(shift: u32) -> ChannelShift {
        Self::Shifts(shift)
    }

    pub fn from_jpeg_upsampling(jpeg_upsampling: [u32; 3], idx: usize) -> Self {
        let upsampling = jpeg_upsampling[idx];
        let hscale = jpeg_upsampling.into_iter().any(|v| v == 1 || v == 2);
        let vscale = jpeg_upsampling.into_iter().any(|v| v == 1 || v == 3);
        let (h, v) = match upsampling {
            0 => (hscale, vscale),
            1 => (false, false),
            2 => (false, vscale),
            3 => (hscale, false),
            _ => panic!("Invalid jpeg_upsampling value of {}", upsampling),
        };
        Self::JpegUpsampling {
            has_h_subsample: hscale,
            h_subsample: h,
            has_v_subsample: vscale,
            v_subsample: v,
        }
    }

    #[inline]
    pub fn hshift(&self) -> i32 {
        match self {
            Self::JpegUpsampling { h_subsample, .. } => *h_subsample as i32,
            Self::Shifts(s) => *s as i32,
            Self::Raw(h, _) => *h,
        }
    }

    #[inline]
    pub fn vshift(&self) -> i32 {
        match self {
            Self::JpegUpsampling { v_subsample, .. } => *v_subsample as i32,
            Self::Shifts(s) => *s as i32,
            Self::Raw(_, v) => *v,
        }
    }

    pub fn shift_size(&self, (width, height): (u32, u32)) -> (u32, u32) {
        match *self {
            Self::JpegUpsampling {
                has_h_subsample,
                has_v_subsample,
                h_subsample,
                v_subsample,
            } => {
                let width = if has_h_subsample {
                    let size = width.div_ceil(2);
                    if h_subsample {
                        size
                    } else {
                        size * 2
                    }
                } else {
                    width
                };
                let height = if has_v_subsample {
                    let size = height.div_ceil(2);
                    if v_subsample {
                        size
                    } else {
                        size * 2
                    }
                } else {
                    height
                };
                (width, height)
            }
            Self::Shifts(s) => {
                let add = (1 << s) - 1;
                ((width + add) >> s, (height + add) >> s)
            }
            Self::Raw(h, v) => {
                let add_h = (1 << h) - 1;
                let add_v = (1 << v) - 1;
                ((width + add_h) >> h, (height + add_v) >> v)
            }
        }
    }
}
