use std::io::Read;

use jxl_bitstream::{define_bundle, read_bits, Bitstream, Bundle};

use crate::Result;

mod image;
mod ma;
mod predictor;
mod transform;
pub use image::Image;
pub use ma::{MaConfig, MaContext};

#[derive(Debug)]
pub struct Modular {
    inner: Option<ModularData>,
}

#[derive(Debug)]
struct ModularData {
    base_width: u32,
    base_height: u32,
    group_dim: u32,
    header: ModularHeader,
    ma_ctx: ma::MaContext,
    channels: ModularChannels,
    subimage_channel_mapping: Option<Vec<SubimageChannelInfo>>,
    image: Image,
}

#[derive(Debug, Clone)]
pub struct ModularParams<'a> {
    pub width: u32,
    pub height: u32,
    pub group_dim: u32,
    pub bit_depth: u32,
    pub channel_shifts: Vec<ChannelShift>,
    pub ma_config: Option<&'a MaConfig>,
    channel_mapping: Option<Vec<SubimageChannelInfo>>,
}

#[derive(Debug, Clone)]
struct SubimageChannelInfo {
    channel_id: usize,
    base_x: u32,
    base_y: u32,
}

impl SubimageChannelInfo {
    fn new(channel_id: usize, base_x: u32, base_y: u32) -> Self {
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

    fn hshift(&self) -> i32 {
        match self {
            Self::JpegUpsampling(h, _) => *h as i32,
            Self::Shifts(s) => *s as i32,
            Self::Raw(h, _) => *h,
        }
    }

    fn vshift(&self) -> i32 {
        match self {
            Self::JpegUpsampling(_, v) => *v as i32,
            Self::Shifts(s) => *s as i32,
            Self::Raw(_, v) => *v,
        }
    }
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
        Self { width, height, group_dim, bit_depth, channel_shifts, ma_config, channel_mapping: None }
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

impl Modular {
    pub fn decode_image_gmodular<R: Read>(&mut self, bitstream: &mut Bitstream<R>) -> Result<()> {
        let Some(image) = &mut self.inner else { return Ok(()); };
        let wp_header = &image.header.wp_params;
        let ma_ctx = &mut image.ma_ctx;
        let (mut subimage, channel_mapping) = image.image.for_global_modular();
        subimage.decode_channels(bitstream, 0, wp_header, ma_ctx)?;
        image.image.copy_from_image(subimage, &channel_mapping);
        Ok(())
    }

    pub fn decode_image<R: Read>(&mut self, bitstream: &mut Bitstream<R>, stream_index: u32) -> Result<()> {
        let Some(image) = &mut self.inner else { return Ok(()); };
        let wp_header = &image.header.wp_params;
        let ma_ctx = &mut image.ma_ctx;
        image.image.decode_channels(bitstream, stream_index, wp_header, ma_ctx)
    }

    pub fn inverse_transform(&mut self) {
        let Some(image) = &mut self.inner else { return; };
        for transform in image.header.transform.iter().rev() {
            transform.inverse(&mut image.image);
        }
    }

    pub fn make_subimage_params_lf_group<'a>(&self, global_ma_config: Option<&'a MaConfig>, lf_group_idx: u32) -> ModularParams<'a> {
        let Some(image) = &self.inner else {
            return ModularParams {
                width: 0,
                height: 0,
                group_dim: 128,
                bit_depth: 8,
                channel_shifts: Vec::new(),
                ma_config: None,
                channel_mapping: None,
            };
        };

        let width = image.base_width;
        let height = image.base_height;
        let group_dim = image.group_dim;
        let lf_dim = group_dim * 8;
        let bit_depth = image.image.bit_depth();

        let lf_group_stride = (width + lf_dim - 1) / lf_dim;
        let lf_group_row = lf_group_idx / lf_group_stride;
        let lf_group_col = lf_group_idx % lf_group_stride;
        let x = lf_group_col * lf_dim;
        let y = lf_group_row * lf_dim;
        let width = lf_dim.min(width - x);
        let height = lf_dim.min(height - y);

        let (channel_shifts, channel_mapping) = image.channels.info
            .iter()
            .enumerate()
            .filter_map(|(i, &ModularChannelInfo { width, height, hshift, vshift, .. })| {
                if i < image.channels.nb_meta_channels as usize ||
                    (width <= group_dim && height <= group_dim) ||
                    hshift < 3 || vshift < 3
                {
                    None
                } else {
                    Some((
                        ChannelShift::Raw(hshift, vshift),
                        SubimageChannelInfo::new(i, x, y),
                    ))
                }
            })
            .unzip();

        let mut params = ModularParams::new(
            width,
            height,
            group_dim,
            bit_depth,
            channel_shifts,
            global_ma_config,
        );
        params.channel_mapping = Some(channel_mapping);
        params
    }

    pub fn make_subimage_params_pass_group<'a>(&self, global_ma_config: Option<&'a MaConfig>, group_idx: u32, minshift: i32, maxshift: i32) -> ModularParams<'a> {
        todo!()
    }

    pub fn copy_from_modular(&mut self, other: Modular) -> &mut Self {
        let Some(image) = &mut self.inner else { return self; };
        let Some(other) = other.inner else { return self; };
        let mapping = other.subimage_channel_mapping.expect("image being copied is not a subimage");
        image.image.copy_from_image(other.image, &mapping);
        self
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

        let image = Image::new(channels.clone(), params.group_dim, params.bit_depth);

        Ok(Self {
            base_width: params.width,
            base_height: params.height,
            group_dim: params.group_dim,
            header,
            ma_ctx,
            channels,
            subimage_channel_mapping: params.channel_mapping,
            image,
        })
    }
}

define_bundle! {
    #[derive(Debug)]
    pub struct ModularHeader error(crate::Error) {
        use_global_tree: ty(Bool),
        wp_params: ty(Bundle(predictor::WpHeader)),
        nb_transforms: ty(U32(0, 1, 2 + u(4), 18 + u(8))),
        transform: ty(Vec[Bundle(transform::TransformInfo)]; nb_transforms) ctx(&wp_params),
    }
}

#[derive(Debug, Clone)]
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
pub struct ModularChannelInfo {
    width: u32,
    height: u32,
    hshift: i32,
    vshift: i32,
}

impl ModularChannelInfo {
    fn new(mut width: u32, mut height: u32, shift: ChannelShift) -> Self {
        let hshift = shift.hshift();
        let vshift = shift.vshift();
        if hshift >= 0 {
            width >>= hshift;
        }
        if vshift >= 0 {
            height >>= vshift;
        }
        Self {
            width,
            height,
            hshift,
            vshift,
        }
    }

    fn new_shifted(width: u32, height: u32, hshift: i32, vshift: i32) -> Self {
        Self {
            width,
            height,
            hshift,
            vshift,
        }
    }
}
