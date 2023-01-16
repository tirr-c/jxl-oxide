use std::io::Read;

use jxl_bitstream::{define_bundle, read_bits, Bitstream, Bundle};

mod error;
mod image;
mod ma;
mod param;
mod predictor;
mod transform;
pub use error::{Error, Result};
pub use image::Image;
pub use ma::{MaConfig, MaContext};
pub use param::*;

#[derive(Debug, Default)]
pub struct Modular {
    inner: Option<ModularData>,
}

#[derive(Debug)]
struct ModularData {
    group_dim: u32,
    header: ModularHeader,
    ma_ctx: ma::MaContext,
    channels: ModularChannels,
    subimage_channel_mapping: Option<Vec<SubimageChannelInfo>>,
    image: Image,
}

impl Bundle<ModularParams<'_>> for Modular {
    type Error = crate::Error;

    fn parse<R: Read>(
        bitstream: &mut Bitstream<R>,
        params: ModularParams<'_>,
    ) -> Result<Self> {
        let inner = if params.channels.is_empty() {
            None
        } else {
            Some(read_bits!(bitstream, Bundle(ModularData), params)?)
        };
        Ok(Self { inner })
    }
}

impl Modular {
    pub fn empty() -> Self {
        Self::default()
    }
}

impl Modular {
    pub fn has_delta_palette(&self) -> bool {
        let Some(image) = &self.inner else { return false; };
        image.header.transform.iter().any(|tr| tr.is_delta_palette())
    }

    pub fn has_squeeze(&self) -> bool {
        let Some(image) = &self.inner else { return false; };
        image.header.transform.iter().any(|tr| tr.is_squeeze())
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
                group_dim: 128,
                bit_depth: 8,
                channels: Vec::new(),
                channel_mapping: None,
                ma_config: None,
            };
        };

        let Some((base_width, _)) = image.channels.base_size else {
            return ModularParams {
                group_dim: 128,
                bit_depth: 8,
                channels: Vec::new(),
                channel_mapping: None,
                ma_config: None,
            };
        };

        let group_dim = image.group_dim;
        let lf_dim = group_dim * 8;
        let bit_depth = image.image.bit_depth();

        let lf_group_stride = (base_width + lf_dim - 1) / lf_dim;
        let lf_group_row = lf_group_idx / lf_group_stride;
        let lf_group_col = lf_group_idx % lf_group_stride;

        let (channels, channel_mapping) = image.channels.info
            .iter()
            .enumerate()
            .skip_while(|&(i, &ModularChannelInfo { width, height, .. })| {
                i < image.channels.nb_meta_channels as usize ||
                    (width <= group_dim && height <= group_dim)
            })
            .filter_map(|(i, &ModularChannelInfo { width, height, hshift, vshift, .. })| {
                if hshift < 3 || vshift < 3 {
                    None
                } else {
                    let gw = lf_dim >> hshift;
                    let gh = lf_dim >> vshift;
                    let x = lf_group_col * gw;
                    let y = lf_group_row * gh;
                    let width = (width - x).min(gw) << hshift;
                    let height = (height - y).min(gh) << vshift;
                    Some((
                        ModularChannelParams::with_shift(width, height, group_dim, ChannelShift::Raw(hshift, vshift)),
                        SubimageChannelInfo::new(i, x, y),
                    ))
                }
            })
            .unzip();

        let mut params = ModularParams::with_channels(group_dim, bit_depth, channels, global_ma_config);
        params.channel_mapping = Some(channel_mapping);
        params
    }

    pub fn make_subimage_params_pass_group<'a>(&self, global_ma_config: Option<&'a MaConfig>, group_idx: u32, minshift: i32, maxshift: i32) -> ModularParams<'a> {
        let Some(image) = &self.inner else {
            return ModularParams {
                group_dim: 128,
                bit_depth: 8,
                channels: Vec::new(),
                channel_mapping: None,
                ma_config: None,
            };
        };

        let Some((base_width, _)) = image.channels.base_size else {
            return ModularParams {
                group_dim: 128,
                bit_depth: 8,
                channels: Vec::new(),
                channel_mapping: None,
                ma_config: None,
            };
        };

        let group_dim = image.group_dim;
        let bit_depth = image.image.bit_depth();

        let group_stride = (base_width + group_dim - 1) / group_dim;
        let group_row = group_idx / group_stride;
        let group_col = group_idx % group_stride;

        let (channels, channel_mapping) = image.channels.info
            .iter()
            .enumerate()
            .skip_while(|&(i, &ModularChannelInfo { width, height, .. })| {
                i < image.channels.nb_meta_channels as usize ||
                    (width <= group_dim && height <= group_dim)
            })
            .filter_map(|(i, &ModularChannelInfo { width, height, hshift, vshift, .. })| {
                let shift = hshift.min(vshift);
                if (hshift >= 3 && vshift >= 3) || shift < minshift || maxshift <= shift {
                    None
                } else {
                    let gw = group_dim >> hshift;
                    let gh = group_dim >> vshift;
                    let x = group_col * gw;
                    let y = group_row * gh;
                    let width = (width - x).min(gw) << hshift;
                    let height = (height - y).min(gh) << vshift;
                    Some((
                        ModularChannelParams::with_shift(width, height, group_dim, ChannelShift::Raw(hshift, vshift)),
                        SubimageChannelInfo::new(i, x, y),
                    ))
                }
            })
            .unzip();

        let mut params = ModularParams::with_channels(group_dim, bit_depth, channels, global_ma_config);
        params.channel_mapping = Some(channel_mapping);
        params
    }

    pub fn copy_from_modular(&mut self, other: Modular) -> &mut Self {
        let Some(image) = &mut self.inner else { return self; };
        let Some(other) = other.inner else { return self; };
        let mapping = other.subimage_channel_mapping.expect("image being copied is not a subimage");
        image.image.copy_from_image(other.image, &mapping);
        self
    }

    pub fn image(&self) -> &Image {
        let Some(image) = &self.inner else { return &image::EMPTY; };
        &image.image
    }

    pub fn into_image(self) -> Image {
        let Some(image) = self.inner else { return Image::empty(); };
        image.image
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
    base_size: Option<(u32, u32)>,
    info: Vec<ModularChannelInfo>,
    nb_meta_channels: u32,
}

impl ModularChannels {
    fn from_params(params: &ModularParams<'_>) -> Self {
        let mut base_size = Some((params.channels[0].width, params.channels[0].height));
        for &ModularChannelParams { width, height, .. } in &params.channels {
            let (bw, bh) = base_size.unwrap();
            if bw != width || bh != height {
                base_size = None;
                break;
            }
        }
        let info = params.channels.iter()
            .map(|ch| ModularChannelInfo::new(ch.width, ch.height, ch.shift))
            .collect();
        Self {
            base_size,
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
