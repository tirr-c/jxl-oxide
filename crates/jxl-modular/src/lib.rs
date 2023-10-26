//! JPEG XL Modular image decoder.
//!
//! A Modular image represents a set of grids (two-dimensional arrays) of integer values. Modular
//! images are used mainly for lossless images, but lossy VarDCT images also use them to store
//! various information, such as quantized LF images and varblock configurations.
use jxl_bitstream::{define_bundle, read_bits, Bitstream, Bundle};

mod error;
mod image;
mod ma;
mod param;
mod predictor;
mod transform;
pub use error::{Error, Result};
pub use image::Image;
pub use ma::MaConfig;
pub use param::*;

/// A Modular encoded image.
///
/// Modular image decoding is done in two steps:
/// 1. Construct a value of `Modular` by either:
///    - reading a Modular header from the bitstream, or
///    - creating a subimage of existing image by calling [self.make_subimage_params_lf_group] or
///      [self.make_subimage_params_pass_group].
/// 2. Decode pixels by calling [self.decode_image] or [self.decode_image_gmodular].
#[derive(Debug, Clone, Default)]
pub struct Modular {
    inner: Option<ModularData>,
}

#[derive(Debug, Clone)]
struct ModularData {
    group_dim: u32,
    header: ModularHeader,
    ma_ctx: MaConfig,
    channels: ModularChannels,
    subimage_channel_mapping: Option<Vec<SubimageChannelInfo>>,
    image: Image,
}

impl Bundle<ModularParams<'_>> for Modular {
    type Error = crate::Error;

    fn parse(
        bitstream: &mut Bitstream,
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
    /// Creates an empty Modular image.
    pub fn empty() -> Self {
        Self::default()
    }
}

impl Modular {
    pub fn has_palette(&self) -> bool {
        let Some(image) = &self.inner else { return false; };
        image.header.transform.iter().any(|tr| tr.is_palette())
    }

    pub fn has_squeeze(&self) -> bool {
        let Some(image) = &self.inner else { return false; };
        image.header.transform.iter().any(|tr| tr.is_squeeze())
    }
}

impl Modular {
    pub fn decode_image_gmodular(&mut self, bitstream: &mut Bitstream, allow_partial: bool) -> Result<()> {
        let Some(image) = &mut self.inner else { return Ok(()); };
        let wp_header = &image.header.wp_params;
        let ma_ctx = &mut image.ma_ctx;
        let (mut subimage, channel_mapping) = image.image.for_global_modular();
        match subimage.decode_channels(bitstream, 0, wp_header, ma_ctx) {
            Err(e) if e.unexpected_eof() && allow_partial => {
                tracing::debug!("Partially decoded Modular image");
            },
            Err(e) => return Err(e),
            Ok(_) => {},
        }
        image.image.copy_from_image(subimage, &channel_mapping);
        Ok(())
    }

    pub fn decode_image(&mut self, bitstream: &mut Bitstream, stream_index: u32, allow_partial: bool) -> Result<()> {
        let Some(image) = &mut self.inner else { return Ok(()); };
        let wp_header = &image.header.wp_params;
        let ma_ctx = &mut image.ma_ctx;
        match image.image.decode_channels(bitstream, stream_index, wp_header, ma_ctx) {
            Err(e) if e.unexpected_eof() && allow_partial => {
                tracing::debug!("Partially decoded Modular image");
                Ok(())
            },
            Err(e) => Err(e),
            Ok(_) => Ok(()),
        }
    }

    /// Apply inverse transforms to the decoded image.
    pub fn inverse_transform(&mut self) {
        let Some(image) = &mut self.inner else { return; };
        for transform in image.header.transform.iter().rev() {
            transform.inverse(&mut image.image);
        }
    }

    pub fn make_subimage_params_lf_group<'a>(
        &self,
        global_ma_config: Option<&'a MaConfig>,
        lf_group_idx: u32,
    ) -> ModularParams<'a> {
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
                        ModularChannelParams::with_shift(width, height, ChannelShift::Raw(hshift, vshift)),
                        SubimageChannelInfo::new(i, x, y),
                    ))
                }
            })
            .unzip();

        let mut params = ModularParams::with_channels(group_dim, bit_depth, channels, global_ma_config);
        params.channel_mapping = Some(channel_mapping);
        params
    }

    pub fn make_subimage_params_pass_group<'a>(
        &self,
        global_ma_config: Option<&'a MaConfig>,
        group_idx: u32,
        minshift: i32,
        maxshift: i32,
    ) -> ModularParams<'a> {
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
                        ModularChannelParams::with_shift(width, height, ChannelShift::Raw(hshift, vshift)),
                        SubimageChannelInfo::new(i, x, y),
                    ))
                }
            })
            .unzip();

        let mut params = ModularParams::with_channels(group_dim, bit_depth, channels, global_ma_config);
        params.channel_mapping = Some(channel_mapping);
        params
    }

    /// Insert the decoded Modular subimage.
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

    fn parse(
        bitstream: &mut Bitstream,
        params: ModularParams<'_>,
    ) -> Result<Self> {
        let mut header = read_bits!(bitstream, Bundle(ModularHeader))?;
        if header.nb_transforms > 512 {
            tracing::error!(nb_transforms = header.nb_transforms, "nb_transforms too large");
            return Err(jxl_bitstream::Error::ProfileConformance(
                "nb_transforms too large"
            ).into());
        }

        let ma_ctx = if header.use_global_tree {
            params.ma_config.ok_or(crate::Error::GlobalMaTreeNotAvailable)?.clone()
        } else {
            read_bits!(bitstream, Bundle(ma::MaConfig))?
        };
        if ma_ctx.tree_depth() > 2048 {
            tracing::error!(tree_depth = ma_ctx.tree_depth(), "Decoded MA tree is too deep");
            return Err(jxl_bitstream::Error::ProfileConformance(
                "decoded MA tree is too deep"
            ).into())
        }

        let mut channels = ModularChannels::from_params(&params);
        for transform in &mut header.transform {
            transform.or_default(&mut channels);
            transform.transform_channel_info(&mut channels)?;
        }

        if channels.info.len() > (1 << 16) {
            tracing::error!(nb_channels_tr = channels.info.len(), "nb_channels_tr too large");
            return Err(jxl_bitstream::Error::ProfileConformance(
                "nb_channels_tr too large"
            ).into());
        }

        if !header.use_global_tree {
            let num_local_samples: u64 = channels.info.iter()
                .map(|ch| (ch.width as u64 * ch.height as u64))
                .sum();
            let local_ma_nodes = ma_ctx.num_tree_nodes();
            let max_local_ma_nodes = (1 << 20).min(1024 + num_local_samples) as usize;
            if ma_ctx.num_tree_nodes() > max_local_ma_nodes {
                tracing::error!(local_ma_nodes, max_local_ma_nodes, "Too many local MA tree nodes");
                return Err(jxl_bitstream::Error::ProfileConformance(
                    "too many local MA tree nodes"
                ).into());
            }
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
    #[derive(Debug, Clone)]
    struct ModularHeader error(crate::Error) {
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
    fn new(width: u32, height: u32, shift: ChannelShift) -> Self {
        let (width, height) = shift.shift_size((width, height));
        Self {
            width,
            height,
            hshift: shift.hshift(),
            vshift: shift.vshift(),
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
