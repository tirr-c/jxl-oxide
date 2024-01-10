//! JPEG XL Modular image decoder.
//!
//! A Modular image represents a set of grids (two-dimensional arrays) of integer values. Modular
//! images are used mainly for lossless images, but lossy VarDCT images also use them to store
//! various information, such as quantized LF images and varblock configurations.
use jxl_bitstream::{define_bundle, read_bits, Bitstream, Bundle};

mod error;
pub mod image;
mod ma;
mod param;
mod predictor;
mod transform;
pub use error::{Error, Result};
pub use ma::{MaConfig, MaConfigParams};
pub use param::*;

/// A Modular encoded image.
///
/// Modular image decoding is done in two steps:
/// 1. Construct a value of `Modular` by either:
///    - reading a Modular header from the bitstream, or
///    - creating a subimage of existing image by calling [self.make_subimage_params_lf_group] or
///      [self.make_subimage_params_pass_group].
/// 2. Decode pixels by calling [self.decode_image] or [self.decode_image_gmodular].
#[derive(Debug, Default)]
pub struct Modular {
    inner: Option<ModularData>,
}

#[derive(Debug)]
struct ModularData {
    image: image::ModularImageDestination,
}

impl Bundle<ModularParams<'_, '_>> for Modular {
    type Error = crate::Error;

    fn parse(bitstream: &mut Bitstream, params: ModularParams) -> Result<Self> {
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

    pub fn try_clone(&self) -> Result<Self> {
        let inner = if let Some(inner) = &self.inner {
            Some(ModularData {
                image: inner.image.try_clone()?,
            })
        } else {
            None
        };

        Ok(Self { inner })
    }
}

impl Modular {
    pub fn has_palette(&self) -> bool {
        let Some(image) = &self.inner else {
            return false;
        };
        image.image.has_palette()
    }

    pub fn has_squeeze(&self) -> bool {
        let Some(image) = &self.inner else {
            return false;
        };
        image.image.has_squeeze()
    }
}

impl Modular {
    pub fn image(&self) -> Option<&image::ModularImageDestination> {
        self.inner.as_ref().map(|x| &x.image)
    }

    pub fn image_mut(&mut self) -> Option<&mut image::ModularImageDestination> {
        self.inner.as_mut().map(|x| &mut x.image)
    }

    pub fn into_image(self) -> Option<image::ModularImageDestination> {
        self.inner.map(|x| x.image)
    }
}

impl Bundle<ModularParams<'_, '_>> for ModularData {
    type Error = crate::Error;

    fn parse(bitstream: &mut Bitstream, params: ModularParams) -> Result<Self> {
        let mut header = read_bits!(bitstream, Bundle(ModularHeader))?;
        if header.nb_transforms > 512 {
            tracing::error!(
                nb_transforms = header.nb_transforms,
                "nb_transforms too large"
            );
            return Err(jxl_bitstream::Error::ProfileConformance("nb_transforms too large").into());
        }

        let channels = ModularChannels::from_params(&params);
        let mut tr_channels = channels.clone();
        for tr in &mut header.transform {
            tr.prepare_transform_info(&mut tr_channels)?;
        }

        let nb_channels_tr = tr_channels.info.len();
        if nb_channels_tr > (1 << 16) {
            tracing::error!(nb_channels_tr, "nb_channels_tr too large");
            return Err(
                jxl_bitstream::Error::ProfileConformance("nb_channels_tr too large").into(),
            );
        }

        let ma_ctx = if header.use_global_tree {
            params
                .ma_config
                .ok_or(crate::Error::GlobalMaTreeNotAvailable)?
                .clone()
        } else {
            let local_samples = tr_channels.info.iter().fold(0usize, |acc, ch| {
                acc + (ch.width as usize * ch.height as usize)
            });
            let params = MaConfigParams {
                tracker: params.tracker,
                node_limit: (1024 + local_samples).min(1 << 20),
            };
            bitstream.read_bundle_with_ctx(params)?
        };
        if ma_ctx.tree_depth() > 2048 {
            tracing::error!(
                tree_depth = ma_ctx.tree_depth(),
                "Decoded MA tree is too deep"
            );
            return Err(
                jxl_bitstream::Error::ProfileConformance("decoded MA tree is too deep").into(),
            );
        }

        Ok(Self {
            image: image::ModularImageDestination::new(
                header,
                ma_ctx,
                params.group_dim,
                params.bit_depth,
                channels,
                params.tracker,
            )?,
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
    info: Vec<ModularChannelInfo>,
    nb_meta_channels: u32,
}

impl ModularChannels {
    fn from_params(params: &ModularParams) -> Self {
        let info = params
            .channels
            .iter()
            .map(|ch| ModularChannelInfo::new(ch.width, ch.height, ch.shift))
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
