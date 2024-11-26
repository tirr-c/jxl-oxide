//! JPEG XL Modular image decoder.
//!
//! A Modular image represents a set of grids (two-dimensional arrays) of integer values. Modular
//! images are used mainly for lossless images, but lossy VarDCT images also use them to store
//! various information, such as quantized LF images and varblock configurations.
use jxl_bitstream::Bitstream;
use jxl_oxide_common::{define_bundle, Bundle};

mod error;
pub mod image;
mod ma;
mod param;
mod predictor;
mod sample;
mod transform;
pub use error::{Error, Result};
use jxl_grid::AllocTracker;
pub use ma::{FlatMaTree, MaConfig, MaConfigParams};
pub use param::*;
pub use sample::Sample;

/// A Modular encoded image.
///
/// Modular image decoding is done in two steps:
/// 1. Construct a value of `Modular` by either:
///    - reading a Modular header from the bitstream, or
///    - creating a subimage of existing image by calling [self.make_subimage_params_lf_group] or
///      [self.make_subimage_params_pass_group].
/// 2. Decode pixels by calling [self.decode_image] or [self.decode_image_gmodular].
#[derive(Debug, Default)]
pub struct Modular<S: Sample> {
    inner: Option<ModularData<S>>,
}

#[derive(Debug)]
struct ModularData<S: Sample> {
    image: image::ModularImageDestination<S>,
}

impl<S: Sample> Bundle<ModularParams<'_, '_>> for Modular<S> {
    type Error = crate::Error;

    fn parse(bitstream: &mut Bitstream, params: ModularParams) -> Result<Self> {
        let inner = if params.channels.is_empty() {
            None
        } else {
            Some(ModularData::<S>::parse(bitstream, params)?)
        };
        Ok(Self { inner })
    }
}

impl<S: Sample> Modular<S> {
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

impl<S: Sample> Modular<S> {
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

impl<S: Sample> Modular<S> {
    pub fn image(&self) -> Option<&image::ModularImageDestination<S>> {
        self.inner.as_ref().map(|x| &x.image)
    }

    pub fn image_mut(&mut self) -> Option<&mut image::ModularImageDestination<S>> {
        self.inner.as_mut().map(|x| &mut x.image)
    }

    pub fn into_image(self) -> Option<image::ModularImageDestination<S>> {
        self.inner.map(|x| x.image)
    }
}

impl<S: Sample> Bundle<ModularParams<'_, '_>> for ModularData<S> {
    type Error = crate::Error;

    fn parse(bitstream: &mut Bitstream, params: ModularParams) -> Result<Self> {
        let channels = ModularChannels::from_params(&params);
        let (header, ma_ctx) = read_and_validate_local_modular_header(
            bitstream,
            &channels,
            params.ma_config,
            params.tracker,
        )?;
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
    original_width: u32,
    original_height: u32,
    hshift: i32,
    vshift: i32,
    original_shift: ChannelShift,
}

impl ModularChannelInfo {
    fn new(original_width: u32, original_height: u32, shift: ChannelShift) -> Self {
        let (width, height) = shift.shift_size((original_width, original_height));
        Self {
            width,
            height,
            original_width,
            original_height,
            hshift: shift.hshift(),
            vshift: shift.vshift(),
            original_shift: shift,
        }
    }

    fn new_unshiftable(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            original_width: width,
            original_height: height,
            hshift: -1,
            vshift: -1,
            original_shift: ChannelShift::from_shift(0),
        }
    }

    pub fn shift(&self) -> ChannelShift {
        self.original_shift
    }

    pub fn original_size(&self) -> (u32, u32) {
        (self.original_width, self.original_height)
    }
}

fn read_and_validate_local_modular_header(
    bitstream: &mut Bitstream,
    channels: &ModularChannels,
    global_ma_config: Option<&MaConfig>,
    tracker: Option<&AllocTracker>,
) -> Result<(ModularHeader, MaConfig)> {
    let mut header = ModularHeader::parse(bitstream, ())?;
    if header.nb_transforms > 512 {
        tracing::error!(
            nb_transforms = header.nb_transforms,
            "nb_transforms too large"
        );
        return Err(jxl_bitstream::Error::ProfileConformance("nb_transforms too large").into());
    }

    let mut tr_channels = channels.clone();
    for tr in &mut header.transform {
        tr.prepare_transform_info(&mut tr_channels)?;
    }

    let nb_channels_tr = tr_channels.info.len();
    if nb_channels_tr > (1 << 16) {
        tracing::error!(nb_channels_tr, "nb_channels_tr too large");
        return Err(jxl_bitstream::Error::ProfileConformance("nb_channels_tr too large").into());
    }

    let ma_ctx = if header.use_global_tree {
        global_ma_config
            .ok_or(crate::Error::GlobalMaTreeNotAvailable)?
            .clone()
    } else {
        let local_samples = tr_channels
            .info
            .iter()
            .fold(0u64, |acc, ch| acc + (ch.width as u64 * ch.height as u64));
        let params = MaConfigParams {
            tracker,
            node_limit: (1024 + local_samples).min(1 << 20) as usize,
            depth_limit: 2048,
        };
        MaConfig::parse(bitstream, params)?
    };

    Ok((header, ma_ctx))
}
