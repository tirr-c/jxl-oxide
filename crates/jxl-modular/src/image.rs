use std::io::Read;

use jxl_bitstream::Bitstream;
use jxl_grid::Grid;

use crate::{
    ModularChannels,
    predictor::{WpHeader, PredictorState},
    Result,
    SubimageChannelInfo, MaConfig,
};

/// Decoded Modular image.
///
/// A decoded Modular image consists of multiple channels. Those channels may not be in the same
/// size.
#[derive(Debug, Clone)]
pub struct Image {
    group_dim: u32,
    bit_depth: u32,
    channels: ModularChannels,
    data: Vec<Grid<i32>>,
}

pub(crate) static EMPTY: Image = Image::empty();

impl Image {
    pub(super) const fn empty() -> Self {
        Self {
            group_dim: 128,
            bit_depth: 8,
            channels: ModularChannels {
                base_size: Some((0, 0)),
                info: Vec::new(),
                nb_meta_channels: 0,
            },
            data: Vec::new(),
        }
    }

    pub(super) fn new(channels: ModularChannels, group_dim: u32, bit_depth: u32) -> Self {
        let span = tracing::span!(tracing::Level::TRACE, "Image::new", group_dim, bit_depth);
        let _guard = span.enter();

        let data = channels.info
            .iter()
            .map(|info| {
                tracing::trace!(info = format_args!("{:?}", info), "Creating modular channel");
                if info.hshift < 0 || info.vshift < 0 {
                    return Grid::new(info.width, info.height, info.width, info.height);
                }

                let group_dim = if info.hshift >= 3 && info.vshift >= 3 {
                    group_dim * 8
                } else {
                    group_dim
                };
                let gw = group_dim >> info.hshift;
                let gh = group_dim >> info.vshift;
                Grid::new(info.width, info.height, gw, gh)
            })
            .collect::<Vec<_>>();
        Self {
            group_dim,
            bit_depth,
            channels,
            data,
        }
    }

    pub(super) fn decode_channels<R: Read>(
        &mut self,
        bitstream: &mut Bitstream<R>,
        stream_index: u32,
        wp_header: &WpHeader,
        ma_ctx: &MaConfig,
    ) -> Result<()> {
        let span = tracing::span!(tracing::Level::TRACE, "decode channels", stream_index);
        let _guard = span.enter();

        let mut decoder = ma_ctx.decoder().clone();
        decoder.begin(bitstream)?;

        let dist_multiplier = self.channels.info.iter()
            .skip(self.channels.nb_meta_channels as usize)
            .map(|info| info.width)
            .max()
            .unwrap_or(0);

        let mut channels: Vec<_> = self.channels.info.iter()
            .zip(self.data.iter_mut())
            .enumerate()
            .filter(|(_, (_, grid))| grid.width() != 0 && grid.height() != 0)
            .collect();
        let len = channels.len();
        for idx in 0..len {
            let (prev, left) = channels.split_at_mut(idx);
            let (i, (info, ref mut grid)) = left[0];
            let prev = prev
                .iter()
                .filter(|(_, (prev_info, _))| {
                    info.width == prev_info.width &&
                        info.height == prev_info.height &&
                        info.hshift == prev_info.hshift &&
                        info.vshift == prev_info.vshift
                })
                .collect::<Vec<_>>();

            let ma_tree = ma_ctx.make_flat_tree(i as u32, stream_index);
            let wp_header = ma_tree.need_self_correcting().then_some(wp_header);

            let width = grid.width();
            let height = grid.height();
            let mut predictor = PredictorState::new(width as u32, i as u32, stream_index, prev.len(), wp_header);
            let mut prev_channel_samples = vec![0i32; prev.len()];

            for y in 0..height {
                for x in 0..width {
                    for ((_, (_, grid)), sample) in prev.iter().zip(&mut prev_channel_samples) {
                        *sample = *grid.get(x, y).unwrap();
                    }

                    let properties = predictor.properties(&prev_channel_samples);
                    let (diff, predictor) = ma_tree.decode_sample(bitstream, &mut decoder, &properties, dist_multiplier)?;
                    let sample_prediction = predictor.predict(&properties);
                    let true_value = (diff as i64 + sample_prediction) as i32;
                    grid.set(x, y, true_value);
                    properties.record(true_value);
                }
            }
        }

        decoder.finalize()?;
        Ok(())
    }
}

impl Image {
    pub fn group_dim(&self) -> u32 {
        self.group_dim
    }

    pub fn bit_depth(&self) -> u32 {
        self.bit_depth
    }
}

impl Image {
    pub(super) fn for_global_modular(&self) -> (Image, Vec<SubimageChannelInfo>) {
        let group_dim = self.group_dim;
        let (channel_info, channel_mapping) = self.channels.info
            .iter()
            .enumerate()
            .take_while(|&(i, info)| {
                i < self.channels.nb_meta_channels as usize ||
                    (info.width <= group_dim && info.height <= group_dim)
            })
            .map(|(i, info)| (info.clone(), SubimageChannelInfo::new(i, 0, 0)))
            .unzip();
        let channels = ModularChannels {
            info: channel_info,
            ..self.channels
        };
        (Image::new(channels, group_dim, self.bit_depth), channel_mapping)
    }

    pub(super) fn copy_from_image(&mut self, subimage: Image, mapping: &[super::SubimageChannelInfo]) -> &mut Self {
        for (mut grid, mapping) in subimage.data.into_iter().zip(mapping) {
            let SubimageChannelInfo { channel_id, base_x, base_y } = *mapping;
            self.data[channel_id].insert_subgrid(&mut grid, base_x as isize, base_y as isize);
        }
        self
    }

    /// Returns a reference to the list of channels.
    pub fn channel_data(&self) -> &[Grid<i32>] {
        &self.data
    }

    pub(crate) fn channel_data_mut(&mut self) -> &mut Vec<Grid<i32>> {
        &mut self.data
    }

    /// Make this `Image` into a [Vec] of channels.
    pub fn into_channel_data(self) -> Vec<Grid<i32>> {
        self.data
    }
}
