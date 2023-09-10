use std::io::Read;

use jxl_bitstream::{Bitstream, unpack_signed};
use jxl_coding::{DecoderRleMode, RleToken, DecoderWithLz77};
use jxl_grid::Grid;

use crate::{
    ModularChannels,
    predictor::{WpHeader, PredictorState, Predictor},
    Result,
    SubimageChannelInfo, MaConfig, ma::{MaTreeLeafClustered, FlatMaTree}, ModularChannelInfo,
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

    fn decode_channel_loop(
        &mut self,
        mut loop_fn: impl FnMut(usize, &mut Grid<i32>, &[&Grid<i32>]) -> Result<()>,
    ) -> Result<()> {
        let mut prev: Vec<(&ModularChannelInfo, &mut Grid<i32>)> = Vec::new();
        for (i, (info, grid)) in self.channels.info.iter().zip(&mut self.data).enumerate() {
            let mut filtered_prev = prev
                .iter()
                .filter(|&(prev_info, _)| {
                    info.width == prev_info.width &&
                        info.height == prev_info.height &&
                        info.hshift == prev_info.hshift &&
                        info.vshift == prev_info.vshift
                })
                .map(|(_, g)| &**g)
                .collect::<Vec<_>>();
            filtered_prev.reverse();

            loop_fn(i, grid, &filtered_prev)?;

            prev.push((info, grid));
        }

        Ok(())
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

        if let Some(rle_decoder) = decoder.as_rle() {
            self.decode_image_rle(bitstream, stream_index, wp_header, ma_ctx, rle_decoder)?;
            decoder.finalize()?;
            return Ok(());
        }
        if let Some(lz77_decoder) = decoder.as_with_lz77() {
            self.decode_image_lz77(bitstream, stream_index, wp_header, ma_ctx, lz77_decoder)?;
            decoder.finalize()?;
            return Ok(());
        }
        let mut no_lz77_decoder = decoder.as_no_lz77().unwrap();

        self.decode_channel_loop(|i, grid, prev_rev| {
            let width = grid.width();
            let height = grid.height();

            let ma_tree = ma_ctx.make_flat_tree(i as u32, stream_index);
            if let Some(&MaTreeLeafClustered { cluster, predictor, offset, multiplier }) = ma_tree.single_node() {
                tracing::trace!(cluster, ?predictor, "Single MA tree node");
                if predictor == Predictor::Zero {
                    if let Some(token) = no_lz77_decoder.single_token(cluster) {
                        tracing::trace!("Single token in cluster: hyper fast path");
                        let value = unpack_signed(token) * multiplier as i32 + offset;
                        for y in 0..height {
                            for x in 0..width {
                                grid.set(x, y, value);
                            }
                        }
                    } else {
                        tracing::trace!("Fast path");
                        for y in 0..height {
                            for x in 0..width {
                                let token = no_lz77_decoder.read_varint_clustered(bitstream, cluster)?;
                                let value = unpack_signed(token) * multiplier as i32 + offset;
                                grid.set(x, y, value);
                            }
                        }
                    }
                    return Ok(());
                }
                if predictor == Predictor::Gradient && offset == 0 && multiplier == 1 {
                    tracing::trace!("Quite fast path");
                    let mut prev_row = vec![0i32; width];
                    for y in 0..height {
                        let mut w = prev_row[0] as i64;
                        let mut nw = w;
                        for (x, prev) in prev_row.iter_mut().enumerate() {
                            let n = if y == 0 { w } else { *prev as i64 };
                            let pred = (n + w - nw).clamp(w.min(n), w.max(n));

                            let token = no_lz77_decoder.read_varint_clustered(bitstream, cluster)?;
                            let value = ((unpack_signed(token) as i64) + pred) as i32;
                            grid.set(x, y, value);
                            *prev = value;
                            nw = n;
                            w = value as i64;
                        }
                    }
                    return Ok(());
                }
            }

            let wp_header = ma_tree.need_self_correcting().then_some(wp_header);
            let mut predictor = PredictorState::new(width as u32, i as u32, stream_index, prev_rev.len(), wp_header);
            let mut next = |cluster: u8| -> Result<i32> {
                let token = no_lz77_decoder.read_varint_clustered(bitstream, cluster)?;
                Ok(unpack_signed(token))
            };
            decode_channel_slow(&mut next, &ma_tree, &mut predictor, grid, prev_rev)
        })?;

        decoder.finalize()?;
        Ok(())
    }

    fn decode_image_lz77<R: Read>(
        &mut self,
        bitstream: &mut Bitstream<R>,
        stream_index: u32,
        wp_header: &WpHeader,
        ma_ctx: &MaConfig,
        mut decoder: DecoderWithLz77<'_>,
    ) -> Result<()> {
        let dist_multiplier = self.channels.info.iter()
            .skip(self.channels.nb_meta_channels as usize)
            .map(|info| info.width)
            .max()
            .unwrap_or(0);

        self.decode_channel_loop(|i, grid, prev_rev| {
            let width = grid.width();
            let height = grid.height();

            let ma_tree = ma_ctx.make_flat_tree(i as u32, stream_index);
            if let Some(&MaTreeLeafClustered { cluster, predictor, offset, multiplier }) = ma_tree.single_node() {
                tracing::trace!(cluster, ?predictor, "Single MA tree node");
                if predictor == Predictor::Zero {
                    tracing::trace!("Fast path");
                    for y in 0..height {
                        for x in 0..width {
                            let token = decoder.read_varint_with_multiplier_clustered(bitstream, cluster, dist_multiplier)?;
                            let value = unpack_signed(token) * multiplier as i32 + offset;
                            grid.set(x, y, value);
                        }
                    }
                    return Ok(());
                }
                if predictor == Predictor::Gradient && offset == 0 && multiplier == 1 {
                    tracing::trace!("Quite fast path");
                    let mut prev_row = vec![0i32; width];
                    for y in 0..height {
                        let mut w = prev_row[0] as i64;
                        let mut nw = w;
                        for (x, prev) in prev_row.iter_mut().enumerate() {
                            let n = if y == 0 { w } else { *prev as i64 };
                            let pred = (n + w - nw).clamp(w.min(n), w.max(n));

                            let token = decoder.read_varint_with_multiplier_clustered(bitstream, cluster, dist_multiplier)?;
                            let value = ((unpack_signed(token) as i64) + pred) as i32;
                            grid.set(x, y, value);
                            *prev = value;
                            nw = n;
                            w = value as i64;
                        }
                    }
                    return Ok(());
                }
            }

            let wp_header = ma_tree.need_self_correcting().then_some(wp_header);
            let mut predictor = PredictorState::new(width as u32, i as u32, stream_index, prev_rev.len(), wp_header);
            let mut next = |cluster: u8| -> Result<i32> {
                let token = decoder.read_varint_with_multiplier_clustered(bitstream, cluster, dist_multiplier)?;
                Ok(unpack_signed(token))
            };
            decode_channel_slow(&mut next, &ma_tree, &mut predictor, grid, prev_rev)
        })
    }

    fn decode_image_rle<R: Read>(
        &mut self,
        bitstream: &mut Bitstream<R>,
        stream_index: u32,
        wp_header: &WpHeader,
        ma_ctx: &MaConfig,
        mut decoder: DecoderRleMode<'_>,
    ) -> Result<()> {
        let mut rle_value = 0i32;
        let mut rle_left = 0u32;

        let mut next = |cluster: u8| -> Result<i32> {
            Ok(if rle_left > 0 {
                rle_left -= 1;
                rle_value
            } else {
                match decoder.read_varint_clustered(bitstream, cluster)? {
                    RleToken::Value(v) => {
                        rle_value = unpack_signed(v);
                        rle_value
                    },
                    RleToken::Repeat(len) => {
                        rle_left = len - 1;
                        rle_value
                    },
                }
            })
        };

        self.decode_channel_loop(|i, grid, prev_rev| {
            let width = grid.width();
            let height = grid.height();

            let ma_tree = ma_ctx.make_flat_tree(i as u32, stream_index);
            if let Some(&MaTreeLeafClustered { cluster, predictor, offset, multiplier }) = ma_tree.single_node() {
                tracing::trace!(cluster, ?predictor, "Single MA tree node");
                if predictor == Predictor::Zero {
                    tracing::trace!("Quite fast path");
                    for y in 0..height {
                        for x in 0..width {
                            let token = next(cluster)?;
                            let value = (token * multiplier as i32) + offset;
                            grid.set(x, y, value);
                        }
                    }
                    return Ok(());
                }
                if predictor == Predictor::Gradient && offset == 0 && multiplier == 1 {
                    tracing::trace!("libjxl fast-lossless: quite fast path");
                    let mut prev_row = vec![0i32; width];
                    for y in 0..height {
                        let mut w = prev_row[0] as i64;
                        let mut nw = w;
                        for (x, prev) in prev_row.iter_mut().enumerate() {
                            let n = if y == 0 { w } else { *prev as i64 };
                            let pred = (n + w - nw).clamp(w.min(n), w.max(n));

                            let token = next(cluster)?;
                            let value = ((token as i64) + pred) as i32;
                            grid.set(x, y, value);
                            *prev = value;
                            nw = n;
                            w = value as i64;
                        }
                    }
                    return Ok(());
                }
            }

            let wp_header = ma_tree.need_self_correcting().then_some(wp_header);
            let mut predictor = PredictorState::new(width as u32, i as u32, stream_index, prev_rev.len(), wp_header);
            decode_channel_slow(&mut next, &ma_tree, &mut predictor, grid, prev_rev)
        })
    }
}

fn decode_channel_slow(
    next: &mut impl FnMut(u8) -> Result<i32>,
    ma_tree: &FlatMaTree,
    predictor: &mut PredictorState,
    grid: &mut Grid<i32>,
    prev_rev: &[&Grid<i32>],
) -> Result<()> {
    let width = grid.width();
    let height = grid.height();

    let mut prev_channel_samples_rev = vec![0i32; prev_rev.len()];

    for y in 0..height {
        for x in 0..width {
            for (grid, sample) in prev_rev.iter().zip(&mut prev_channel_samples_rev) {
                *sample = *grid.get(x, y).unwrap();
            }

            let properties = predictor.properties(&prev_channel_samples_rev);
            let (diff, predictor) = ma_tree.decode_sample_rle(next, &properties)?;
            let sample_prediction = predictor.predict(&properties);
            let true_value = (diff as i64 + sample_prediction) as i32;
            grid.set(x, y, true_value);
            properties.record(true_value);
        }
    }

    Ok(())
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
