use jxl_bitstream::{unpack_signed, Bitstream};
use jxl_coding::{DecoderRleMode, DecoderWithLz77, RleToken};
use jxl_grid::{AllocTracker, CutGrid, SimpleGrid};

use crate::{
    ma::{FlatMaTree, MaTreeLeafClustered},
    predictor::{Predictor, PredictorState},
    MaConfig, ModularChannelInfo, ModularChannels, ModularHeader, Result,
};

#[derive(Debug)]
pub enum TransformedGrid<'dest> {
    Single(CutGrid<'dest, i32>),
    Merged {
        leader: CutGrid<'dest, i32>,
        members: Vec<TransformedGrid<'dest>>,
    },
}

impl<'dest> From<CutGrid<'dest, i32>> for TransformedGrid<'dest> {
    fn from(value: CutGrid<'dest, i32>) -> Self {
        Self::Single(value)
    }
}

impl TransformedGrid<'_> {
    fn reborrow(&mut self) -> TransformedGrid {
        match self {
            TransformedGrid::Single(g) => TransformedGrid::Single(g.split_horizontal(0).1),
            TransformedGrid::Merged { leader, .. } => {
                TransformedGrid::Single(leader.split_horizontal(0).1)
            }
        }
    }
}

impl<'dest> TransformedGrid<'dest> {
    pub(crate) fn grid(&self) -> &CutGrid<'dest, i32> {
        match self {
            Self::Single(g) => g,
            Self::Merged { leader, .. } => leader,
        }
    }

    pub(crate) fn grid_mut(&mut self) -> &mut CutGrid<'dest, i32> {
        match self {
            Self::Single(g) => g,
            Self::Merged { leader, .. } => leader,
        }
    }

    pub(crate) fn merge(&mut self, members: Vec<TransformedGrid<'dest>>) {
        if members.is_empty() {
            return;
        }

        match self {
            Self::Single(leader) => {
                let tmp = CutGrid::from_buf(&mut [], 0, 0, 0);
                let leader = std::mem::replace(leader, tmp);
                *self = Self::Merged { leader, members };
            }
            Self::Merged {
                members: original_members,
                ..
            } => {
                original_members.extend(members);
            }
        }
    }

    pub(crate) fn unmerge(&mut self, count: usize) -> Vec<TransformedGrid<'dest>> {
        if count == 0 {
            return Vec::new();
        }

        match self {
            Self::Single(_) => panic!("cannot unmerge TransformedGrid::Single"),
            Self::Merged { leader, members } => {
                let len = members.len();
                let members = members.drain((len - count)..).collect();
                if len == count {
                    let tmp = CutGrid::from_buf(&mut [], 0, 0, 0);
                    let leader = std::mem::replace(leader, tmp);
                    *self = Self::Single(leader);
                }
                members
            }
        }
    }
}

#[derive(Debug)]
pub struct ModularImageDestination {
    header: ModularHeader,
    ma_ctx: MaConfig,
    group_dim: u32,
    bit_depth: u32,
    channels: ModularChannels,
    meta_channels: Vec<SimpleGrid<i32>>,
    image_channels: Vec<SimpleGrid<i32>>,
}

impl ModularImageDestination {
    pub(crate) fn new(
        header: ModularHeader,
        ma_ctx: MaConfig,
        group_dim: u32,
        bit_depth: u32,
        channels: ModularChannels,
        tracker: Option<&AllocTracker>,
    ) -> Result<Self> {
        let mut meta_channels = Vec::new();
        for tr in &header.transform {
            tr.prepare_meta_channels(&mut meta_channels, tracker)?;
        }

        let image_channels = channels
            .info
            .iter()
            .map(|ch| {
                SimpleGrid::with_alloc_tracker(ch.width as usize, ch.height as usize, tracker)
            })
            .collect::<std::result::Result<_, _>>()?;

        Ok(Self {
            header,
            ma_ctx,
            group_dim,
            bit_depth,
            channels,
            meta_channels,
            image_channels,
        })
    }

    pub fn try_clone(&self) -> Result<Self> {
        Ok(Self {
            header: self.header.clone(),
            ma_ctx: self.ma_ctx.clone(),
            group_dim: self.group_dim,
            bit_depth: self.bit_depth,
            channels: self.channels.clone(),
            meta_channels: self
                .meta_channels
                .iter()
                .map(|x| x.try_clone())
                .collect::<std::result::Result<_, _>>()?,
            image_channels: self
                .image_channels
                .iter()
                .map(|x| x.try_clone())
                .collect::<std::result::Result<_, _>>()?,
        })
    }

    pub fn image_channels(&self) -> &[SimpleGrid<i32>] {
        &self.image_channels
    }

    pub fn into_image_channels(self) -> Vec<SimpleGrid<i32>> {
        self.image_channels
    }

    pub fn has_palette(&self) -> bool {
        self.header.transform.iter().any(|tr| tr.is_palette())
    }

    pub fn has_squeeze(&self) -> bool {
        self.header.transform.iter().any(|tr| tr.is_squeeze())
    }
}

impl ModularImageDestination {
    pub fn prepare_gmodular(&mut self) -> Result<TransformedModularSubimage> {
        assert_ne!(self.group_dim, 0);

        let group_dim = self.group_dim;
        let subimage = self.prepare_subimage()?;
        let (channel_info, grids): (Vec<_>, Vec<_>) = subimage
            .channel_info
            .into_iter()
            .zip(subimage.grid)
            .enumerate()
            .take_while(|&(i, (ref info, _))| {
                i < subimage.nb_meta_channels
                    || (info.width <= group_dim && info.height <= group_dim)
            })
            .map(|(_, x)| x)
            .unzip();
        let channel_indices = (0..channel_info.len()).collect();
        Ok(TransformedModularSubimage {
            channel_info,
            channel_indices,
            grid: grids,
            ..subimage
        })
    }

    pub fn prepare_groups(
        &mut self,
        pass_shifts: &std::collections::BTreeMap<u32, (i32, i32)>,
    ) -> Result<TransformedGlobalModular> {
        assert_ne!(self.group_dim, 0);

        let num_passes = *pass_shifts.last_key_value().unwrap().0 as usize + 1;

        let group_dim = self.group_dim;
        let bit_depth = self.bit_depth;
        let subimage = self.prepare_subimage()?;
        let it = subimage
            .channel_info
            .into_iter()
            .zip(subimage.grid)
            .enumerate()
            .skip_while(|&(i, (ref info, _))| {
                i < subimage.nb_meta_channels
                    || (info.width <= group_dim && info.height <= group_dim)
            });

        let mut lf_groups = Vec::new();
        let mut pass_groups = Vec::with_capacity(num_passes);
        pass_groups.resize_with(num_passes, Vec::new);
        for (i, (info, grid)) in it {
            let ModularChannelInfo { hshift, vshift, .. } = info;
            let grid = match grid {
                TransformedGrid::Single(g) => g,
                TransformedGrid::Merged { leader, .. } => leader,
            };
            let (groups, grids) = if hshift < 3 || vshift < 3 {
                let shift = hshift.min(vshift); // shift < 3
                let pass_idx = *pass_shifts
                    .iter()
                    .find(|(_, &(minshift, maxshift))| (minshift..maxshift).contains(&shift))
                    .unwrap()
                    .0;
                let pass_idx = pass_idx as usize;

                let group_width = group_dim >> hshift;
                let group_height = group_dim >> vshift;
                let grids = grid.into_groups(group_width as usize, group_height as usize);
                (&mut pass_groups[pass_idx], grids)
            } else {
                // hshift >= 3 && vshift >= 3
                let lf_group_width = group_dim >> (hshift - 3);
                let lf_group_height = group_dim >> (vshift - 3);
                let grids = grid.into_groups(lf_group_width as usize, lf_group_height as usize);
                (&mut lf_groups, grids)
            };

            if groups.is_empty() {
                groups.resize_with(grids.len(), || {
                    TransformedModularSubimage::empty(&subimage.header, &subimage.ma_ctx, bit_depth)
                });
            } else if groups.len() != grids.len() {
                panic!();
            }

            for (subimage, grid) in groups.iter_mut().zip(grids) {
                let width = grid.width() as u32;
                let height = grid.height() as u32;
                subimage.channel_info.push(ModularChannelInfo {
                    width,
                    height,
                    hshift,
                    vshift,
                });
                subimage.channel_indices.push(i);
                subimage.grid.push(grid.into());
                subimage.partial = true;
            }
        }

        Ok(TransformedGlobalModular {
            lf_groups,
            pass_groups,
        })
    }

    pub fn prepare_subimage(&mut self) -> Result<TransformedModularSubimage> {
        let mut channels = self.channels.clone();
        let mut meta_channel_grids = self
            .meta_channels
            .iter_mut()
            .map(|g| {
                let width = g.width();
                let height = g.height();
                CutGrid::from_buf(g.buf_mut(), width, height, width)
            })
            .collect::<Vec<_>>();
        let mut grids = self
            .image_channels
            .iter_mut()
            .map(|g| {
                let width = g.width();
                let height = g.height();
                CutGrid::from_buf(g.buf_mut(), width, height, width).into()
            })
            .collect::<Vec<_>>();
        for tr in &self.header.transform {
            tr.transform_channels(&mut channels, &mut meta_channel_grids, &mut grids)?;
        }

        let channel_info = channels.info;
        let channel_indices = (0..channel_info.len()).collect();
        Ok(TransformedModularSubimage {
            header: self.header.clone(),
            ma_ctx: self.ma_ctx.clone(),
            bit_depth: self.bit_depth,
            nb_meta_channels: channels.nb_meta_channels as usize,
            channel_info,
            channel_indices,
            grid: grids,
            partial: true,
        })
    }
}

#[derive(Debug)]
pub struct TransformedGlobalModular<'dest> {
    pub lf_groups: Vec<TransformedModularSubimage<'dest>>,
    pub pass_groups: Vec<Vec<TransformedModularSubimage<'dest>>>,
}

#[derive(Debug)]
pub struct TransformedModularSubimage<'dest> {
    header: ModularHeader,
    ma_ctx: MaConfig,
    bit_depth: u32,
    nb_meta_channels: usize,
    channel_info: Vec<ModularChannelInfo>,
    channel_indices: Vec<usize>,
    grid: Vec<TransformedGrid<'dest>>,
    partial: bool,
}

impl<'dest> TransformedModularSubimage<'dest> {
    fn empty(header: &ModularHeader, ma_ctx: &MaConfig, bit_depth: u32) -> Self {
        Self {
            header: header.clone(),
            ma_ctx: ma_ctx.clone(),
            bit_depth,
            nb_meta_channels: 0,
            channel_info: Vec::new(),
            channel_indices: Vec::new(),
            grid: Vec::new(),
            partial: false,
        }
    }
}

impl<'dest> TransformedModularSubimage<'dest> {
    pub fn recursive(
        self,
        bitstream: &mut Bitstream,
        global_ma_config: Option<&MaConfig>,
        tracker: Option<&AllocTracker>,
    ) -> Result<RecursiveModularImage<'dest>> {
        let header = bitstream.read_bundle::<crate::ModularHeader>()?;
        let ma_ctx = if header.use_global_tree {
            global_ma_config
                .ok_or(crate::Error::GlobalMaTreeNotAvailable)?
                .clone()
        } else {
            bitstream.read_bundle::<crate::MaConfig>()?
        };

        let mut image = RecursiveModularImage {
            header,
            ma_ctx,
            bit_depth: self.bit_depth,
            channels: ModularChannels {
                info: self.channel_info,
                nb_meta_channels: 0,
            },
            meta_channels: Vec::new(),
            image_channels: self.grid,
        };
        for tr in &image.header.transform {
            tr.prepare_meta_channels(&mut image.meta_channels, tracker)?;
        }
        Ok(image)
    }

    pub fn finish(mut self, pool: &jxl_threadpool::JxlThreadPool) -> bool {
        for tr in self.header.transform.iter().rev() {
            tr.inverse(&mut self.grid, self.bit_depth, pool);
        }
        !self.partial
    }
}

impl<'dest> TransformedModularSubimage<'dest> {
    fn decode_channel_loop(
        &mut self,
        stream_index: u32,
        mut loop_fn: impl FnMut(
            usize,
            &mut CutGrid<i32>,
            &[&CutGrid<i32>],
            FlatMaTree,
            &crate::predictor::WpHeader,
        ) -> Result<()>,
    ) -> Result<()> {
        let wp_header = &self.header.wp_params;
        let mut prev: Vec<(&ModularChannelInfo, &CutGrid<'dest, i32>)> = Vec::new();
        for (i, (info, grid)) in self.channel_info.iter().zip(&mut self.grid).enumerate() {
            if info.width == 0 || info.height == 0 {
                continue;
            }

            let mut filtered_prev = prev
                .iter()
                .filter(|&(prev_info, _)| {
                    info.width == prev_info.width
                        && info.height == prev_info.height
                        && info.hshift == prev_info.hshift
                        && info.vshift == prev_info.vshift
                })
                .map(|(_, g)| *g)
                .collect::<Vec<_>>();
            filtered_prev.reverse();

            let ma_tree = self.ma_ctx.make_flat_tree(i as u32, stream_index);
            loop_fn(i, grid.grid_mut(), &filtered_prev, ma_tree, wp_header)?;

            prev.push((info, grid.grid()));
        }

        Ok(())
    }

    pub fn decode(
        &mut self,
        bitstream: &mut Bitstream,
        stream_index: u32,
        allow_partial: bool,
    ) -> Result<()> {
        match self.decode_channels_inner(bitstream, stream_index) {
            Err(e) if e.unexpected_eof() && allow_partial => {
                tracing::debug!("Partially decoded Modular image");
            }
            Err(e) => return Err(e),
            Ok(_) => {
                self.partial = false;
            }
        }
        Ok(())
    }

    fn decode_channels_inner(
        &mut self,
        bitstream: &mut Bitstream,
        stream_index: u32,
    ) -> Result<()> {
        let span = tracing::span!(tracing::Level::TRACE, "decode channels", stream_index);
        let _guard = span.enter();

        let mut decoder = self.ma_ctx.decoder().clone();
        decoder.begin(bitstream)?;

        if let Some(rle_decoder) = decoder.as_rle() {
            self.decode_image_rle(bitstream, stream_index, rle_decoder)?;
            decoder.finalize()?;
            return Ok(());
        }
        if let Some(lz77_decoder) = decoder.as_with_lz77() {
            self.decode_image_lz77(bitstream, stream_index, lz77_decoder)?;
            decoder.finalize()?;
            return Ok(());
        }
        let mut no_lz77_decoder = decoder.as_no_lz77().unwrap();

        self.decode_channel_loop(stream_index, |i, grid, prev_rev, ma_tree, wp_header| {
            let width = grid.width();
            let height = grid.height();

            if let Some(&MaTreeLeafClustered {
                cluster,
                predictor,
                offset,
                multiplier,
            }) = ma_tree.single_node()
            {
                tracing::trace!(cluster, ?predictor, "Single MA tree node");
                if predictor == Predictor::Zero {
                    if let Some(token) = no_lz77_decoder.single_token(cluster) {
                        tracing::trace!("Single token in cluster: hyper fast path");
                        let value = unpack_signed(token) * multiplier as i32 + offset;
                        for y in 0..height {
                            for x in 0..width {
                                *grid.get_mut(x, y) = value;
                            }
                        }
                    } else {
                        tracing::trace!("Fast path");
                        for y in 0..height {
                            for x in 0..width {
                                let token =
                                    no_lz77_decoder.read_varint_clustered(bitstream, cluster)?;
                                let value = unpack_signed(token) * multiplier as i32 + offset;
                                *grid.get_mut(x, y) = value;
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

                            let token =
                                no_lz77_decoder.read_varint_clustered(bitstream, cluster)?;
                            let value = ((unpack_signed(token) as i64) + pred) as i32;
                            *grid.get_mut(x, y) = value;
                            *prev = value;
                            nw = n;
                            w = value as i64;
                        }
                    }
                    return Ok(());
                }
            }

            let wp_header = ma_tree.need_self_correcting().then_some(wp_header);
            let mut predictor = PredictorState::new(
                width as u32,
                i as u32,
                stream_index,
                prev_rev.len(),
                wp_header,
            );
            let mut next = |cluster: u8| -> Result<i32> {
                let token = no_lz77_decoder.read_varint_clustered(bitstream, cluster)?;
                Ok(unpack_signed(token))
            };
            decode_channel_slow(&mut next, &ma_tree, &mut predictor, grid, prev_rev)
        })?;

        decoder.finalize()?;
        Ok(())
    }

    fn decode_image_lz77(
        &mut self,
        bitstream: &mut Bitstream,
        stream_index: u32,
        mut decoder: DecoderWithLz77<'_>,
    ) -> Result<()> {
        let dist_multiplier = self
            .channel_info
            .iter()
            .skip_while(|ch| ch.hshift == -1 || ch.vshift == -1)
            .map(|info| info.width)
            .max()
            .unwrap_or(0);

        self.decode_channel_loop(stream_index, |i, grid, prev_rev, ma_tree, wp_header| {
            let width = grid.width();
            let height = grid.height();

            if let Some(&MaTreeLeafClustered {
                cluster,
                predictor,
                offset,
                multiplier,
            }) = ma_tree.single_node()
            {
                tracing::trace!(cluster, ?predictor, "Single MA tree node");
                if predictor == Predictor::Zero {
                    tracing::trace!("Fast path");
                    for y in 0..height {
                        for x in 0..width {
                            let token = decoder.read_varint_with_multiplier_clustered(
                                bitstream,
                                cluster,
                                dist_multiplier,
                            )?;
                            let value = unpack_signed(token) * multiplier as i32 + offset;
                            *grid.get_mut(x, y) = value;
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

                            let token = decoder.read_varint_with_multiplier_clustered(
                                bitstream,
                                cluster,
                                dist_multiplier,
                            )?;
                            let value = ((unpack_signed(token) as i64) + pred) as i32;
                            *grid.get_mut(x, y) = value;
                            *prev = value;
                            nw = n;
                            w = value as i64;
                        }
                    }
                    return Ok(());
                }
            }

            let wp_header = ma_tree.need_self_correcting().then_some(wp_header);
            let mut predictor = PredictorState::new(
                width as u32,
                i as u32,
                stream_index,
                prev_rev.len(),
                wp_header,
            );
            let mut next = |cluster: u8| -> Result<i32> {
                let token = decoder.read_varint_with_multiplier_clustered(
                    bitstream,
                    cluster,
                    dist_multiplier,
                )?;
                Ok(unpack_signed(token))
            };
            decode_channel_slow(&mut next, &ma_tree, &mut predictor, grid, prev_rev)
        })
    }

    fn decode_image_rle(
        &mut self,
        bitstream: &mut Bitstream,
        stream_index: u32,
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
                    }
                    RleToken::Repeat(len) => {
                        rle_left = len - 1;
                        rle_value
                    }
                }
            })
        };

        self.decode_channel_loop(stream_index, |i, grid, prev_rev, ma_tree, wp_header| {
            let width = grid.width();
            let height = grid.height();

            if let Some(&MaTreeLeafClustered {
                cluster,
                predictor,
                offset,
                multiplier,
            }) = ma_tree.single_node()
            {
                tracing::trace!(cluster, ?predictor, "Single MA tree node");
                if predictor == Predictor::Zero {
                    tracing::trace!("Quite fast path");
                    for y in 0..height {
                        for x in 0..width {
                            let token = next(cluster)?;
                            let value = (token * multiplier as i32) + offset;
                            *grid.get_mut(x, y) = value;
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
                            *grid.get_mut(x, y) = value;
                            *prev = value;
                            nw = n;
                            w = value as i64;
                        }
                    }
                    return Ok(());
                }
            }

            let wp_header = ma_tree.need_self_correcting().then_some(wp_header);
            let mut predictor = PredictorState::new(
                width as u32,
                i as u32,
                stream_index,
                prev_rev.len(),
                wp_header,
            );
            decode_channel_slow(&mut next, &ma_tree, &mut predictor, grid, prev_rev)
        })
    }
}

fn decode_channel_slow(
    next: &mut impl FnMut(u8) -> Result<i32>,
    ma_tree: &FlatMaTree,
    predictor: &mut PredictorState,
    grid: &mut CutGrid<i32>,
    prev_rev: &[&CutGrid<i32>],
) -> Result<()> {
    let width = grid.width();
    let height = grid.height();

    let mut prev_channel_samples_rev = vec![0i32; prev_rev.len()];

    for y in 0..height {
        for x in 0..width {
            for (grid, sample) in prev_rev.iter().zip(&mut prev_channel_samples_rev) {
                *sample = grid.get(x, y);
            }

            let properties = predictor.properties(&prev_channel_samples_rev);
            let (diff, predictor) = ma_tree.decode_sample_rle(next, &properties)?;
            let sample_prediction = predictor.predict(&properties);
            let true_value = (diff as i64 + sample_prediction) as i32;
            *grid.get_mut(x, y) = true_value;
            properties.record(true_value);
        }
    }

    Ok(())
}

#[derive(Debug)]
pub struct RecursiveModularImage<'dest> {
    header: ModularHeader,
    ma_ctx: MaConfig,
    bit_depth: u32,
    channels: ModularChannels,
    meta_channels: Vec<SimpleGrid<i32>>,
    image_channels: Vec<TransformedGrid<'dest>>,
}

impl<'dest> RecursiveModularImage<'dest> {
    pub fn prepare_subimage(&mut self) -> Result<TransformedModularSubimage> {
        let mut channels = self.channels.clone();
        let mut meta_channel_grids = self
            .meta_channels
            .iter_mut()
            .map(|g| {
                let width = g.width();
                let height = g.height();
                CutGrid::from_buf(g.buf_mut(), width, height, width)
            })
            .collect::<Vec<_>>();
        let mut grids = self
            .image_channels
            .iter_mut()
            .map(|g| g.reborrow())
            .collect();
        for tr in &self.header.transform {
            tr.transform_channels(&mut channels, &mut meta_channel_grids, &mut grids)?;
        }

        let channel_info = channels.info;
        let channel_indices = (0..channel_info.len()).collect();
        Ok(TransformedModularSubimage {
            header: self.header.clone(),
            ma_ctx: self.ma_ctx.clone(),
            bit_depth: self.bit_depth,
            nb_meta_channels: channels.nb_meta_channels as usize,
            channel_info,
            channel_indices,
            grid: grids,
            partial: true,
        })
    }
}
