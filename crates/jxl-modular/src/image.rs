use jxl_bitstream::{unpack_signed, Bitstream};
use jxl_coding::{DecoderRleMode, DecoderWithLz77, RleToken};
use jxl_grid::{AllocTracker, CutGrid, SimpleGrid};

use crate::{
    ma::{FlatMaTree, MaTreeLeafClustered},
    predictor::{Predictor, PredictorState},
    sample::{Sample, Sealed},
    MaConfig, ModularChannelInfo, ModularChannels, ModularHeader, Result,
};

#[derive(Debug)]
pub enum TransformedGrid<'dest, S: Sample> {
    Single(CutGrid<'dest, S>),
    Merged {
        leader: CutGrid<'dest, S>,
        members: Vec<TransformedGrid<'dest, S>>,
    },
}

impl<'dest, S: Sample> From<CutGrid<'dest, S>> for TransformedGrid<'dest, S> {
    fn from(value: CutGrid<'dest, S>) -> Self {
        Self::Single(value)
    }
}

impl<S: Sample> TransformedGrid<'_, S> {
    fn reborrow(&mut self) -> TransformedGrid<S> {
        match self {
            TransformedGrid::Single(g) => TransformedGrid::Single(g.split_horizontal(0).1),
            TransformedGrid::Merged { leader, .. } => {
                TransformedGrid::Single(leader.split_horizontal(0).1)
            }
        }
    }
}

impl<'dest, S: Sample> TransformedGrid<'dest, S> {
    pub(crate) fn grid(&self) -> &CutGrid<'dest, S> {
        match self {
            Self::Single(g) => g,
            Self::Merged { leader, .. } => leader,
        }
    }

    pub(crate) fn grid_mut(&mut self) -> &mut CutGrid<'dest, S> {
        match self {
            Self::Single(g) => g,
            Self::Merged { leader, .. } => leader,
        }
    }

    pub(crate) fn merge(&mut self, members: Vec<TransformedGrid<'dest, S>>) {
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

    pub(crate) fn unmerge(&mut self, count: usize) -> Vec<TransformedGrid<'dest, S>> {
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
pub struct ModularImageDestination<S: Sample> {
    header: ModularHeader,
    ma_ctx: MaConfig,
    group_dim: u32,
    bit_depth: u32,
    channels: ModularChannels,
    meta_channels: Vec<SimpleGrid<S>>,
    image_channels: Vec<SimpleGrid<S>>,
}

impl<S: Sample> ModularImageDestination<S> {
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

    pub fn image_channels(&self) -> &[SimpleGrid<S>] {
        &self.image_channels
    }

    pub fn into_image_channels(self) -> Vec<SimpleGrid<S>> {
        self.image_channels
    }

    pub fn has_palette(&self) -> bool {
        self.header.transform.iter().any(|tr| tr.is_palette())
    }

    pub fn has_squeeze(&self) -> bool {
        self.header.transform.iter().any(|tr| tr.is_squeeze())
    }
}

impl<S: Sample> ModularImageDestination<S> {
    pub fn prepare_gmodular(&mut self) -> Result<TransformedModularSubimage<S>> {
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
    ) -> Result<TransformedGlobalModular<S>> {
        assert_ne!(self.group_dim, 0);

        let num_passes = *pass_shifts.last_key_value().unwrap().0 as usize + 1;

        let group_dim = self.group_dim;
        let group_dim_shift = group_dim.trailing_zeros();
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
            let ModularChannelInfo {
                original_width,
                original_height,
                hshift,
                vshift,
                ..
            } = info;
            assert!(hshift >= 0 && vshift >= 0);

            let grid = match grid {
                TransformedGrid::Single(g) => g,
                TransformedGrid::Merged { leader, .. } => leader,
            };
            tracing::trace!(
                i,
                width = grid.width(),
                height = grid.height(),
                hshift,
                vshift
            );

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
                if group_width == 0 || group_height == 0 {
                    tracing::error!(
                        group_dim,
                        hshift,
                        vshift,
                        "Channel shift value too large after transform"
                    );
                    return Err(crate::Error::InvalidSqueezeParams);
                }

                let grids = grid.into_groups_with_fixed_count(
                    group_width as usize,
                    group_height as usize,
                    (original_width + group_dim - 1) as usize >> group_dim_shift,
                    (original_height + group_dim - 1) as usize >> group_dim_shift,
                );
                (&mut pass_groups[pass_idx], grids)
            } else {
                // hshift >= 3 && vshift >= 3
                let lf_group_width = group_dim >> (hshift - 3);
                let lf_group_height = group_dim >> (vshift - 3);
                if lf_group_width == 0 || lf_group_height == 0 {
                    tracing::error!(
                        group_dim,
                        hshift,
                        vshift,
                        "Channel shift value too large after transform"
                    );
                    return Err(crate::Error::InvalidSqueezeParams);
                }
                let grids = grid.into_groups_with_fixed_count(
                    lf_group_width as usize,
                    lf_group_height as usize,
                    (original_width + (group_dim << 3) - 1) as usize >> (group_dim_shift + 3),
                    (original_height + (group_dim << 3) - 1) as usize >> (group_dim_shift + 3),
                );
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
                if width == 0 || height == 0 {
                    continue;
                }

                subimage.channel_info.push(ModularChannelInfo {
                    width,
                    height,
                    original_width: width << hshift,
                    original_height: height << vshift,
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

    pub fn prepare_subimage(&mut self) -> Result<TransformedModularSubimage<S>> {
        let mut channels = self.channels.clone();
        let mut meta_channel_grids = self
            .meta_channels
            .iter_mut()
            .map(CutGrid::from_simple_grid)
            .collect::<Vec<_>>();
        let mut grids = self
            .image_channels
            .iter_mut()
            .map(|g| CutGrid::from_simple_grid(g).into())
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
pub struct TransformedGlobalModular<'dest, S: Sample> {
    pub lf_groups: Vec<TransformedModularSubimage<'dest, S>>,
    pub pass_groups: Vec<Vec<TransformedModularSubimage<'dest, S>>>,
}

#[derive(Debug)]
pub struct TransformedModularSubimage<'dest, S: Sample> {
    header: ModularHeader,
    ma_ctx: MaConfig,
    bit_depth: u32,
    nb_meta_channels: usize,
    channel_info: Vec<ModularChannelInfo>,
    channel_indices: Vec<usize>,
    grid: Vec<TransformedGrid<'dest, S>>,
    partial: bool,
}

impl<'dest, S: Sample> TransformedModularSubimage<'dest, S> {
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

impl<'dest, S: Sample> TransformedModularSubimage<'dest, S> {
    pub fn is_empty(&self) -> bool {
        self.channel_info.is_empty()
    }

    pub fn recursive(
        self,
        bitstream: &mut Bitstream,
        global_ma_config: Option<&MaConfig>,
        tracker: Option<&AllocTracker>,
    ) -> Result<RecursiveModularImage<'dest, S>> {
        let channels = crate::ModularChannels {
            info: self.channel_info,
            nb_meta_channels: 0,
        };
        let (header, ma_ctx) = crate::read_and_validate_local_modular_header(
            bitstream,
            &channels,
            global_ma_config,
            tracker,
        )?;

        let mut image = RecursiveModularImage {
            header,
            ma_ctx,
            bit_depth: self.bit_depth,
            channels,
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

impl<'dest, S: Sample> TransformedModularSubimage<'dest, S> {
    fn decode_channel_loop<'image>(
        &'image mut self,
        stream_index: u32,
        mut loop_fn: impl FnMut(
            &mut CutGrid<S>,
            &[&'image CutGrid<S>],
            FlatMaTree,
            &crate::predictor::WpHeader,
        ) -> Result<()>,
    ) -> Result<()> {
        let wp_header = &self.header.wp_params;
        let mut prev: Vec<(&ModularChannelInfo, &CutGrid<'dest, S>)> = Vec::new();
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

            let ma_tree =
                self.ma_ctx
                    .make_flat_tree(i as u32, stream_index, filtered_prev.len() as u32);
            let filtered_prev = &filtered_prev[..ma_tree.max_prev_channel_depth()];
            loop_fn(grid.grid_mut(), filtered_prev, ma_tree, wp_header)?;

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

        let mut predictor = PredictorState::new();
        self.decode_channel_loop(stream_index, |grid, prev_rev, ma_tree, wp_header| {
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
                        let value = S::unpack_signed_u32(token)
                            .wrapping_muladd_i32(multiplier as i32, offset);
                        for y in 0..height {
                            grid.get_row_mut(y).fill(value);
                        }
                    } else {
                        tracing::trace!("Fast path");
                        for y in 0..height {
                            let row = grid.get_row_mut(y);
                            for out in row {
                                let token =
                                    no_lz77_decoder.read_varint_clustered(bitstream, cluster)?;
                                let value = S::unpack_signed_u32(token)
                                    .wrapping_muladd_i32(multiplier as i32, offset);
                                *out = value;
                            }
                        }
                    }
                    return Ok(());
                }
                if predictor == Predictor::Gradient && offset == 0 && multiplier == 1 {
                    tracing::trace!("Quite fast path");
                    let mut prev_row_cache = Vec::with_capacity(width);

                    {
                        let mut w = 0i64;
                        let out_row = grid.get_row_mut(0);
                        for out in out_row[..width].iter_mut() {
                            let pred = w;

                            let token =
                                no_lz77_decoder.read_varint_clustered(bitstream, cluster)?;
                            let value = S::unpack_signed_u32(token).add(S::from_i32(pred as i32));
                            *out = value;
                            prev_row_cache.push(value.to_i64());
                            w = value.to_i64();
                        }
                    }

                    for y in 1..height {
                        let out_row = grid.get_row_mut(y);

                        let n = prev_row_cache[0];
                        let pred = n;
                        let token = no_lz77_decoder.read_varint_clustered(bitstream, cluster)?;
                        let value = S::unpack_signed_u32(token).add(S::from_i32(pred as i32));
                        out_row[0] = value;
                        prev_row_cache[0] = value.to_i64();

                        let mut w = value.to_i64();
                        let mut nw = n;
                        for (prev, out) in prev_row_cache[1..].iter_mut().zip(&mut out_row[1..]) {
                            let n = *prev;
                            let pred = (n + w - nw).clamp(w.min(n), w.max(n));

                            let token =
                                no_lz77_decoder.read_varint_clustered(bitstream, cluster)?;
                            let value = S::unpack_signed_u32(token).add(S::from_i32(pred as i32));
                            *out = value;
                            *prev = value.to_i64();
                            nw = n;
                            w = value.to_i64();
                        }
                    }
                    return Ok(());
                }
            }

            let wp_header = ma_tree.need_self_correcting().then_some(wp_header);
            predictor.reset(width as u32, prev_rev, wp_header);
            let mut next = |cluster: u8| -> Result<i32> {
                let token = no_lz77_decoder.read_varint_clustered(bitstream, cluster)?;
                Ok(unpack_signed(token))
            };
            decode_channel_slow(&mut next, &ma_tree, &mut predictor, grid)
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
        let mut next = |cluster: u8| -> Result<i32> {
            let token = decoder.read_varint_with_multiplier_clustered(
                bitstream,
                cluster,
                dist_multiplier,
            )?;
            Ok(unpack_signed(token))
        };

        let mut predictor = PredictorState::new();
        self.decode_channel_loop(stream_index, |grid, prev_rev, ma_tree, wp_header| {
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
                        let row = grid.get_row_mut(y);
                        for out in row {
                            let token = next(cluster)?;
                            let value = token.wrapping_muladd_i32(multiplier as i32, offset);
                            *out = S::from_i32(value);
                        }
                    }
                    return Ok(());
                }
                if predictor == Predictor::Gradient && offset == 0 && multiplier == 1 {
                    tracing::trace!("Quite fast path");
                    let mut prev_row_cache = Vec::with_capacity(width);

                    {
                        let mut w = 0i32;
                        let out_row = grid.get_row_mut(0);
                        for out in out_row[..width].iter_mut() {
                            let pred = w;

                            let token = next(cluster)?;
                            let value = token.wrapping_add(pred);
                            *out = S::from_i32(value);
                            prev_row_cache.push(value);
                            w = value;
                        }
                    }

                    for y in 1..height {
                        let out_row = grid.get_row_mut(y);

                        let n = prev_row_cache[0];
                        let pred = n;
                        let token = next(cluster)?;
                        let value = token.wrapping_add(pred);
                        out_row[0] = S::from_i32(value);
                        prev_row_cache[0] = value;

                        let mut w = value as i64;
                        let mut nw = n as i64;
                        for (prev, out) in prev_row_cache[1..].iter_mut().zip(&mut out_row[1..]) {
                            let n = *prev as i64;
                            let pred = (n + w - nw).clamp(w.min(n), w.max(n)) as i32;

                            let token = next(cluster)?;
                            let value = token.wrapping_add(pred);
                            *out = S::from_i32(value);
                            *prev = value;
                            nw = n;
                            w = value as i64;
                        }
                    }
                    return Ok(());
                }
            }

            let wp_header = ma_tree.need_self_correcting().then_some(wp_header);
            predictor.reset(width as u32, prev_rev, wp_header);
            decode_channel_slow(&mut next, &ma_tree, &mut predictor, grid)
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

        let mut predictor = PredictorState::new();
        self.decode_channel_loop(stream_index, |grid, prev_rev, ma_tree, wp_header| {
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
                        let row = grid.get_row_mut(y);
                        for out in row {
                            let token = next(cluster)?;
                            let value = token.wrapping_muladd_i32(multiplier as i32, offset);
                            *out = S::from_i32(value);
                        }
                    }
                    return Ok(());
                }
                if predictor == Predictor::Gradient && offset == 0 && multiplier == 1 {
                    tracing::trace!("libjxl fast-lossless: quite fast path");
                    let mut prev_row_cache = Vec::with_capacity(width);

                    {
                        let mut w = 0i32;
                        let out_row = grid.get_row_mut(0);
                        for out in out_row[..width].iter_mut() {
                            let pred = w;

                            let token = next(cluster)?;
                            let value = token.wrapping_add(pred);
                            *out = S::from_i32(value);
                            prev_row_cache.push(value);
                            w = value;
                        }
                    }

                    for y in 1..height {
                        let out_row = grid.get_row_mut(y);

                        let n = prev_row_cache[0];
                        let pred = n;
                        let token = next(cluster)?;
                        let value = token.wrapping_add(pred);
                        out_row[0] = S::from_i32(value);
                        prev_row_cache[0] = value;

                        let mut w = value as i64;
                        let mut nw = n as i64;
                        for (prev, out) in prev_row_cache[1..].iter_mut().zip(&mut out_row[1..]) {
                            let n = *prev as i64;
                            let pred = (n + w - nw).clamp(w.min(n), w.max(n)) as i32;

                            let token = next(cluster)?;
                            let value = token.wrapping_add(pred);
                            *out = S::from_i32(value);
                            *prev = value;
                            nw = n;
                            w = value as i64;
                        }
                    }
                    return Ok(());
                }
            }

            let wp_header = ma_tree.need_self_correcting().then_some(wp_header);
            predictor.reset(width as u32, prev_rev, wp_header);
            decode_channel_slow(&mut next, &ma_tree, &mut predictor, grid)
        })
    }
}

fn decode_channel_slow<S: Sample>(
    next: &mut impl FnMut(u8) -> Result<i32>,
    ma_tree: &FlatMaTree,
    predictor: &mut PredictorState<S>,
    grid: &mut CutGrid<S>,
) -> Result<()> {
    let height = grid.height();
    for y in 0..height {
        let row = grid.get_row_mut(y);

        for out in row.iter_mut() {
            let properties = predictor.properties();
            let (diff, predictor) = ma_tree.decode_sample_with_fn(next, &properties)?;
            let sample_prediction = predictor.predict(&properties);
            let true_value = diff.wrapping_add(sample_prediction);
            *out = S::from_i32(true_value);
            properties.record(true_value);
        }
    }

    Ok(())
}

#[derive(Debug)]
pub struct RecursiveModularImage<'dest, S: Sample> {
    header: ModularHeader,
    ma_ctx: MaConfig,
    bit_depth: u32,
    channels: ModularChannels,
    meta_channels: Vec<SimpleGrid<S>>,
    image_channels: Vec<TransformedGrid<'dest, S>>,
}

impl<'dest, S: Sample> RecursiveModularImage<'dest, S> {
    pub fn prepare_subimage(&mut self) -> Result<TransformedModularSubimage<S>> {
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
