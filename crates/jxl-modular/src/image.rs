use std::collections::HashMap;

use jxl_bitstream::Bitstream;
use jxl_coding::{Decoder, DecoderRleMode, RleToken};
use jxl_grid::{AlignedGrid, AllocTracker, MutableSubgrid};

use crate::{
    MaConfig, ModularChannelInfo, ModularChannels, ModularHeader, Result,
    ma::{FlatMaTree, MaTreeLeafClustered, SimpleMaTable},
    predictor::{Predictor, PredictorState, Properties, WpHeader},
    sample::Sample,
};

#[derive(Debug)]
pub enum TransformedGrid<'dest, S: Sample> {
    Single(MutableSubgrid<'dest, S>),
    Merged {
        leader: MutableSubgrid<'dest, S>,
        members: Vec<TransformedGrid<'dest, S>>,
    },
}

impl<'dest, S: Sample> From<MutableSubgrid<'dest, S>> for TransformedGrid<'dest, S> {
    fn from(value: MutableSubgrid<'dest, S>) -> Self {
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
    pub(crate) fn grid(&self) -> &MutableSubgrid<'dest, S> {
        match self {
            Self::Single(g) => g,
            Self::Merged { leader, .. } => leader,
        }
    }

    pub(crate) fn grid_mut(&mut self) -> &mut MutableSubgrid<'dest, S> {
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
                let tmp = MutableSubgrid::empty();
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
                    let tmp = MutableSubgrid::empty();
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
    meta_channels: Vec<AlignedGrid<S>>,
    image_channels: Vec<AlignedGrid<S>>,
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
                AlignedGrid::with_alloc_tracker(ch.width as usize, ch.height as usize, tracker)
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

    pub fn image_channels(&self) -> &[AlignedGrid<S>] {
        &self.image_channels
    }

    pub fn into_image_channels(self) -> Vec<AlignedGrid<S>> {
        self.image_channels
    }

    pub fn into_image_channels_with_info(
        self,
    ) -> impl Iterator<Item = (AlignedGrid<S>, ModularChannelInfo)> {
        self.image_channels.into_iter().zip(self.channels.info)
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
                original_shift,
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
                    .find(|&(_, &(minshift, maxshift))| (minshift..maxshift).contains(&shift))
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
                    original_shift,
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
            .map(MutableSubgrid::from)
            .collect::<Vec<_>>();
        let mut grids = self
            .image_channels
            .iter_mut()
            .map(|g| g.as_subgrid_mut().into())
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

impl<S: Sample> TransformedModularSubimage<'_, S> {
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

impl<S: Sample> TransformedModularSubimage<'_, S> {
    fn decode_inner(&mut self, bitstream: &mut Bitstream, stream_index: u32) -> Result<()> {
        let span = tracing::span!(tracing::Level::TRACE, "decode channels", stream_index);
        let _guard = span.enter();

        let dist_multiplier = self
            .channel_info
            .iter()
            .map(|info| info.width)
            .max()
            .unwrap_or(0);

        let mut decoder = self.ma_ctx.decoder().clone();
        decoder.begin(bitstream)?;

        let mut ma_tree_list = Vec::with_capacity(self.channel_info.len());
        for (i, info) in self.channel_info.iter().enumerate() {
            if info.width == 0 || info.height == 0 {
                ma_tree_list.push(None);
                continue;
            }

            let filtered_prev_len = self.channel_info[..i]
                .iter()
                .filter(|prev_info| {
                    info.width == prev_info.width
                        && info.height == prev_info.height
                        && info.hshift == prev_info.hshift
                        && info.vshift == prev_info.vshift
                })
                .count();

            let ma_tree =
                self.ma_ctx
                    .make_flat_tree(i as u32, stream_index, filtered_prev_len as u32);
            ma_tree_list.push(Some(ma_tree));
        }

        if let Some(mut rle_decoder) = decoder.as_rle() {
            let is_fast_lossless = ma_tree_list.iter().all(|ma_tree| {
                ma_tree
                    .as_ref()
                    .map(|ma_tree| {
                        matches!(
                            ma_tree.single_node(),
                            Some(MaTreeLeafClustered {
                                predictor: Predictor::Gradient,
                                offset: 0,
                                multiplier: 1,
                                ..
                            })
                        )
                    })
                    .unwrap_or(true)
            });

            if is_fast_lossless {
                tracing::trace!("libjxl fast-lossless");
                let mut rle_state = RleState::<S>::new();

                for (ma_tree, grid) in ma_tree_list.into_iter().zip(&mut self.grid) {
                    let Some(ma_tree) = ma_tree else {
                        continue;
                    };

                    let node = ma_tree.single_node().unwrap();
                    let cluster = node.cluster;
                    decode_fast_lossless(
                        bitstream,
                        &mut rle_decoder,
                        &mut rle_state,
                        cluster,
                        grid.grid_mut(),
                    );
                }

                rle_state.check_error()?;
                // Prefix code doesn't have checksum
                return Ok(());
            }
        }

        let wp_header = &self.header.wp_params;
        let mut predictor = PredictorState::new();
        let mut prev_map = HashMap::new();
        for ((info, ma_tree), grid) in self
            .channel_info
            .iter()
            .zip(ma_tree_list)
            .zip(&mut self.grid)
        {
            let Some(ma_tree) = ma_tree else {
                continue;
            };
            let key = (info.width, info.height, info.hshift, info.vshift);

            let filtered_prev = prev_map.entry(key).or_insert_with(Vec::new);

            if let Some(node) = ma_tree.single_node() {
                decode_single_node(
                    bitstream,
                    &mut decoder,
                    dist_multiplier,
                    &mut predictor,
                    wp_header,
                    grid.grid_mut(),
                    node,
                )?;
            } else if let Some(table) = ma_tree.simple_table() {
                decode_simple_table(
                    bitstream,
                    &mut decoder,
                    dist_multiplier,
                    &mut predictor,
                    wp_header,
                    grid.grid_mut(),
                    &table,
                )?;
            } else {
                let grid = grid.grid_mut();
                let filtered_prev = &filtered_prev[..ma_tree.max_prev_channel_depth()];
                let wp_header = ma_tree.need_self_correcting().then_some(wp_header);
                predictor.reset(grid.width() as u32, filtered_prev, wp_header);
                decode_slow(
                    bitstream,
                    &mut decoder,
                    dist_multiplier,
                    &ma_tree,
                    &mut predictor,
                    grid,
                )?;
            }

            filtered_prev.insert(0, grid.grid());
        }

        decoder.finalize()?;
        Ok(())
    }

    pub fn decode(
        &mut self,
        bitstream: &mut Bitstream,
        stream_index: u32,
        allow_partial: bool,
    ) -> Result<()> {
        match self.decode_inner(bitstream, stream_index) {
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
}

#[derive(Debug)]
pub struct RecursiveModularImage<'dest, S: Sample> {
    header: ModularHeader,
    ma_ctx: MaConfig,
    bit_depth: u32,
    channels: ModularChannels,
    meta_channels: Vec<AlignedGrid<S>>,
    image_channels: Vec<TransformedGrid<'dest, S>>,
}

impl<S: Sample> RecursiveModularImage<'_, S> {
    pub fn prepare_subimage(&mut self) -> Result<TransformedModularSubimage<S>> {
        let mut channels = self.channels.clone();
        let mut meta_channel_grids = self
            .meta_channels
            .iter_mut()
            .map(|g| {
                let width = g.width();
                let height = g.height();
                MutableSubgrid::from_buf(g.buf_mut(), width, height, width)
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

struct RleState<S: Sample> {
    value: S,
    repeat: u32,
    error: Option<Box<jxl_coding::Error>>,
}

impl<S: Sample> RleState<S> {
    #[inline]
    fn new() -> Self {
        Self {
            value: S::default(),
            repeat: 0,
            error: None,
        }
    }

    #[inline(always)]
    fn decode(
        &mut self,
        bitstream: &mut Bitstream,
        decoder: &mut DecoderRleMode,
        cluster: u8,
    ) -> S {
        if self.repeat == 0 {
            let result = decoder.read_varint_clustered(bitstream, cluster);
            match result {
                Ok(RleToken::Value(v)) => {
                    self.value = S::unpack_signed_u32(v);
                    self.repeat = 1;
                }
                Ok(RleToken::Repeat(len)) => {
                    self.repeat = len;
                }
                Err(e) if self.error.is_none() => {
                    self.error = Some(Box::new(e));
                }
                _ => {}
            }
        }

        self.repeat = self.repeat.wrapping_sub(1);
        self.value
    }

    #[inline]
    fn check_error(&mut self) -> Result<()> {
        let error = self.error.take();
        if let Some(error) = error {
            let error = *error;
            Err(error.into())
        } else {
            Ok(())
        }
    }
}

fn decode_single_node<S: Sample>(
    bitstream: &mut Bitstream,
    decoder: &mut Decoder,
    dist_multiplier: u32,
    predictor_state: &mut PredictorState<S>,
    wp_header: &WpHeader,
    grid: &mut MutableSubgrid<S>,
    node: &MaTreeLeafClustered,
) -> Result<()> {
    let &MaTreeLeafClustered {
        cluster,
        predictor,
        offset,
        multiplier,
    } = node;
    tracing::trace!(cluster, ?predictor, "Single MA tree node");

    let height = grid.height();
    let single_token = decoder.single_token(cluster);
    match (predictor, single_token) {
        (Predictor::Zero, Some(token)) => {
            tracing::trace!("Single token in cluster, Zero predictor: hyper fast path");
            let value = S::unpack_signed_u32(token).wrapping_muladd_i32(multiplier as i32, offset);
            for y in 0..height {
                grid.get_row_mut(y).fill(value);
            }
            Ok(())
        }
        (Predictor::Zero, None) => {
            tracing::trace!("Zero predictor: fast path");
            for y in 0..height {
                let row = grid.get_row_mut(y);
                for out in row {
                    let token = decoder.read_varint_with_multiplier_clustered(
                        bitstream,
                        cluster,
                        dist_multiplier,
                    )?;
                    *out =
                        S::unpack_signed_u32(token).wrapping_muladd_i32(multiplier as i32, offset);
                }
            }
            Ok(())
        }
        (Predictor::Gradient, _) if offset == 0 && multiplier == 1 => {
            tracing::trace!("Simple gradient: quite fast path");
            decode_simple_grad(bitstream, decoder, cluster, dist_multiplier, grid)
        }
        _ => {
            let wp_header = (predictor == Predictor::SelfCorrecting).then_some(wp_header);
            predictor_state.reset(grid.width() as u32, &[], wp_header);
            decode_single_node_slow(
                bitstream,
                decoder,
                dist_multiplier,
                node,
                predictor_state,
                grid,
            )
        }
    }
}

#[inline(never)]
fn decode_fast_lossless<S: Sample>(
    bitstream: &mut Bitstream,
    decoder: &mut DecoderRleMode,
    rle_state: &mut RleState<S>,
    cluster: u8,
    grid: &mut MutableSubgrid<S>,
) {
    let height = grid.height();

    {
        let mut w = S::default();
        let out_row = grid.get_row_mut(0);
        for out in &mut *out_row {
            let token = rle_state.decode(bitstream, decoder, cluster);
            w = w.add(token);
            *out = w;
        }
    }

    for y in 1..height {
        let (u, mut d) = grid.split_vertical(y);
        let prev_row = u.get_row(y - 1);
        let out_row = d.get_row_mut(0);

        let token = rle_state.decode(bitstream, decoder, cluster);
        let mut w = token.add(prev_row[0]);
        out_row[0] = w;

        for (window, out) in prev_row.windows(2).zip(&mut out_row[1..]) {
            let nw = window[0];
            let n = window[1];
            let pred = S::grad_clamped(n, w, nw);

            let token = rle_state.decode(bitstream, decoder, cluster);
            w = token.add(pred);
            *out = w;
        }
    }
}

#[inline(never)]
fn decode_simple_grad<S: Sample>(
    bitstream: &mut Bitstream,
    decoder: &mut Decoder,
    cluster: u8,
    dist_multiplier: u32,
    grid: &mut MutableSubgrid<S>,
) -> Result<()> {
    let width = grid.width();
    let height = grid.height();

    {
        let mut w = S::default();
        let out_row = grid.get_row_mut(0);
        for out in out_row[..width].iter_mut() {
            let token = decoder.read_varint_with_multiplier_clustered(
                bitstream,
                cluster,
                dist_multiplier,
            )?;
            w = S::unpack_signed_u32(token).add(w);
            *out = w;
        }
    }

    for y in 1..height {
        let (u, mut d) = grid.split_vertical(y);
        let prev_row = u.get_row(y - 1);
        let out_row = d.get_row_mut(0);

        let token =
            decoder.read_varint_with_multiplier_clustered(bitstream, cluster, dist_multiplier)?;
        let mut w = S::unpack_signed_u32(token).add(prev_row[0]);
        out_row[0] = w;

        for (window, out) in prev_row.windows(2).zip(&mut out_row[1..]) {
            let nw = window[0];
            let n = window[1];
            let pred = S::grad_clamped(n, w, nw);

            let token = decoder.read_varint_with_multiplier_clustered(
                bitstream,
                cluster,
                dist_multiplier,
            )?;
            let value = S::unpack_signed_u32(token).add(pred);
            *out = value;
            w = value;
        }
    }

    Ok(())
}

#[inline(always)]
fn decode_one<S: Sample, const EDGE: bool>(
    bitstream: &mut Bitstream,
    decoder: &mut Decoder,
    dist_multiplier: u32,
    leaf: &MaTreeLeafClustered,
    properties: &Properties<S>,
) -> Result<S> {
    let diff = S::unpack_signed_u32(decoder.read_varint_with_multiplier_clustered(
        bitstream,
        leaf.cluster,
        dist_multiplier,
    )?);
    let diff = diff.wrapping_muladd_i32(leaf.multiplier as i32, leaf.offset);
    let predictor = leaf.predictor;
    let sample_prediction = predictor.predict::<_, EDGE>(properties);
    Ok(diff.add(S::from_i32(sample_prediction)))
}

#[inline(never)]
fn decode_single_node_slow<S: Sample>(
    bitstream: &mut Bitstream,
    decoder: &mut Decoder,
    dist_multiplier: u32,
    leaf: &MaTreeLeafClustered,
    predictor: &mut PredictorState<S>,
    grid: &mut MutableSubgrid<S>,
) -> Result<()> {
    let height = grid.height();
    for y in 0..2usize.min(height) {
        let row = grid.get_row_mut(y);

        for out in row.iter_mut() {
            let properties = predictor.properties::<true>();
            let true_value =
                decode_one::<_, true>(bitstream, decoder, dist_multiplier, leaf, &properties)?;
            *out = true_value;
            properties.record(true_value.to_i32());
        }
    }

    for y in 2..height {
        let row = grid.get_row_mut(y);
        let (row_left, row_middle, row_right) = if row.len() <= 4 {
            (row, [].as_mut(), [].as_mut())
        } else {
            let (l, m) = row.split_at_mut(2);
            let (m, r) = m.split_at_mut(m.len() - 2);
            (l, m, r)
        };

        for out in row_left {
            let properties = predictor.properties::<true>();
            let true_value =
                decode_one::<_, true>(bitstream, decoder, dist_multiplier, leaf, &properties)?;
            *out = true_value;
            properties.record(true_value.to_i32());
        }
        for out in row_middle {
            let properties = predictor.properties::<false>();
            let true_value =
                decode_one::<_, false>(bitstream, decoder, dist_multiplier, leaf, &properties)?;
            *out = true_value;
            properties.record(true_value.to_i32());
        }
        for out in row_right {
            let properties = predictor.properties::<true>();
            let true_value =
                decode_one::<_, true>(bitstream, decoder, dist_multiplier, leaf, &properties)?;
            *out = true_value;
            properties.record(true_value.to_i32());
        }
    }

    Ok(())
}

fn decode_simple_table<S: Sample>(
    bitstream: &mut Bitstream,
    decoder: &mut Decoder,
    dist_multiplier: u32,
    predictor_state: &mut PredictorState<S>,
    wp_header: &WpHeader,
    grid: &mut MutableSubgrid<S>,
    table: &SimpleMaTable,
) -> Result<()> {
    let &SimpleMaTable {
        decision_prop,
        value_base,
        predictor,
        offset,
        multiplier,
        ref cluster_table,
    } = table;

    if offset == 0 && multiplier == 1 && decision_prop == 9 && predictor == Predictor::Gradient {
        return decode_gradient_table(
            bitstream,
            decoder,
            dist_multiplier,
            grid,
            value_base,
            cluster_table,
        );
    }

    decode_simple_table_slow(
        bitstream,
        decoder,
        dist_multiplier,
        predictor_state,
        wp_header,
        grid,
        table,
    )
}

#[inline(always)]
fn cluster_from_table(sample: i32, value_base: i32, cluster_table: &[u8]) -> u8 {
    let index = (sample - value_base).clamp(0, cluster_table.len() as i32 - 1);
    cluster_table[index as usize]
}

fn decode_gradient_table<S: Sample>(
    bitstream: &mut Bitstream,
    decoder: &mut Decoder,
    dist_multiplier: u32,
    grid: &mut MutableSubgrid<S>,
    value_base: i32,
    cluster_table: &[u8],
) -> Result<()> {
    tracing::trace!("Gradient-only lookup table");

    let width = grid.width();
    let height = grid.height();

    {
        let mut w = S::default();
        let out_row = grid.get_row_mut(0);
        for out in out_row[..width].iter_mut() {
            let cluster = cluster_from_table(w.to_i32(), value_base, cluster_table);
            let token = decoder.read_varint_with_multiplier_clustered(
                bitstream,
                cluster,
                dist_multiplier,
            )?;
            w = S::unpack_signed_u32(token).add(w);
            *out = w;
        }
    }

    for y in 1..height {
        let (u, mut d) = grid.split_vertical(y);
        let prev_row = u.get_row(y - 1);
        let out_row = d.get_row_mut(0);

        let cluster = cluster_from_table(prev_row[0].to_i32(), value_base, cluster_table);
        let token =
            decoder.read_varint_with_multiplier_clustered(bitstream, cluster, dist_multiplier)?;
        let mut w = S::unpack_signed_u32(token).add(prev_row[0]);
        out_row[0] = w;

        for (window, out) in prev_row.windows(2).zip(&mut out_row[1..]) {
            let nw = window[0];
            let n = window[1];
            let prop = n
                .to_i32()
                .wrapping_add(w.to_i32())
                .wrapping_sub(nw.to_i32());
            let pred = S::grad_clamped(n, w, nw);

            let cluster = cluster_from_table(prop, value_base, cluster_table);
            let token = decoder.read_varint_with_multiplier_clustered(
                bitstream,
                cluster,
                dist_multiplier,
            )?;
            let value = S::unpack_signed_u32(token).add(pred);
            *out = value;
            w = value;
        }
    }

    Ok(())
}

#[inline(always)]
fn decode_table_one<S: Sample, const EDGE: bool>(
    bitstream: &mut Bitstream,
    decoder: &mut Decoder,
    dist_multiplier: u32,
    table: &SimpleMaTable,
    properties: &Properties<S>,
) -> Result<S> {
    let prop_value = properties.get(table.decision_prop as usize);

    let cluster = cluster_from_table(prop_value, table.value_base, &table.cluster_table);

    let diff = S::unpack_signed_u32(decoder.read_varint_with_multiplier_clustered(
        bitstream,
        cluster,
        dist_multiplier,
    )?);
    let diff = diff.wrapping_muladd_i32(table.multiplier as i32, table.offset);
    let predictor = table.predictor;
    let sample_prediction = predictor.predict::<_, EDGE>(properties);
    Ok(diff.add(S::from_i32(sample_prediction)))
}

#[inline(never)]
fn decode_simple_table_slow<S: Sample>(
    bitstream: &mut Bitstream,
    decoder: &mut Decoder,
    dist_multiplier: u32,
    predictor_state: &mut PredictorState<S>,
    wp_header: &WpHeader,
    grid: &mut MutableSubgrid<S>,
    table: &SimpleMaTable,
) -> Result<()> {
    tracing::trace!("Slow lookup table");

    let need_wp_header = table.decision_prop == 15 || table.predictor == Predictor::SelfCorrecting;
    let wp_header = need_wp_header.then_some(wp_header);
    predictor_state.reset(grid.width() as u32, &[], wp_header);

    let height = grid.height();
    for y in 0..2usize.min(height) {
        let row = grid.get_row_mut(y);

        for out in row.iter_mut() {
            let properties = predictor_state.properties::<true>();
            let true_value = decode_table_one::<_, true>(
                bitstream,
                decoder,
                dist_multiplier,
                table,
                &properties,
            )?;
            *out = true_value;
            properties.record(true_value.to_i32());
        }
    }

    for y in 2..height {
        let row = grid.get_row_mut(y);
        let (row_left, row_middle, row_right) = if row.len() <= 4 {
            (row, [].as_mut(), [].as_mut())
        } else {
            let (l, m) = row.split_at_mut(2);
            let (m, r) = m.split_at_mut(m.len() - 2);
            (l, m, r)
        };

        for out in row_left {
            let properties = predictor_state.properties::<true>();
            let true_value = decode_table_one::<_, true>(
                bitstream,
                decoder,
                dist_multiplier,
                table,
                &properties,
            )?;
            *out = true_value;
            properties.record(true_value.to_i32());
        }
        for out in row_middle {
            let properties = predictor_state.properties::<false>();
            let true_value = decode_table_one::<_, false>(
                bitstream,
                decoder,
                dist_multiplier,
                table,
                &properties,
            )?;
            *out = true_value;
            properties.record(true_value.to_i32());
        }
        for out in row_right {
            let properties = predictor_state.properties::<true>();
            let true_value = decode_table_one::<_, true>(
                bitstream,
                decoder,
                dist_multiplier,
                table,
                &properties,
            )?;
            *out = true_value;
            properties.record(true_value.to_i32());
        }
    }

    Ok(())
}

#[inline(never)]
fn decode_slow<S: Sample>(
    bitstream: &mut Bitstream,
    decoder: &mut Decoder,
    dist_multiplier: u32,
    ma_tree: &FlatMaTree,
    predictor: &mut PredictorState<S>,
    grid: &mut MutableSubgrid<S>,
) -> Result<()> {
    let height = grid.height();
    for y in 0..2usize.min(height) {
        let row = grid.get_row_mut(y);

        for out in row.iter_mut() {
            let properties = predictor.properties::<true>();
            let leaf = ma_tree.get_leaf(&properties);
            let true_value =
                decode_one::<_, true>(bitstream, decoder, dist_multiplier, leaf, &properties)?;
            *out = true_value;
            properties.record(true_value.to_i32());
        }
    }

    for y in 2..height {
        let row = grid.get_row_mut(y);
        let (row_left, row_middle, row_right) = if row.len() <= 4 {
            (row, [].as_mut(), [].as_mut())
        } else {
            let (l, m) = row.split_at_mut(2);
            let (m, r) = m.split_at_mut(m.len() - 2);
            (l, m, r)
        };

        for out in row_left {
            let properties = predictor.properties::<true>();
            let leaf = ma_tree.get_leaf(&properties);
            let true_value =
                decode_one::<_, true>(bitstream, decoder, dist_multiplier, leaf, &properties)?;
            *out = true_value;
            properties.record(true_value.to_i32());
        }
        for out in row_middle {
            let properties = predictor.properties::<false>();
            let leaf = ma_tree.get_leaf(&properties);
            let true_value =
                decode_one::<_, false>(bitstream, decoder, dist_multiplier, leaf, &properties)?;
            *out = true_value;
            properties.record(true_value.to_i32());
        }
        for out in row_right {
            let properties = predictor.properties::<true>();
            let leaf = ma_tree.get_leaf(&properties);
            let true_value =
                decode_one::<_, true>(bitstream, decoder, dist_multiplier, leaf, &properties)?;
            *out = true_value;
            properties.record(true_value.to_i32());
        }
    }

    Ok(())
}
