use std::collections::VecDeque;
use std::sync::Arc;

use jxl_bitstream::{unpack_signed, Bitstream};
use jxl_coding::Decoder;
use jxl_grid::{AllocHandle, AllocTracker};
use jxl_oxide_common::Bundle;

use super::predictor::{Predictor, Properties};
use crate::{sample::Sealed, Result, Sample};

/// Meta-adaptive tree configuration.
///
/// Meta-adaptive (MA) tree is a decision tree that controls how the sample is decoded in the given
/// context. The configuration consists of two components: the MA tree itself, and the distribution
/// information of an entropy decoder. These components are read from the bitstream.
#[derive(Debug, Clone)]
pub struct MaConfig {
    num_tree_nodes: usize,
    tree_depth: usize,
    tree: Arc<(MaTreeNode, Option<AllocHandle>)>,
    decoder: Decoder,
}

impl MaConfig {
    /// Returns the entropy decoder.
    ///
    /// The decoder should be cloned to be used for decoding.
    pub fn decoder(&self) -> &Decoder {
        &self.decoder
    }

    /// Creates a simplified MA tree with given channel index and stream index, which then can be
    /// used to decode samples.
    ///
    /// The method will evaluate the tree with the given information and prune branches which are
    /// always not taken.
    pub fn make_flat_tree(&self, channel: u32, stream_idx: u32, prev_channels: u32) -> FlatMaTree {
        let nodes = self.tree.0.flatten(channel, stream_idx, prev_channels);
        FlatMaTree::new(nodes)
    }
}

impl MaConfig {
    /// Returns the number of MA tree nodes.
    #[inline]
    pub fn num_tree_nodes(&self) -> usize {
        self.num_tree_nodes
    }

    /// Returns the maximum distance from root to any leaf node.
    #[inline]
    pub fn tree_depth(&self) -> usize {
        self.tree_depth
    }
}

/// Parameters for decoding [`MaConfig`].
#[derive(Debug, Copy, Clone)]
pub struct MaConfigParams<'a> {
    /// Allocation tracker.
    pub tracker: Option<&'a AllocTracker>,
    /// Maximum number of meta-adaptive tree nodes.
    pub node_limit: usize,
    pub depth_limit: usize,
}

impl Bundle<MaConfigParams<'_>> for MaConfig {
    type Error = crate::Error;

    fn parse(bitstream: &mut Bitstream, params: MaConfigParams) -> crate::Result<Self> {
        struct FoldingTreeLeaf {
            ctx: u32,
            predictor: super::predictor::Predictor,
            offset: i32,
            multiplier: u32,
        }

        enum FoldingTree {
            Decision(u32, i32),
            Leaf(FoldingTreeLeaf),
        }

        let MaConfigParams {
            tracker,
            node_limit,
            depth_limit,
        } = params;

        let mut tree_decoder = Decoder::parse(bitstream, 6)?;
        if is_infinite_tree_dist(&tree_decoder) {
            tracing::error!("Infinite MA tree");
            return Err(crate::Error::InvalidMaTree);
        }

        let mut ctx = 0u32;
        let mut nodes_left = 1usize;
        let mut tmp_alloc_handle = tracker
            .map(|tracker| tracker.alloc::<FoldingTree>(16))
            .transpose()?;
        let mut nodes = Vec::with_capacity(16);
        let mut max_depth = 1usize;

        tree_decoder.begin(bitstream)?;
        while nodes_left > 0 {
            if nodes.len() >= (1 << 26) {
                return Err(crate::Error::InvalidMaTree);
            }
            if nodes.len() > node_limit {
                tracing::error!(node_limit, "MA tree limit exceeded");
                return Err(
                    jxl_bitstream::Error::ProfileConformance("MA tree limit exceeded").into(),
                );
            }

            if nodes.len() == nodes.capacity() && tmp_alloc_handle.is_some() {
                let tracker = tracker.unwrap();
                let current_len = nodes.len();
                if current_len <= 16 {
                    drop(tmp_alloc_handle);
                    tmp_alloc_handle = Some(tracker.alloc::<FoldingTree>(256)?);
                    nodes.reserve(256 - current_len);
                } else if current_len <= 256 {
                    drop(tmp_alloc_handle);
                    tmp_alloc_handle = Some(tracker.alloc::<FoldingTree>(1024)?);
                    nodes.reserve(1024 - current_len);
                } else {
                    drop(tmp_alloc_handle);
                    tmp_alloc_handle = Some(tracker.alloc::<FoldingTree>(current_len * 2)?);
                    nodes.reserve(current_len);
                }
            }

            nodes_left -= 1;
            let property = tree_decoder.read_varint(bitstream, 1)?;
            let node = if let Some(property) = property.checked_sub(1) {
                let value = unpack_signed(tree_decoder.read_varint(bitstream, 0)?);
                let node = FoldingTree::Decision(property, value);
                nodes_left += 2;
                node
            } else {
                let predictor = tree_decoder.read_varint(bitstream, 2)?;
                let predictor = Predictor::try_from(predictor)?;
                let offset = unpack_signed(tree_decoder.read_varint(bitstream, 3)?);
                let mul_log = tree_decoder.read_varint(bitstream, 4)?;
                if mul_log > 30 {
                    return Err(crate::Error::InvalidMaTree);
                }
                let mul_bits = tree_decoder.read_varint(bitstream, 5)?;
                if mul_bits > (1 << (31 - mul_log)) - 2 {
                    return Err(crate::Error::InvalidMaTree);
                }
                let multiplier = (mul_bits + 1) << mul_log;
                let node = FoldingTree::Leaf(FoldingTreeLeaf {
                    ctx,
                    predictor,
                    offset,
                    multiplier,
                });
                ctx += 1;
                node
            };
            nodes.push(node);
            max_depth = max_depth.max(nodes_left);
        }
        tree_decoder.finalize()?;
        let num_tree_nodes = nodes.len();
        let decoder = Decoder::parse(bitstream, ctx)?;
        let cluster_map = decoder.cluster_map();

        let tree_alloc_handle = tracker
            .map(|tracker| tracker.alloc::<FoldingTree>(nodes.len()))
            .transpose()?;
        let mut tmp = VecDeque::<(_, usize)>::with_capacity(max_depth);
        for node in nodes.into_iter().rev() {
            match node {
                FoldingTree::Decision(property, value) => {
                    let (right, dr) = tmp.pop_front().unwrap();
                    let (left, dl) = tmp.pop_front().unwrap();
                    let node = Box::new(MaTreeNode::Decision {
                        property,
                        value,
                        left,
                        right,
                    });
                    let depth = dr.max(dl) + 1;
                    if depth > depth_limit {
                        tracing::error!(depth_limit, "Decoded MA tree is too deep");
                        return Err(jxl_bitstream::Error::ProfileConformance(
                            "decoded MA tree is too deep",
                        )
                        .into());
                    }

                    tmp.push_back((node, depth));
                }
                FoldingTree::Leaf(FoldingTreeLeaf {
                    ctx,
                    predictor,
                    offset,
                    multiplier,
                }) => {
                    let cluster = cluster_map[ctx as usize];
                    let leaf = MaTreeLeafClustered {
                        cluster,
                        predictor,
                        offset,
                        multiplier,
                    };
                    let node = Box::new(MaTreeNode::Leaf(leaf));
                    tmp.push_back((node, 0));
                }
            }
        }
        assert_eq!(tmp.len(), 1);
        let (tree, tree_depth) = tmp.pop_front().unwrap();
        let tree = *tree;

        Ok(Self {
            num_tree_nodes,
            tree_depth,
            tree: Arc::new((tree, tree_alloc_handle)),
            decoder,
        })
    }
}

fn is_infinite_tree_dist(decoder: &Decoder) -> bool {
    let cluster_map = decoder.cluster_map();

    // Distribution #1 decides whether it's decision node or leaf node; if it reads 0 it's a leaf
    // node. Therefore, the tree is infinitely large if the dist always reads token other than 0.
    let cluster = cluster_map[1];
    let Some(token) = decoder.single_token(cluster) else {
        return false;
    };
    token != 0
}

/// A "flat" meta-adaptive tree, constructed with [`MaConfig::make_flat_tree`].
#[derive(Debug)]
pub struct FlatMaTree {
    nodes: Vec<FlatMaTreeNode>,
    need_self_correcting: bool,
    max_prev_channel_depth: usize,
}

#[derive(Debug)]
enum FlatMaTreeNode {
    FusedDecision {
        prop_level0: u32,
        value_level0: i32,
        props_level1: (u32, u32),
        values_level1: (i32, i32),
        index_base: u32,
    },
    Table {
        prop: u32,
        value_base: i32,
        indices: Box<[u32]>,
    },
    Leaf(MaTreeLeafClustered),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MaTreeLeafClustered {
    pub(crate) cluster: u8,
    pub(crate) predictor: super::predictor::Predictor,
    pub(crate) offset: i32,
    pub(crate) multiplier: u32,
}

impl FlatMaTree {
    fn new(nodes: Vec<FlatMaTreeNode>) -> Self {
        let need_self_correcting = nodes.iter().any(|node| match *node {
            FlatMaTreeNode::FusedDecision {
                prop_level0: p,
                props_level1: (pl, pr),
                ..
            } => p == 15 || pl == 15 || pr == 15,
            FlatMaTreeNode::Table { prop, .. } => prop == 15,
            FlatMaTreeNode::Leaf(MaTreeLeafClustered { predictor, .. }) => {
                predictor == Predictor::SelfCorrecting
            }
        });

        let mut max_prev_channel_depth = 0usize;
        for node in &nodes {
            if let FlatMaTreeNode::FusedDecision {
                prop_level0: p,
                props_level1: (pl, pr),
                ..
            } = *node
            {
                if let Some(p) = p.checked_sub(16) {
                    max_prev_channel_depth = max_prev_channel_depth.max((p as usize / 4) + 1);
                }
                if let Some(p) = pl.checked_sub(16) {
                    max_prev_channel_depth = max_prev_channel_depth.max((p as usize / 4) + 1);
                }
                if let Some(p) = pr.checked_sub(16) {
                    max_prev_channel_depth = max_prev_channel_depth.max((p as usize / 4) + 1);
                }
            } else if let FlatMaTreeNode::Table { prop, .. } = *node {
                if let Some(p) = prop.checked_sub(16) {
                    max_prev_channel_depth = max_prev_channel_depth.max((p as usize / 4) + 1);
                }
            }
        }

        Self {
            nodes,
            need_self_correcting,
            max_prev_channel_depth,
        }
    }

    pub(crate) fn get_leaf<S: Sample>(&self, properties: &Properties<S>) -> &MaTreeLeafClustered {
        let mut current_node = &self.nodes[0];
        loop {
            match current_node {
                &FlatMaTreeNode::FusedDecision {
                    prop_level0: p,
                    value_level0: v,
                    props_level1: (pl, pr),
                    values_level1: (vl, vr),
                    index_base,
                } => {
                    let p0v = properties.get(p as usize);
                    let plv = properties.get(pl as usize);
                    let prv = properties.get(pr as usize);
                    let high_bit = p0v <= v;
                    let l = (plv <= vl) as u32;
                    let r = 2 | (prv <= vr) as u32;
                    let next_node = index_base + if high_bit { r } else { l };
                    current_node = &self.nodes[next_node as usize];
                }
                &FlatMaTreeNode::Table {
                    prop,
                    value_base,
                    ref indices,
                } => {
                    let v = properties.get(prop as usize);
                    let idx = v
                        .saturating_sub(value_base)
                        .clamp(0, indices.len() as i32 - 1) as usize;
                    let next_node = indices[idx];
                    current_node = &self.nodes[next_node as usize];
                }
                FlatMaTreeNode::Leaf(leaf) => return leaf,
            }
        }
    }
}

impl FlatMaTree {
    /// Returns whether self-correcting predictor should be initialized.
    ///
    /// The return value of this method can be used to optimize the decoding process, since
    /// self-correcting predictors are computationally heavy.
    #[inline]
    pub fn need_self_correcting(&self) -> bool {
        self.need_self_correcting
    }

    /// Returns the number of previously decoded channels needed in order to traverse the MA tree.
    #[inline]
    pub fn max_prev_channel_depth(&self) -> usize {
        self.max_prev_channel_depth
    }

    /// Decode a sample with the given state.
    pub fn decode_sample<S: Sample>(
        &self,
        bitstream: &mut Bitstream,
        decoder: &mut Decoder,
        properties: &Properties<S>,
        dist_multiplier: u32,
    ) -> Result<(i32, super::predictor::Predictor)> {
        let leaf = self.get_leaf(properties);
        let diff = decoder.read_varint_with_multiplier_clustered(
            bitstream,
            leaf.cluster,
            dist_multiplier,
        )?;
        let diff = unpack_signed(diff).wrapping_muladd_i32(leaf.multiplier as i32, leaf.offset);
        Ok((diff, leaf.predictor))
    }

    #[inline]
    pub(crate) fn single_node(&self) -> Option<&MaTreeLeafClustered> {
        match self.nodes.first() {
            Some(FlatMaTreeNode::Leaf(node)) => Some(node),
            _ => None,
        }
    }

    pub(crate) fn simple_table(&self) -> Option<SimpleMaTable> {
        let Some(&FlatMaTreeNode::Table {
            prop: decision_prop,
            value_base,
            ref indices,
        }) = self.nodes.first()
        else {
            return None;
        };

        let mut state: Option<(Predictor, i32, u32)> = None;
        let mut cluster_table = Vec::with_capacity(indices.len());
        for &index in &**indices {
            let node = &self.nodes[index as usize];
            let FlatMaTreeNode::Leaf(leaf) = node else {
                return None;
            };

            let leaf_props = (leaf.predictor, leaf.offset, leaf.multiplier);
            let &mut state = state.get_or_insert(leaf_props);
            if leaf_props != state {
                return None;
            }

            cluster_table.push(leaf.cluster);
        }

        let (predictor, offset, multiplier) = state.unwrap();
        Some(SimpleMaTable {
            decision_prop,
            value_base,
            predictor,
            offset,
            multiplier,
            cluster_table: cluster_table.into_boxed_slice(),
        })
    }
}

#[derive(Debug)]
pub(crate) struct SimpleMaTable {
    pub(crate) decision_prop: u32,
    pub(crate) value_base: i32,
    pub(crate) predictor: Predictor,
    pub(crate) offset: i32,
    pub(crate) multiplier: u32,
    pub(crate) cluster_table: Box<[u8]>,
}

#[derive(Debug)]
enum MaTreeNode {
    Decision {
        property: u32,
        value: i32,
        left: Box<MaTreeNode>,
        right: Box<MaTreeNode>,
    },
    Leaf(MaTreeLeafClustered),
}

impl MaTreeNode {
    fn next_decision_node(&self, channel: u32, stream_idx: u32, prev_channels: u32) -> &MaTreeNode {
        match *self {
            MaTreeNode::Decision {
                property: property @ (0 | 1),
                value,
                ref left,
                ref right,
            } => {
                let target = if property == 0 { channel } else { stream_idx };
                let node = if target as i32 > value { left } else { right };
                node.next_decision_node(channel, stream_idx, prev_channels)
            }
            ref node @ MaTreeNode::Decision {
                property,
                value,
                ref left,
                ref right,
            } if property >= 16 => {
                let prev_channel_idx = (property - 16) / 4;
                if prev_channel_idx >= prev_channels {
                    let node = if value < 0 { left } else { right };
                    node.next_decision_node(channel, stream_idx, prev_channels)
                } else {
                    node
                }
            }
            ref node => node,
        }
    }

    fn try_compile_to_table(
        &self,
        channel: u32,
        stream_idx: u32,
        prev_channels: u32,
        next_index_base: u32,
    ) -> Option<(FlatMaTreeNode, Vec<&MaTreeNode>)> {
        let &MaTreeNode::Decision {
            property,
            value,
            ref left,
            ref right,
        } = self
        else {
            return None;
        };

        let mut lower_bound = value;
        let mut upper_bound = value;
        let mut stack = vec![
            (&**left, (value + 1)..=i32::MAX),
            (&**right, i32::MIN..=value),
        ];
        let mut range_nodes = Vec::new();
        while let Some((node, range)) = stack.pop() {
            let node = node.next_decision_node(channel, stream_idx, prev_channels);
            let (value, left, right) = match node {
                &MaTreeNode::Decision {
                    property: target_property,
                    value,
                    ref left,
                    ref right,
                } if target_property == property => (value, left, right),
                _ => {
                    range_nodes.push((node, *range.end()));
                    continue;
                }
            };
            let new_lower_bound = lower_bound.min(value);
            let new_upper_bound = upper_bound.max(value);
            if new_upper_bound.abs_diff(new_lower_bound) > 1024 - 2 {
                range_nodes.push((node, *range.end()));
                continue;
            }
            lower_bound = new_lower_bound;
            upper_bound = new_upper_bound;

            let left_range = (value + 1)..=(*range.end());
            let right_range = (*range.start())..=value;
            if !left_range.is_empty() {
                stack.push((&**left, left_range));
            }
            if !right_range.is_empty() {
                stack.push((&**right, right_range));
            }
        }
        if range_nodes.len() < 4 {
            return None;
        }

        range_nodes.sort_unstable_by_key(|(_, range_end)| *range_end);

        let index_count = upper_bound.abs_diff(lower_bound) as usize + 2;
        let mut indices = vec![0u32; index_count];
        let mut nodes = Vec::with_capacity(range_nodes.len());

        let mut range_start = lower_bound - 1;
        let mut next_index = 0usize;
        for (idx, (node, range_end)) in range_nodes.into_iter().enumerate() {
            if range_end == i32::MAX {
                *indices.last_mut().unwrap() = next_index_base + idx as u32;
                nodes.push(node);
                break;
            }
            let len = range_end.abs_diff(range_start) as usize;
            let end_index = next_index + len;
            indices[next_index..end_index].fill(next_index_base + idx as u32);
            nodes.push(node);
            next_index = end_index;
            range_start = range_end;
        }

        let node = FlatMaTreeNode::Table {
            prop: property,
            value_base: lower_bound,
            indices: indices.into_boxed_slice(),
        };
        Some((node, nodes))
    }

    fn flatten(&self, channel: u32, stream_idx: u32, prev_channels: u32) -> Vec<FlatMaTreeNode> {
        let target = self.next_decision_node(channel, stream_idx, prev_channels);
        let mut q = std::collections::VecDeque::new();
        q.push_back(target);

        let mut out = Vec::new();
        let mut next_base = 1u32;
        while let Some(target) = q.pop_front() {
            let target = target.next_decision_node(channel, stream_idx, prev_channels);
            if let Some((out_node, nodes)) =
                target.try_compile_to_table(channel, stream_idx, prev_channels, next_base)
            {
                let len = nodes.len() as u32;
                out.push(out_node);
                q.extend(nodes);
                next_base += len;
                continue;
            }

            match *target {
                MaTreeNode::Decision {
                    property,
                    value,
                    ref left,
                    ref right,
                } => {
                    let left = left.next_decision_node(channel, stream_idx, prev_channels);
                    let (lp, lv, ll, lr) = match left {
                        &MaTreeNode::Decision {
                            property,
                            value,
                            ref left,
                            ref right,
                        } => (property, value, &**left, &**right),
                        node => (0, 0, node, node),
                    };
                    let right = right.next_decision_node(channel, stream_idx, prev_channels);
                    let (rp, rv, rl, rr) = match right {
                        &MaTreeNode::Decision {
                            property,
                            value,
                            ref left,
                            ref right,
                        } => (property, value, &**left, &**right),
                        node => (0, 0, node, node),
                    };
                    out.push(FlatMaTreeNode::FusedDecision {
                        prop_level0: property,
                        value_level0: value,
                        props_level1: (lp, rp),
                        values_level1: (lv, rv),
                        index_base: next_base,
                    });
                    q.push_back(ll);
                    q.push_back(lr);
                    q.push_back(rl);
                    q.push_back(rr);
                    next_base += 4;
                }
                MaTreeNode::Leaf(ref leaf) => {
                    out.push(FlatMaTreeNode::Leaf(leaf.clone()));
                }
            }
        }

        out
    }
}
