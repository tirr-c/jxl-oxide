use std::io::Read;
use std::sync::Arc;

use jxl_bitstream::{unpack_signed, Bitstream, Bundle};
use jxl_coding::Decoder;

use crate::Result;
use super::predictor::{Predictor, Properties};

#[derive(Debug, Clone)]
pub struct MaConfig {
    tree: Arc<MaTree>,
    decoder: Decoder,
}

impl MaConfig {
    pub fn make_context(&self) -> MaContext {
        self.clone().into()
    }
}

impl<Ctx> Bundle<Ctx> for MaConfig {
    type Error = crate::Error;

    fn parse<R: Read>(bitstream: &mut Bitstream<R>, _: Ctx) -> crate::Result<Self> {
        let mut tree_decoder = Decoder::parse(bitstream, 6)?;
        let mut ctx = 0u32;
        let mut nodes_left = 1usize;
        let mut nodes = Vec::new();
        while nodes_left > 0 {
            if nodes.len() >= (1 << 26) {
                return Err(crate::Error::InvalidMaTree);
            }

            nodes_left -= 1;
            let property = tree_decoder.read_varint(bitstream, 1)?;
            let node = if let Some(property) = property.checked_sub(1) {
                let value = unpack_signed(tree_decoder.read_varint(bitstream, 0)?);
                let node = MaTreeNode::Decision {
                    property: property as usize,
                    value,
                    left_idx: nodes.len() + nodes_left + 1,
                    right_idx: nodes.len() + nodes_left + 2,
                };
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
                let node = MaTreeNode::Leaf(MaTreeLeaf {
                    ctx,
                    predictor,
                    offset,
                    multiplier,
                });
                ctx += 1;
                node
            };
            nodes.push(node);
        }

        let decoder = Decoder::parse(bitstream, ((nodes.len() + 1) / 2) as u32)?;
        Ok(Self {
            tree: Arc::new(MaTree::new(nodes)),
            decoder,
        })
    }
}

#[derive(Debug)]
pub struct MaContext {
    tree: Arc<MaTree>,
    decoder: Decoder,
}

impl From<MaConfig> for MaContext {
    fn from(config: MaConfig) -> Self {
        Self {
            tree: config.tree,
            decoder: config.decoder,
        }
    }
}

impl MaContext {
    pub fn need_self_correcting(&self) -> bool {
        self.tree.need_self_correcting
    }

    pub fn decode_sample<R: Read>(
        &mut self,
        bitstream: &mut Bitstream<R>,
        properties: &Properties,
        dist_multiplier: u32,
    ) -> Result<(i32, super::predictor::Predictor)> {
        let leaf = self.tree.get_leaf(properties)?;
        let diff = self.decoder.read_varint_with_multiplier(bitstream, leaf.ctx, dist_multiplier)?;
        let diff = unpack_signed(diff) * leaf.multiplier as i32 + leaf.offset;
        Ok((diff, leaf.predictor))
    }
}

#[derive(Debug)]
struct MaTree {
    nodes: Vec<MaTreeNode>,
    need_self_correcting: bool,
}

#[derive(Debug)]
enum MaTreeNode {
    Decision {
        property: usize,
        value: i32,
        left_idx: usize,
        right_idx: usize,
    },
    Leaf(MaTreeLeaf),
}

#[derive(Debug)]
struct MaTreeLeaf {
    ctx: u32,
    predictor: super::predictor::Predictor,
    offset: i32,
    multiplier: u32,
}

impl MaTree {
    fn new(nodes: Vec<MaTreeNode>) -> Self {
        let need_self_correcting = nodes.iter().any(|node| match *node {
            MaTreeNode::Decision { property, .. } => property == 15,
            MaTreeNode::Leaf(MaTreeLeaf { predictor, .. }) => predictor == Predictor::SelfCorrecting,
        });

        Self { nodes, need_self_correcting }
    }

    fn get_leaf(&self, properties: &Properties) -> Result<&MaTreeLeaf> {
        let mut current_node = &self.nodes[0];
        loop {
            match current_node {
                &MaTreeNode::Decision { property, value, left_idx, right_idx } => {
                    let prop_value = properties.get(property)?;
                    let next_node = if prop_value > value { left_idx } else { right_idx };
                    current_node = &self.nodes[next_node];
                },
                MaTreeNode::Leaf(leaf) => return Ok(leaf),
            }
        }
    }
}
