use std::io::Read;
use std::sync::Arc;

use jxl_bitstream::{unpack_signed, Bitstream, Bundle};
use jxl_coding::Decoder;

use super::predictor::Predictor;

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
                let node = MaTreeNode::Leaf {
                    ctx,
                    predictor,
                    offset,
                    multiplier,
                };
                ctx += 1;
                node
            };
            nodes.push(node);
        }

        let decoder = Decoder::parse(bitstream, ((nodes.len() + 1) / 2) as u32)?;
        Ok(Self {
            tree: Arc::new(MaTree { nodes }),
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

#[derive(Debug)]
struct MaTree {
    nodes: Vec<MaTreeNode>,
}

#[derive(Debug)]
enum MaTreeNode {
    Decision {
        property: usize,
        value: i32,
        left_idx: usize,
        right_idx: usize,
    },
    Leaf {
        ctx: u32,
        predictor: super::predictor::Predictor,
        offset: i32,
        multiplier: u32,
    },
}
