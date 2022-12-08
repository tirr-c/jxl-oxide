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
                let property = MaProperty::try_from(property)?;
                let value = unpack_signed(tree_decoder.read_varint(bitstream, 0)?);
                let node = MaTreeNode::Decision {
                    property,
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
        property: MaProperty,
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

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
#[repr(u8)]
enum MaProperty {
    ChannelIndex = 0,
    StreamIndex,
    Y,
    X,
    /// abs(N)
    AbsN,
    /// abs(W)
    AbsW,
    N,
    W,
    /// if x > 0, then W - (W + N - NW), else W
    WsProp9,
    /// W + N - NW
    WpNsNw,
    /// W - NW
    WsNw,
    /// NW - N
    NwsN,
    /// N - NE
    NsNe,
    /// N - NN
    NsNn,
    /// W - WW
    WsWw,
    MaxError,
}

impl TryFrom<u32> for MaProperty {
    type Error = jxl_bitstream::Error;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        use MaProperty::*;
        Ok(match value {
            0 => ChannelIndex,
            1 => StreamIndex,
            2 => Y,
            3 => X,
            4 => AbsN,
            5 => AbsW,
            6 => N,
            7 => W,
            8 => WsProp9,
            9 => WpNsNw,
            10 => WsNw,
            11 => NwsN,
            12 => NsNe,
            13 => NsNn,
            14 => WsWw,
            15 => MaxError,
            _ => return Err(jxl_bitstream::Error::InvalidEnum { name: "MaProperty", value }),
        })
    }
}
