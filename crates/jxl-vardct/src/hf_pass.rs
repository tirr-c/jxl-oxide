use jxl_bitstream::{read_bits, Bitstream, Bundle};
use jxl_coding::Decoder;

#[derive(Debug, Copy, Clone)]
pub struct HfPassParams<'a> {
    hf_block_ctx: &'a crate::HfBlockContext,
    num_hf_presets: u32,
}

impl<'a> HfPassParams<'a> {
    pub fn new(hf_block_ctx: &'a crate::HfBlockContext, num_hf_presets: u32) -> Self {
        Self { hf_block_ctx, num_hf_presets }
    }
}

#[derive(Debug)]
pub struct HfPass {
    order: [[Vec<(u8, u8)>; 3]; 13],
    hf_dist: Decoder,
}

impl Bundle<HfPassParams<'_>> for HfPass {
    type Error = crate::Error;

    fn parse<R: std::io::Read>(bitstream: &mut Bitstream<R>, params: HfPassParams<'_>) -> crate::Result<Self> {
        let HfPassParams { hf_block_ctx, num_hf_presets } = params;
        let mut used_orders = read_bits!(bitstream, U32(0x5F, 0x13, 0x00, u(13)))?;
        let mut decoder = (used_orders != 0)
            .then(|| Decoder::parse(bitstream, 8))
            .transpose()?;

        let mut order: [_; 13] = std::array::from_fn(|_| [Vec::new(), Vec::new(), Vec::new()]);
        for (order, natural_order) in order.iter_mut().zip(NATURAL_ORDER) {
            if used_orders & 1 != 0 {
                let Some(decoder) = &mut decoder else { unreachable!() };
                let size = natural_order.len() as u32;
                let skip = size / 64;
                for order in order {
                    let permutation = jxl_coding::read_permutation(bitstream, decoder, size, skip)?;
                    for (val, perm) in order.iter_mut().zip(permutation) {
                        *val = natural_order[perm];
                    }
                }
            } else {
                for order in order {
                    *order = natural_order.to_vec();
                }
            }

            used_orders >>= 1;
        }

        let hf_dist = Decoder::parse(
            bitstream,
            495 * num_hf_presets * hf_block_ctx.num_block_clusters,
        )?;

        Ok(Self {
            order,
            hf_dist,
        })
    }
}

const BLOCK_SIZES: [(usize, usize); 13] = [
    (8, 8),
    (8, 8),
    (16, 16),
    (32, 32),
    (16, 8),
    (32, 8),
    (32, 16),
    (64, 64),
    (64, 32),
    (128, 128),
    (128, 64),
    (256, 256),
    (256, 128),
];
const NATURAL_ORDER: [&[(u8, u8)]; 13] = [
    &const_compute_natural_order::<{BLOCK_SIZES[0].0 * BLOCK_SIZES[0].1}>(BLOCK_SIZES[0]),
    &const_compute_natural_order::<{BLOCK_SIZES[1].0 * BLOCK_SIZES[1].1}>(BLOCK_SIZES[1]),
    &const_compute_natural_order::<{BLOCK_SIZES[2].0 * BLOCK_SIZES[2].1}>(BLOCK_SIZES[2]),
    &const_compute_natural_order::<{BLOCK_SIZES[3].0 * BLOCK_SIZES[3].1}>(BLOCK_SIZES[3]),
    &const_compute_natural_order::<{BLOCK_SIZES[4].0 * BLOCK_SIZES[4].1}>(BLOCK_SIZES[4]),
    &const_compute_natural_order::<{BLOCK_SIZES[5].0 * BLOCK_SIZES[5].1}>(BLOCK_SIZES[5]),
    &const_compute_natural_order::<{BLOCK_SIZES[6].0 * BLOCK_SIZES[6].1}>(BLOCK_SIZES[6]),
    &const_compute_natural_order::<{BLOCK_SIZES[7].0 * BLOCK_SIZES[7].1}>(BLOCK_SIZES[7]),
    &const_compute_natural_order::<{BLOCK_SIZES[8].0 * BLOCK_SIZES[8].1}>(BLOCK_SIZES[8]),
    &const_compute_natural_order::<{BLOCK_SIZES[9].0 * BLOCK_SIZES[9].1}>(BLOCK_SIZES[9]),
    &const_compute_natural_order::<{BLOCK_SIZES[10].0 * BLOCK_SIZES[10].1}>(BLOCK_SIZES[10]),
    &const_compute_natural_order::<{BLOCK_SIZES[11].0 * BLOCK_SIZES[11].1}>(BLOCK_SIZES[11]),
    &const_compute_natural_order::<{BLOCK_SIZES[12].0 * BLOCK_SIZES[12].1}>(BLOCK_SIZES[12]),
];

const fn const_compute_natural_order<const N: usize>((bw, bh): (usize, usize)) -> [(u8, u8); N] {
    let y_scale = bw / bh;

    let mut ret = [(0u8, 0u8); N];
    let mut idx = 0usize;
    let lbw = bw / 8;
    let lbh = bh / 8;

    while idx < lbw * lbh {
        let x = idx % lbw;
        let y = idx / lbw;
        ret[idx] = (x as u8, y as u8);
        idx += 1;
    }

    let mut dist = 0usize;
    let mut order = 0usize;
    while idx < bw * bh {
        let (x, y) = if dist % 2 == 0 {
            (order, dist - order)
        } else {
            (dist - order, order)
        };

        order += 1;
        if order > dist || order >= bw {
            dist += 1;
            order = if dist < bw {
                0
            } else {
                dist - bw + 1
            };
        }

        if x < lbw && y < lbw {
            continue;
        }
        if y % y_scale != 0 {
            continue;
        }
        ret[idx] = (x as u8, y as u8);
        idx += 1;
    }

    ret
}
