use jxl_bitstream::{read_bits, Bitstream, Bundle};
use jxl_coding::Decoder;

/// Parameters for decoding `HfPass`.
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

/// HF coefficient decoder configuration.
///
/// Includes distribution for the entropy decoder and the order of HF coefficients. This struct is
/// passed as a parameter when decoding [`HfCoeff`][crate::HfCoeff].
#[derive(Debug)]
pub struct HfPass {
    permutation: [[Vec<usize>; 3]; 13],
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

        let mut permutation: [_; 13] = std::array::from_fn(|_| [Vec::new(), Vec::new(), Vec::new()]);
        if let Some(decoder) = &mut decoder {
            for (permutation, (bw, bh)) in permutation.iter_mut().zip(BLOCK_SIZES) {
                if used_orders & 1 != 0 {
                    let size = (bw * bh) as u32;
                    let skip = size / 64;
                    for permutation in permutation {
                        *permutation = jxl_coding::read_permutation(bitstream, decoder, size, skip)?;
                    }
                }

                used_orders >>= 1;
            }
            decoder.finalize()?;
        }

        let hf_dist = Decoder::parse(
            bitstream,
            495 * num_hf_presets * hf_block_ctx.num_block_clusters,
        )?;

        Ok(Self {
            permutation,
            hf_dist,
        })
    }
}

impl HfPass {
    pub(crate) fn clone_decoder(&self) -> Decoder {
        self.hf_dist.clone()
    }

    pub(crate) fn order(&self, order_id: usize, channel: usize) -> impl Iterator<Item = (u8, u8)> + '_ {
        struct OrderIter<'a> {
            permutation: &'a [usize],
            natural_order: &'static [(u8, u8)],
            idx: usize,
        }

        impl Iterator for OrderIter<'_> {
            type Item = (u8, u8);

            fn next(&mut self) -> Option<(u8, u8)> {
                let idx = if self.permutation.is_empty() {
                    self.idx
                } else {
                    *self.permutation.get(self.idx)?
                };
                let ret = *self.natural_order.get(idx)?;
                self.idx += 1;
                Some(ret)
            }
        }

        let permutation = &self.permutation[order_id][channel];
        let natural_order = natural_order_lazy(order_id);
        OrderIter { permutation, natural_order, idx: 0 }
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
const NATURAL_ORDER: [&[(u8, u8)]; 9] = [
    &const_compute_natural_order::<{BLOCK_SIZES[0].0 * BLOCK_SIZES[0].1}>(BLOCK_SIZES[0]),
    &const_compute_natural_order::<{BLOCK_SIZES[1].0 * BLOCK_SIZES[1].1}>(BLOCK_SIZES[1]),
    &const_compute_natural_order::<{BLOCK_SIZES[2].0 * BLOCK_SIZES[2].1}>(BLOCK_SIZES[2]),
    &const_compute_natural_order::<{BLOCK_SIZES[3].0 * BLOCK_SIZES[3].1}>(BLOCK_SIZES[3]),
    &const_compute_natural_order::<{BLOCK_SIZES[4].0 * BLOCK_SIZES[4].1}>(BLOCK_SIZES[4]),
    &const_compute_natural_order::<{BLOCK_SIZES[5].0 * BLOCK_SIZES[5].1}>(BLOCK_SIZES[5]),
    &const_compute_natural_order::<{BLOCK_SIZES[6].0 * BLOCK_SIZES[6].1}>(BLOCK_SIZES[6]),
    &const_compute_natural_order::<{BLOCK_SIZES[7].0 * BLOCK_SIZES[7].1}>(BLOCK_SIZES[7]),
    &const_compute_natural_order::<{BLOCK_SIZES[8].0 * BLOCK_SIZES[8].1}>(BLOCK_SIZES[8]),
];

fn natural_order_lazy(idx: usize) -> &'static [(u8, u8)] {
    if idx >= 13 {
        panic!("Order ID out of bounds");
    }
    let block_size = BLOCK_SIZES[idx];
    let Some(idx) = idx.checked_sub(NATURAL_ORDER.len()) else {
        return NATURAL_ORDER[idx];
    };

    static INITIALIZER: [std::sync::Once; 4] = [
        std::sync::Once::new(),
        std::sync::Once::new(),
        std::sync::Once::new(),
        std::sync::Once::new(),
    ];
    static mut LARGE_NATURAL_ORDER: [Vec<(u8, u8)>; 4] = [
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    ];

    // TODO: Replace this with `OnceLock` when it is available in stable.
    INITIALIZER[idx].call_once(|| {
        // SAFETY: this is the only thread accessing LARGE_NATURAL_ORDER[idx],
        // as we're in call_once
        let natural_order = unsafe { &mut LARGE_NATURAL_ORDER[idx] };
        natural_order.resize(block_size.0 * block_size.1, (0, 0));
        fill_natural_order(block_size, natural_order);
    });
    // SAFETY: none of the threads will have mutable access to LARGE_NATURAL_ORDER[idx],
    // as we used call_once
    unsafe { &LARGE_NATURAL_ORDER[idx] }
}

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

    let mut dist = 1usize;
    while dist < 2 * bw {
        let margin = dist.saturating_sub(bw);
        let mut order = margin;
        while order < dist - margin {
            let (x, y) = if dist % 2 == 1 {
                (order, dist - 1 - order)
            } else {
                (dist - 1 - order, order)
            };
            order += 1;

            if x < lbw && y < lbw {
                continue;
            }
            if y % y_scale != 0 {
                continue;
            }
            ret[idx] = (x as u8, (y / y_scale) as u8);
            idx += 1;
        }
        dist += 1;
    }

    ret
}

fn fill_natural_order((bw, bh): (usize, usize), output: &mut [(u8, u8)]) {
    let y_scale = bw / bh;

    let mut idx = 0usize;
    let lbw = bw / 8;
    let lbh = bh / 8;

    while idx < lbw * lbh {
        let x = idx % lbw;
        let y = idx / lbw;
        output[idx] = (x as u8, y as u8);
        idx += 1;
    }

    for dist in 1..(2 * bw) {
        let margin = dist.saturating_sub(bw);
        for order in margin..(dist - margin) {
            let (x, y) = if dist % 2 == 1 {
                (order, dist - 1 - order)
            } else {
                (dist - 1 - order, order)
            };

            if x < lbw && y < lbw {
                continue;
            }
            if y % y_scale != 0 {
                continue;
            }
            output[idx] = (x as u8, (y / y_scale) as u8);
            idx += 1;
        }
    }
}
