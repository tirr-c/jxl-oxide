use std::io::Read;
use std::sync::Arc;

use jxl_bitstream::{Bitstream, read_bits};

mod ans;
mod error;
mod permutation;
mod prefix;

pub use error::Error;
pub type Result<T> = std::result::Result<T, Error>;

pub use permutation::read_permutation;

#[derive(Debug, Clone)]
pub struct Decoder {
    lz77: Lz77,
    inner: DecoderInner,
}

impl Decoder {
    const SPECIAL_DISTANCES: [[i8; 2]; 120] = [
        [0, 1], [1, 0],  [1, 1], [-1, 1], [0, 2], [2, 0], [1, 2], [-1, 2], [2, 1], [-2, 1], [2, 2],
        [-2, 2], [0, 3], [3, 0], [1, 3], [-1, 3], [3, 1], [-3, 1], [2, 3], [-2, 3], [3, 2],
        [-3, 2], [0, 4], [4, 0], [1, 4], [-1, 4], [4, 1], [-4, 1], [3, 3], [-3, 3], [2, 4],
        [-2, 4], [4, 2], [-4, 2], [0, 5], [3, 4],  [-3, 4], [4, 3], [-4, 3], [5, 0], [1, 5],
        [-1, 5], [5, 1], [-5, 1], [2, 5], [-2, 5], [5, 2], [-5, 2], [4, 4], [-4, 4], [3, 5],
        [-3, 5], [5, 3], [-5, 3], [0, 6], [6, 0], [1, 6], [-1, 6], [6, 1], [-6, 1], [2, 6],
        [-2, 6], [6, 2], [-6, 2], [4, 5], [-4, 5], [5, 4], [-5, 4], [3, 6], [-3, 6], [6, 3],
        [-6, 3], [0, 7], [7, 0],  [1, 7], [-1, 7], [5, 5], [-5, 5], [7, 1], [-7, 1], [4, 6],
        [-4, 6], [6, 4], [-6, 4], [2, 7], [-2, 7], [7, 2], [-7, 2], [3, 7], [-3, 7], [7, 3],
        [-7, 3], [5, 6], [-5, 6], [6, 5], [-6, 5], [8, 0], [4, 7],  [-4, 7], [7, 4], [-7, 4],
        [8, 1], [8, 2], [6, 6], [-6, 6], [8, 3], [5, 7], [-5, 7], [7, 5], [-7, 5], [8, 4], [6, 7],
        [-6, 7], [7, 6], [-7, 6], [8, 5], [7, 7], [-7, 7], [8, 6], [8, 7],
    ];

    pub fn parse<R: std::io::Read>(bitstream: &mut Bitstream<R>, num_dist: u32) -> Result<Self> {
        let lz77 = Lz77::parse(bitstream)?;
        let num_dist = if let Lz77::Disabled = &lz77 {
            num_dist
        } else {
            num_dist + 1
        };
        let inner = DecoderInner::parse(bitstream, num_dist)?;
        Ok(Self { lz77, inner })
    }

    fn parse_assume_no_lz77<R: std::io::Read>(bitstream: &mut Bitstream<R>, num_dist: u32) -> Result<Self> {
        let lz77_enabled = read_bits!(bitstream, Bool)?;
        if lz77_enabled {
            return Err(Error::Lz77NotAllowed);
        }
        let inner = DecoderInner::parse(bitstream, num_dist)?;
        Ok(Self { lz77: Lz77::Disabled, inner })
    }

    pub fn read_varint<R: std::io::Read>(&mut self, bitstream: &mut Bitstream<R>, ctx: u32) -> Result<u32> {
        self.read_varint_with_multiplier(bitstream, ctx, 0)
    }

    pub fn read_varint_with_multiplier<R: std::io::Read>(&mut self, bitstream: &mut Bitstream<R>, ctx: u32, dist_multiplier: u32) -> Result<u32> {
        let cluster = self.inner.clusters[ctx as usize];
        Ok(if let Lz77::Enabled { ref mut state, min_symbol, min_length } = self.lz77 {
            let min_symbol = min_symbol as u16;
            let r;
            if state.num_to_copy > 0 {
                r = state.window[(state.copy_pos & 0xfffff) as usize];
                state.copy_pos += 1;
                state.num_to_copy -= 1;
            } else {
                let token = self.inner.code.read_symbol(bitstream, cluster)?;
                if token >= min_symbol {
                    let lz_dist_cluster = self.inner.lz_dist_cluster();

                    state.num_to_copy = self.inner.read_uint(bitstream, &state.lz_len_conf, (token - min_symbol) as u32)? + min_length;
                    let token = self.inner.code.read_symbol(bitstream, lz_dist_cluster)?;
                    let distance = self.inner.read_uint(bitstream, &self.inner.configs[lz_dist_cluster as usize], token as u32)?;
                    let distance = if dist_multiplier == 0 {
                        distance + 1
                    } else if distance < 120 {
                        let [offset, dist] = Self::SPECIAL_DISTANCES[distance as usize];
                        let dist = offset as i32 + dist_multiplier as i32 * dist as i32;
                        dist.max(1) as u32
                    } else {
                        distance - 119
                    };

                    let distance = (1 << 20).min(distance).min(state.num_decoded);
                    state.copy_pos = state.num_decoded - distance;

                    r = state.window[(state.copy_pos & 0xfffff) as usize];
                    state.copy_pos += 1;
                    state.num_to_copy -= 1;
                } else {
                    r = self.inner.read_uint(bitstream, &self.inner.configs[cluster as usize], token as u32)?;
                }
            }
            state.window[(state.num_decoded & 0xfffff) as usize] = r;
            state.num_decoded += 1;
            r
        } else {
            let token = self.inner.code.read_symbol(bitstream, cluster)?;
            self.inner.read_uint(bitstream, &self.inner.configs[cluster as usize], token as u32)?
        })
    }

    pub fn begin<R: Read>(&mut self, bitstream: &mut Bitstream<R>) -> Result<()> {
        self.inner.code.begin(bitstream)
    }

    pub fn finalize(&self) -> Result<()> {
        self.inner.code.finalize()
    }
}

#[derive(Debug, Clone)]
enum Lz77 {
    Disabled,
    Enabled {
        min_symbol: u32,
        min_length: u32,
        state: Lz77State,
    },
}

impl Lz77 {
    fn parse<R: std::io::Read>(bitstream: &mut Bitstream<R>) -> Result<Self> {
        Ok(if read_bits!(bitstream, Bool)? { // enabled
            let min_symbol = read_bits!(bitstream, U32(224, 512, 4096, 8 + u(15)))?;
            let min_length = read_bits!(bitstream, U32(3, 4, 5 + u(2), 9 + u(8)))?;
            let lz_len_conf = IntegerConfig::parse(bitstream, 8)?;
            Self::Enabled {
                min_symbol,
                min_length,
                state: Lz77State::new(lz_len_conf),
            }
        } else {
            Self::Disabled
        })
    }
}

#[derive(Clone)]
struct Lz77State {
    lz_len_conf: IntegerConfig,
    window: Vec<u32>,
    num_to_copy: u32,
    copy_pos: u32,
    num_decoded: u32,
}

impl std::fmt::Debug for Lz77State {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Lz77State")
            .field("lz_len_conf", &self.lz_len_conf)
            .field("num_to_copy", &self.num_to_copy)
            .field("copy_pos", &self.copy_pos)
            .field("num_decoded", &self.num_decoded)
            .finish_non_exhaustive()
    }
}

impl Lz77State {
    const WINDOW_LEN: usize = 1 << 20;

    fn new(lz_len_conf: IntegerConfig) -> Self {
        Self {
            lz_len_conf,
            window: vec![0u32; Self::WINDOW_LEN],
            num_to_copy: 0,
            copy_pos: 0,
            num_decoded: 0,
        }
    }
}

#[derive(Debug, Clone)]
struct IntegerConfig {
    split_exponent: u32,
    split: u32,
    msb_in_token: u32,
    lsb_in_token: u32,
}

impl IntegerConfig {
    fn parse<R: std::io::Read>(bitstream: &mut Bitstream<R>, log_alphabet_size: u32) -> Result<Self> {
        let split_exponent_bits = add_log2_ceil(log_alphabet_size);
        let split_exponent = bitstream.read_bits(split_exponent_bits)?;
        let (msb_in_token, lsb_in_token) = if split_exponent != log_alphabet_size {
            let msb_bits = add_log2_ceil(split_exponent);
            let msb_in_token = bitstream.read_bits(msb_bits)?;
            let lsb_bits = add_log2_ceil(split_exponent - msb_in_token);
            let lsb_in_token = bitstream.read_bits(lsb_bits)?;
            (msb_in_token, lsb_in_token)
        } else {
            (0u32, 0u32)
        };
        Ok(Self {
            split_exponent,
            split: 1 << split_exponent,
            msb_in_token,
            lsb_in_token,
        })
    }
}

#[derive(Debug, Clone)]
struct DecoderInner {
    clusters: Vec<u8>, // num_dist, [0, num_clusters)
    configs: Vec<IntegerConfig>, // num_clusters
    code: Coder,
}

impl DecoderInner {
    fn parse<R: std::io::Read>(bitstream: &mut Bitstream<R>, num_dist: u32) -> Result<Self> {
        let (num_clusters, clusters) = read_clusters(bitstream, num_dist)?;
        let use_prefix_code = read_bits!(bitstream, Bool)?;
        let log_alphabet_size = if use_prefix_code {
            15
        } else {
            read_bits!(bitstream, 5 + u(2))?
        };
        let configs = (0..num_clusters)
            .map(|_| IntegerConfig::parse(bitstream, log_alphabet_size))
            .collect::<Result<Vec<_>>>()?;
        let code = if use_prefix_code {
            let counts = (0..num_clusters)
                .map(|_| -> Result<_> {
                    let count = if read_bits!(bitstream, Bool)? {
                        let n = bitstream.read_bits(4)?;
                        1 + (1 << n) + bitstream.read_bits(n)?
                    } else {
                        1
                    };
                    if count > 1 << 15 {
                        return Err(Error::InvalidPrefixHistogram);
                    }
                    Ok(count)
                })
                .collect::<Result<Vec<_>>>()?;
            let dist = counts
                .into_iter()
                .map(|count| prefix::Histogram::parse(bitstream, count))
                .collect::<Result<Vec<_>>>()?;
            Coder::PrefixCode(Arc::new(dist))
        } else {
            let dist = (0..num_clusters)
                .map(|_| ans::Histogram::parse(bitstream, log_alphabet_size))
                .collect::<Result<Vec<_>>>()?;
            Coder::Ans {
                dist: Arc::new(dist),
                state: 0,
                initial: true,
            }
        };
        Ok(Self {
            clusters,
            configs,
            code,
        })
    }

    fn read_uint<R: std::io::Read>(&self, bitstream: &mut Bitstream<R>, config: &IntegerConfig, token: u32) -> Result<u32> {
        let &IntegerConfig { split_exponent, split, msb_in_token, lsb_in_token, .. } = config;
        if token < split {
            return Ok(token);
        }

        let n = split_exponent - (msb_in_token + lsb_in_token) +
            ((token - split) >> (msb_in_token + lsb_in_token));
        let low_bits = token & ((1 << lsb_in_token) - 1);
        let token = token >> lsb_in_token;
        let token = token & ((1 << msb_in_token) - 1);
        let token = token | (1 << msb_in_token);
        Ok((((token << n) | bitstream.read_bits(n)?) << lsb_in_token) | low_bits)
    }

    fn lz_dist_cluster(&self) -> u8 {
        *self.clusters.last().unwrap()
    }
}

#[derive(Debug, Clone)]
enum Coder {
    PrefixCode(Arc<Vec<prefix::Histogram>>),
    Ans {
        dist: Arc<Vec<ans::Histogram>>,
        state: u32,
        initial: bool,
    },
}

impl Coder {
    fn read_symbol<R: Read>(&mut self, bitstream: &mut Bitstream<R>, cluster: u8) -> Result<u16> {
        match self {
            Self::PrefixCode(dist) => {
                let dist = &dist[cluster as usize];
                dist.read_symbol(bitstream)
            },
            Self::Ans { dist, state, initial } => {
                if *initial {
                    *state = bitstream.read_bits(32)?;
                    *initial = false;
                }
                let dist = &dist[cluster as usize];
                dist.read_symbol(bitstream, state)
            },
        }
    }

    fn begin<R: Read>(&mut self, bitstream: &mut Bitstream<R>) -> Result<()> {
        match self {
            Self::PrefixCode(_) => Ok(()),
            Self::Ans { state, initial, .. } => {
                *state = bitstream.read_bits(32)?;
                *initial = false;
                Ok(())
            },
        }
    }

    fn finalize(&self) -> Result<()> {
        match *self {
            Self::PrefixCode(_) => Ok(()),
            Self::Ans { state, .. } => {
                if state == 0x130000 {
                    Ok(())
                } else {
                    Err(Error::InvalidAnsStream)
                }
            },
        }
    }
}

fn add_log2_ceil(x: u32) -> u32 {
    (x + 1).next_power_of_two().trailing_zeros()
}

pub fn read_clusters<R: std::io::Read>(bitstream: &mut Bitstream<R>, num_dist: u32) -> Result<(u32, Vec<u8>)> {
    if num_dist == 1 {
        return Ok((1, vec![0u8]));
    }

    Ok(if bitstream.read_bool()? {
        // simple dist
        let nbits = bitstream.read_bits(2)?;
        let ret = (0..num_dist)
            .map(|_| bitstream.read_bits(nbits).map(|b| b as u8))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let num_clusters = *ret.iter().max().unwrap() as u32 + 1;
        (num_clusters, ret)
    } else {
        let use_mtf = read_bits!(bitstream, Bool)?;
        let mut decoder = if num_dist <= 2 {
            Decoder::parse_assume_no_lz77(bitstream, 1)?
        } else {
            Decoder::parse(bitstream, 1)?
        };
        decoder.begin(bitstream)?;
        let mut ret = (0..num_dist)
            .map(|_| decoder.read_varint(bitstream, 0).map(|b| b as u8))
            .collect::<Result<Vec<_>>>()?;
        decoder.finalize()?;
        if use_mtf {
            let mut mtfmap = [0u8; 256];
            for (idx, mtf) in mtfmap.iter_mut().enumerate() {
                *mtf = idx as u8;
            }
            for cluster in &mut ret {
                let idx = *cluster as usize;
                *cluster = mtfmap[idx];
                mtfmap.copy_within(0..idx, 1);
                mtfmap[0] = *cluster;
            }
        }
        let num_clusters = *ret.iter().max().unwrap() as u32 + 1;
        (num_clusters, ret)
    })
}
