//! This crate provides [`Decoder`], an entropy decoder, implemented as specified in the JPEG XL
//! specification.
//!
//! This also provides [`read_permutation`] and [`read_clusters`], which are used in some parts of
//! the specification.

use std::sync::Arc;

use jxl_bitstream::{read_bits, Bitstream};

mod ans;
mod error;
mod permutation;
mod prefix;

pub use error::Error;
pub type Result<T> = std::result::Result<T, Error>;

pub use permutation::read_permutation;

/// An entropy decoder.
#[derive(Debug, Clone)]
pub struct Decoder {
    lz77: Lz77,
    inner: DecoderInner,
}

impl Decoder {
    /// Create a decoder by reading symbol distribution, integer configurations and LZ77
    /// configuration from the bitstream.
    pub fn parse(bitstream: &mut Bitstream, num_dist: u32) -> Result<Self> {
        let lz77 = Lz77::parse(bitstream)?;
        let num_dist = if let Lz77::Disabled = &lz77 {
            num_dist
        } else {
            num_dist + 1
        };
        let inner = DecoderInner::parse(bitstream, num_dist)?;
        Ok(Self { lz77, inner })
    }

    fn parse_assume_no_lz77(bitstream: &mut Bitstream, num_dist: u32) -> Result<Self> {
        let lz77_enabled = read_bits!(bitstream, Bool)?;
        if lz77_enabled {
            return Err(Error::Lz77NotAllowed);
        }
        let inner = DecoderInner::parse(bitstream, num_dist)?;
        Ok(Self {
            lz77: Lz77::Disabled,
            inner,
        })
    }

    /// Read an integer from the bitstream with the given context.
    #[inline]
    pub fn read_varint(&mut self, bitstream: &mut Bitstream, ctx: u32) -> u32 {
        self.read_varint_with_multiplier(bitstream, ctx, 0)
    }

    /// Read an integer from the bitstream with the given context and LZ77 distance multiplier.
    #[inline]
    pub fn read_varint_with_multiplier(
        &mut self,
        bitstream: &mut Bitstream,
        ctx: u32,
        dist_multiplier: u32,
    ) -> u32 {
        let cluster = self.inner.clusters[ctx as usize];
        self.read_varint_with_multiplier_clustered(bitstream, cluster, dist_multiplier)
    }

    /// Read an integer from the bitstream with the given *cluster* and LZ77 distance multiplier.
    ///
    /// Contexts can be converted to clusters using [the cluster map][Self::cluster_map].
    #[inline(always)]
    pub fn read_varint_with_multiplier_clustered(
        &mut self,
        bitstream: &mut Bitstream,
        cluster: u8,
        dist_multiplier: u32,
    ) -> u32 {
        if let Lz77::Enabled {
            ref mut state,
            min_symbol,
            min_length,
        } = self.lz77
        {
            self.inner.read_varint_with_multiplier_clustered_lz77(
                bitstream,
                cluster,
                dist_multiplier,
                state,
                min_symbol,
                min_length,
            )
        } else {
            self.inner
                .read_varint_with_multiplier_clustered(bitstream, cluster)
        }
    }

    pub fn as_rle(&self) -> Option<DecoderRleMode> {
        let &Lz77::Enabled {
            ref state,
            min_symbol,
            min_length,
        } = &self.lz77
        else {
            return None;
        };

        let Coder::PrefixCode(dist) = &self.inner.code else {
            return None;
        };

        let lz_cluster = self.inner.lz_dist_cluster();
        let lz_conf = &self.inner.configs[lz_cluster as usize];
        if lz_conf.split_exponent != 0 {
            return None;
        }

        let sym = dist[lz_cluster as usize].single_symbol()?;
        if sym != 1 {
            return None;
        }

        Some(DecoderRleMode {
            configs: &self.inner.configs,
            dist,
            min_symbol,
            min_length,
            len_config: state.lz_len_conf.clone(),
            current_config: &self.inner.configs[0],
            current_dist: &dist[0],
        })
    }

    pub fn as_with_lz77(&mut self) -> Option<DecoderWithLz77<'_>> {
        if let Lz77::Enabled {
            ref mut state,
            min_symbol,
            min_length,
        } = self.lz77
        {
            Some(DecoderWithLz77 {
                inner: &mut self.inner,
                state,
                min_symbol,
                min_length,
            })
        } else {
            None
        }
    }

    pub fn as_no_lz77(&mut self) -> Option<DecoderNoLz77<'_>> {
        if let Lz77::Disabled = self.lz77 {
            Some(DecoderNoLz77(&mut self.inner))
        } else {
            None
        }
    }

    #[inline]
    pub fn single_token(&self, cluster: u8) -> Option<u32> {
        self.inner.single_token(cluster)
    }

    /// Explicitly start reading an entropy encoded stream.
    ///
    /// This involves reading an initial state for the ANS stream. It's okay to skip this method,
    /// as the state will be initialized on the first read.
    #[inline]
    pub fn begin(&mut self, bitstream: &mut Bitstream) -> Result<()> {
        self.inner.code.begin(bitstream)
    }

    /// Finalizes the stream, and check whether the stream was valid.
    ///
    /// For prefix code stream, this method will always succeed. For ANS streams, this method
    /// checks if the final state matches expected state, which is specified in the specification.
    #[inline]
    pub fn finalize(&self, bitstream: &mut Bitstream) -> Result<()> {
        bitstream.check_in_bounds()?;
        if self.inner.lz77_error {
            return Err(Error::UnexpectedLz77Repeat);
        }
        self.inner.code.finalize()
    }

    /// Returns the cluster mapping of distributions.
    #[inline]
    pub fn cluster_map(&self) -> &[u8] {
        &self.inner.clusters
    }
}

/// An entropy decoder, in RLE mode.
#[derive(Debug)]
pub struct DecoderRleMode<'dec> {
    configs: &'dec [IntegerConfig],
    dist: &'dec [prefix::Histogram],
    min_symbol: u32,
    min_length: u32,
    len_config: IntegerConfig,
    current_config: &'dec IntegerConfig,
    current_dist: &'dec prefix::Histogram,
}

#[derive(Debug, Copy, Clone)]
pub enum RleToken {
    Value(u32),
    Repeat(u32),
}

impl DecoderRleMode<'_> {
    #[inline]
    pub fn select_cluster(&mut self, cluster: u8) {
        self.current_dist = &self.dist[cluster as usize];
        self.current_config = &self.configs[cluster as usize];
    }

    #[inline(always)]
    pub fn read_varint_clustered(&mut self, bitstream: &mut Bitstream) -> RleToken {
        let token = self.current_dist.read_symbol(bitstream);
        if let Some(token) = token.checked_sub(self.min_symbol) {
            RleToken::Repeat(
                self.len_config.read_uint_prefilled(bitstream, token) + self.min_length,
            )
        } else {
            RleToken::Value(self.current_config.read_uint_prefilled(bitstream, token))
        }
    }
}

#[derive(Debug)]
pub struct DecoderWithLz77<'dec> {
    inner: &'dec mut DecoderInner,
    state: &'dec mut Lz77State,
    min_symbol: u32,
    min_length: u32,
}

impl DecoderWithLz77<'_> {
    #[inline]
    pub fn read_varint_with_multiplier_clustered(
        &mut self,
        bitstream: &mut Bitstream,
        cluster: u8,
        dist_multiplier: u32,
    ) -> u32 {
        self.inner.read_varint_with_multiplier_clustered_lz77(
            bitstream,
            cluster,
            dist_multiplier,
            self.state,
            self.min_symbol,
            self.min_length,
        )
    }
}

#[derive(Debug)]
pub struct DecoderNoLz77<'dec>(&'dec mut DecoderInner);

impl DecoderNoLz77<'_> {
    #[inline]
    pub fn read_varint_clustered(&mut self, bitstream: &mut Bitstream, cluster: u8) -> u32 {
        self.0
            .read_varint_with_multiplier_clustered(bitstream, cluster)
    }

    #[inline]
    pub fn single_token(&self, cluster: u8) -> Option<u32> {
        self.0.single_token(cluster)
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
    fn parse(bitstream: &mut Bitstream) -> Result<Self> {
        Ok(if read_bits!(bitstream, Bool)? {
            // enabled
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
    fn new(lz_len_conf: IntegerConfig) -> Self {
        Self {
            lz_len_conf,
            window: Vec::new(),
            num_to_copy: 0,
            copy_pos: 0,
            num_decoded: 0,
        }
    }
}

#[derive(Debug, Clone)]
struct IntegerConfig {
    split_exponent: u8,
    msb_in_token: u8,
    lsb_in_token: u8,
}

impl IntegerConfig {
    fn parse(bitstream: &mut Bitstream, log_alphabet_size: u32) -> Result<Self> {
        let split_exponent_bits = add_log2_ceil(log_alphabet_size);
        let split_exponent = bitstream.read_bits(split_exponent_bits)?;
        let (msb_in_token, lsb_in_token) = if split_exponent != log_alphabet_size {
            let msb_bits = add_log2_ceil(split_exponent);
            let msb_in_token = bitstream.read_bits(msb_bits)?;
            if msb_in_token > split_exponent {
                return Err(Error::InvalidIntegerConfig);
            }
            let lsb_bits = add_log2_ceil(split_exponent - msb_in_token);
            let lsb_in_token = bitstream.read_bits(lsb_bits)?;
            (msb_in_token, lsb_in_token)
        } else {
            (0u32, 0u32)
        };
        if lsb_in_token + msb_in_token > split_exponent {
            return Err(Error::InvalidIntegerConfig);
        }
        Ok(Self {
            split_exponent: split_exponent as u8,
            msb_in_token: msb_in_token as u8,
            lsb_in_token: lsb_in_token as u8,
        })
    }

    #[inline]
    fn read_uint_prefilled(&self, bitstream: &mut Bitstream, token: u32) -> u32 {
        let &IntegerConfig {
            split_exponent,
            msb_in_token,
            lsb_in_token,
        } = self;
        let split = 1 << split_exponent;
        if token < split {
            return token;
        }

        let n = split_exponent - (msb_in_token + lsb_in_token)
            + ((token - split) >> (msb_in_token + lsb_in_token)) as u8;
        // n < 32.
        let n = (n & 31) as u32;
        let rest_bits = bitstream.peek_bits_prefilled(n) as u64;
        bitstream.consume_bits_silent(n);

        let low_bits = token & ((1 << lsb_in_token) - 1);
        let low_bits = low_bits as u64;
        let token = token >> lsb_in_token;
        let token = token & ((1 << msb_in_token) - 1);
        let token = token | (1 << msb_in_token);
        let token = token as u64;
        let result = (((token << n) | rest_bits) << lsb_in_token) | low_bits;
        // result fits in u32.
        result as u32
    }
}

#[derive(Debug, Clone)]
struct DecoderInner {
    clusters: Vec<u8>,           // num_dist, [0, num_clusters)
    configs: Vec<IntegerConfig>, // num_clusters
    code: Coder,
    lz77_error: bool,
}

impl DecoderInner {
    fn parse(bitstream: &mut Bitstream, num_dist: u32) -> Result<Self> {
        let (num_clusters, clusters) = read_clusters(bitstream, num_dist)?;
        let use_prefix_code = read_bits!(bitstream, Bool)?;
        let log_alphabet_size = if use_prefix_code {
            15
        } else {
            bitstream.read_bits(2)? + 5
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
            lz77_error: false,
        })
    }

    #[inline]
    fn single_token(&self, cluster: u8) -> Option<u32> {
        let single_symbol = self.code.single_symbol(cluster)?;
        let IntegerConfig { split_exponent, .. } = self.configs[cluster as usize];
        (single_symbol < (1 << split_exponent)).then_some(single_symbol)
    }

    #[inline]
    pub fn read_varint_with_multiplier_clustered(
        &mut self,
        bitstream: &mut Bitstream,
        cluster: u8,
    ) -> u32 {
        let token = self.code.read_symbol(bitstream, cluster);
        self.configs[cluster as usize].read_uint_prefilled(bitstream, token)
    }

    fn read_varint_with_multiplier_clustered_lz77(
        &mut self,
        bitstream: &mut Bitstream,
        cluster: u8,
        dist_multiplier: u32,
        state: &mut Lz77State,
        min_symbol: u32,
        min_length: u32,
    ) -> u32 {
        #[rustfmt::skip]
        const SPECIAL_DISTANCES: [[i8; 2]; 120] = [
            [0, 1], [1, 0], [1, 1], [-1, 1], [0, 2], [2, 0], [1, 2], [-1, 2], [2, 1], [-2, 1],
            [2, 2], [-2, 2], [0, 3], [3, 0], [1, 3], [-1, 3], [3, 1], [-3, 1], [2, 3], [-2, 3],
            [3, 2], [-3, 2], [0, 4], [4, 0], [1, 4], [-1, 4], [4, 1], [-4, 1], [3, 3], [-3, 3],
            [2, 4], [-2, 4], [4, 2], [-4, 2], [0, 5], [3, 4], [-3, 4], [4, 3], [-4, 3], [5, 0],
            [1, 5], [-1, 5], [5, 1], [-5, 1], [2, 5], [-2, 5], [5, 2], [-5, 2], [4, 4], [-4, 4],
            [3, 5], [-3, 5], [5, 3], [-5, 3], [0, 6], [6, 0], [1, 6], [-1, 6], [6, 1], [-6, 1],
            [2, 6], [-2, 6], [6, 2], [-6, 2], [4, 5], [-4, 5], [5, 4], [-5, 4], [3, 6], [-3, 6],
            [6, 3], [-6, 3], [0, 7], [7, 0], [1, 7], [-1, 7], [5, 5], [-5, 5], [7, 1], [-7, 1],
            [4, 6], [-4, 6], [6, 4], [-6, 4], [2, 7], [-2, 7], [7, 2], [-7, 2], [3, 7], [-3, 7],
            [7, 3], [-7, 3], [5, 6], [-5, 6], [6, 5], [-6, 5], [8, 0], [4, 7], [-4, 7], [7, 4],
            [-7, 4], [8, 1], [8, 2], [6, 6], [-6, 6], [8, 3], [5, 7], [-5, 7], [7, 5], [-7, 5],
            [8, 4], [6, 7], [-6, 7], [7, 6], [-7, 6], [8, 5], [7, 7], [-7, 7], [8, 6], [8, 7],
        ];

        let r;
        if state.num_to_copy > 0 {
            r = state.window[(state.copy_pos & 0xfffff) as usize];
            state.copy_pos += 1;
            state.num_to_copy -= 1;
        } else {
            let token = self.code.read_symbol(bitstream, cluster);
            if token >= min_symbol {
                if state.num_decoded == 0 {
                    tracing::error!("LZ77 repeat symbol encountered without decoding any symbols");
                    self.lz77_error = true;
                    state.window.push(0);
                    state.num_decoded += 1;
                    return 0;
                }

                let lz_dist_cluster = self.lz_dist_cluster();

                state.num_to_copy = state
                    .lz_len_conf
                    .read_uint_prefilled(bitstream, token - min_symbol)
                    + min_length;
                let token = self.code.read_symbol(bitstream, lz_dist_cluster);
                let distance =
                    self.configs[lz_dist_cluster as usize].read_uint_prefilled(bitstream, token);
                let distance = if dist_multiplier == 0 {
                    distance
                } else if distance < 120 {
                    let [offset, dist] = SPECIAL_DISTANCES[distance as usize];
                    let dist = offset as i32 + dist_multiplier as i32 * dist as i32;
                    (dist - 1).max(0) as u32
                } else {
                    distance - 120
                };

                let distance = (((1 << 20) - 1).min(distance) + 1).min(state.num_decoded);
                state.copy_pos = state.num_decoded - distance;

                r = state.window[(state.copy_pos & 0xfffff) as usize];
                state.copy_pos += 1;
                state.num_to_copy -= 1;
            } else {
                r = self.configs[cluster as usize].read_uint_prefilled(bitstream, token);
            }
        }
        let offset = (state.num_decoded & 0xfffff) as usize;
        if state.window.len() <= offset {
            state.window.push(r);
        } else {
            state.window[offset] = r;
        }
        state.num_decoded += 1;
        r
    }

    #[inline]
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
    #[inline(always)]
    fn read_symbol(&mut self, bitstream: &mut Bitstream, cluster: u8) -> u32 {
        match self {
            Self::PrefixCode(dist) => {
                let dist = &dist[cluster as usize];
                dist.read_symbol(bitstream)
            }
            Self::Ans {
                dist,
                state,
                initial,
            } => {
                if *initial {
                    *state = bitstream.peek_bits(32);
                    bitstream.consume_bits_silent(32);
                    *initial = false;
                }
                let dist = &dist[cluster as usize];
                dist.read_symbol(bitstream, state)
            }
        }
    }

    #[inline]
    fn single_symbol(&self, cluster: u8) -> Option<u32> {
        match self {
            Self::PrefixCode(dist) => dist[cluster as usize].single_symbol(),
            Self::Ans { dist, .. } => dist[cluster as usize].single_symbol(),
        }
    }

    fn begin(&mut self, bitstream: &mut Bitstream) -> Result<()> {
        match self {
            Self::PrefixCode(_) => Ok(()),
            Self::Ans { state, initial, .. } => {
                *state = bitstream.read_bits(32)?;
                *initial = false;
                Ok(())
            }
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
            }
        }
    }
}

fn add_log2_ceil(x: u32) -> u32 {
    if x >= 0x80000000 {
        32
    } else {
        (x + 1).next_power_of_two().trailing_zeros()
    }
}

/// Read a clustering information of distributions from the bitstream.
pub fn read_clusters(bitstream: &mut Bitstream, num_dist: u32) -> Result<(u32, Vec<u8>)> {
    if num_dist == 1 {
        return Ok((1, vec![0u8]));
    }

    let cluster = if bitstream.read_bool()? {
        // simple dist
        let nbits = bitstream.read_bits(2)?;
        (0..num_dist)
            .map(|_| bitstream.read_bits(nbits).map(|b| b as u8))
            .collect::<std::result::Result<Vec<_>, _>>()?
    } else {
        let use_mtf = read_bits!(bitstream, Bool)?;
        let mut decoder = if num_dist <= 2 {
            Decoder::parse_assume_no_lz77(bitstream, 1)?
        } else {
            Decoder::parse(bitstream, 1)?
        };
        decoder.begin(bitstream)?;
        let mut ret = (0..num_dist)
            .map(|_| -> Result<_> {
                let b = decoder.read_varint(bitstream, 0);
                u8::try_from(b).map_err(|_| Error::InvalidCluster)
            })
            .collect::<Result<Vec<_>>>()?;
        decoder.finalize(bitstream)?;
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
        ret
    };

    let num_clusters = *cluster.iter().max().unwrap() as u32 + 1;
    let set = cluster
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>();
    if set.len() != num_clusters as usize {
        tracing::error!("distribution cluster has a hole");
        Err(Error::InvalidCluster)
    } else {
        Ok((num_clusters, cluster))
    }
}
