//! This crate provides [`Decoder`], an entropy decoder, implemented as specified in the JPEG XL
//! specification.
//!
//! This also provides [`read_permutation`] and [`read_clusters`], which are used in some parts of
//! the specification.

use std::sync::Arc;

use jxl_bitstream::{Bitstream, U};

mod ans;
mod error;
mod permutation;
mod prefix;

pub use error::Error;

/// Shorthand for result type of entropy decoding.
pub type CodingResult<T> = std::result::Result<T, Error>;

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
    pub fn parse(bitstream: &mut Bitstream, num_dist: u32) -> CodingResult<Self> {
        let lz77 = Lz77::parse(bitstream)?;
        let num_dist = if let Lz77::Disabled = &lz77 {
            num_dist
        } else {
            num_dist + 1
        };
        let inner = DecoderInner::parse(bitstream, num_dist)?;
        Ok(Self { lz77, inner })
    }

    fn parse_assume_no_lz77(bitstream: &mut Bitstream, num_dist: u32) -> CodingResult<Self> {
        let lz77_enabled = bitstream.read_bool()?;
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
    pub fn read_varint(&mut self, bitstream: &mut Bitstream, ctx: u32) -> CodingResult<u32> {
        self.read_varint_with_multiplier(bitstream, ctx, 0)
    }

    /// Read an integer from the bitstream with the given context and LZ77 distance multiplier.
    #[inline]
    pub fn read_varint_with_multiplier(
        &mut self,
        bitstream: &mut Bitstream,
        ctx: u32,
        dist_multiplier: u32,
    ) -> CodingResult<u32> {
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
    ) -> CodingResult<u32> {
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

    /// Converts the decoder to one in RLE mode.
    pub fn as_rle(&mut self) -> Option<DecoderRleMode<'_>> {
        let &Lz77::Enabled {
            ref state,
            min_symbol,
            min_length,
        } = &self.lz77
        else {
            return None;
        };
        let lz_cluster = self.inner.lz_dist_cluster();
        let lz_conf = &self.inner.configs[lz_cluster as usize];
        let sym = self.inner.code.single_symbol(lz_cluster)?;
        (sym == 1 && lz_conf.split_exponent == 0).then_some(DecoderRleMode {
            inner: &mut self.inner,
            min_symbol,
            min_length,
            len_config: state.lz_len_conf.clone(),
        })
    }

    /// Converts the decoder to LZ77-enabled one.
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

    /// Converts the decoder to one without LZ77.
    pub fn as_no_lz77(&mut self) -> Option<DecoderNoLz77<'_>> {
        if let Lz77::Disabled = self.lz77 {
            Some(DecoderNoLz77(&mut self.inner))
        } else {
            None
        }
    }

    /// Returns the token to be decoded if the decoder always emits single token repeatedly.
    #[inline]
    pub fn single_token(&self, cluster: u8) -> Option<u32> {
        self.inner.single_token(cluster)
    }

    /// Explicitly start reading an entropy encoded stream.
    ///
    /// This involves reading an initial state for the ANS stream. It's okay to skip this method,
    /// as the state will be initialized on the first read.
    #[inline]
    pub fn begin(&mut self, bitstream: &mut Bitstream) -> CodingResult<()> {
        self.inner.code.begin(bitstream)
    }

    /// Finalizes the stream, and check whether the stream was valid.
    ///
    /// For prefix code stream, this method will always succeed. For ANS streams, this method
    /// checks if the final state matches expected state, which is specified in the specification.
    #[inline]
    pub fn finalize(&self) -> CodingResult<()> {
        self.inner.code.finalize()
    }

    /// Returns the cluster mapping of distributions.
    #[inline]
    pub fn cluster_map(&self) -> &[u8] {
        &self.inner.clusters
    }
}

/// An entropy decoder in RLE mode.
#[derive(Debug)]
pub struct DecoderRleMode<'dec> {
    inner: &'dec mut DecoderInner,
    min_symbol: u32,
    min_length: u32,
    len_config: IntegerConfig,
}

/// Decoded token from an entropy decoder in RLE mode.
#[derive(Debug, Copy, Clone)]
pub enum RleToken {
    /// Emit the given value once.
    Value(u32),
    /// Repeat previously decoded value by the given number of times.
    Repeat(u32),
}

impl DecoderRleMode<'_> {
    /// Read an integer from the bitstream with the given *cluster*.
    ///
    /// Contexts can be converted to clusters using [the cluster map][Self::cluster_map].
    #[inline]
    pub fn read_varint_clustered(
        &mut self,
        bitstream: &mut Bitstream,
        cluster: u8,
    ) -> CodingResult<RleToken> {
        self.inner
            .code
            .read_symbol(bitstream, cluster)
            .map(|token| {
                if let Some(token) = token.checked_sub(self.min_symbol) {
                    RleToken::Repeat(
                        self.inner
                            .read_uint_prefilled(bitstream, &self.len_config, token)
                            + self.min_length,
                    )
                } else {
                    RleToken::Value(self.inner.read_uint_prefilled(
                        bitstream,
                        &self.inner.configs[cluster as usize],
                        token,
                    ))
                }
            })
    }

    /// Returns the cluster mapping of distributions.
    #[inline]
    pub fn cluster_map(&self) -> &[u8] {
        &self.inner.clusters
    }
}

/// A LZ77-enabled entropy decoder.
#[derive(Debug)]
pub struct DecoderWithLz77<'dec> {
    inner: &'dec mut DecoderInner,
    state: &'dec mut Lz77State,
    min_symbol: u32,
    min_length: u32,
}

impl DecoderWithLz77<'_> {
    /// Read an integer from the bitstream with the given *cluster* and LZ77 distance multiplier.
    ///
    /// Contexts can be converted to clusters using [the cluster map][Self::cluster_map].
    #[inline]
    pub fn read_varint_with_multiplier_clustered(
        &mut self,
        bitstream: &mut Bitstream,
        cluster: u8,
        dist_multiplier: u32,
    ) -> CodingResult<u32> {
        self.inner.read_varint_with_multiplier_clustered_lz77(
            bitstream,
            cluster,
            dist_multiplier,
            self.state,
            self.min_symbol,
            self.min_length,
        )
    }

    /// Returns the cluster mapping of distributions.
    #[inline]
    pub fn cluster_map(&self) -> &[u8] {
        &self.inner.clusters
    }
}

/// An entropy decoder without LZ77.
#[derive(Debug)]
pub struct DecoderNoLz77<'dec>(&'dec mut DecoderInner);

impl DecoderNoLz77<'_> {
    /// Read an integer from the bitstream with the given *cluster*.
    ///
    /// Contexts can be converted to clusters using [the cluster map][Self::cluster_map].
    #[inline]
    pub fn read_varint_clustered(
        &mut self,
        bitstream: &mut Bitstream,
        cluster: u8,
    ) -> CodingResult<u32> {
        self.0
            .read_varint_with_multiplier_clustered(bitstream, cluster)
    }

    /// Returns the token to be decoded if the decoder always emits single token repeatedly.
    #[inline]
    pub fn single_token(&self, cluster: u8) -> Option<u32> {
        self.0.single_token(cluster)
    }

    /// Returns the cluster mapping of distributions.
    #[inline]
    pub fn cluster_map(&self) -> &[u8] {
        &self.0.clusters
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
    fn parse(bitstream: &mut Bitstream) -> CodingResult<Self> {
        Ok(if bitstream.read_bool()? {
            // enabled
            let min_symbol = bitstream.read_u32(224, 512, 4096, 8 + U(15))?;
            let min_length = bitstream.read_u32(3, 4, 5 + U(2), 9 + U(8))?;
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
    split_exponent: u32,
    split: u32,
    msb_in_token: u32,
    lsb_in_token: u32,
}

impl IntegerConfig {
    fn parse(bitstream: &mut Bitstream, log_alphabet_size: u32) -> CodingResult<Self> {
        let split_exponent_bits = add_log2_ceil(log_alphabet_size);
        let split_exponent = bitstream.read_bits(split_exponent_bits as usize)?;
        let (msb_in_token, lsb_in_token) = if split_exponent != log_alphabet_size {
            let msb_bits = add_log2_ceil(split_exponent) as usize;
            let msb_in_token = bitstream.read_bits(msb_bits)?;
            if msb_in_token > split_exponent {
                return Err(Error::InvalidIntegerConfig {
                    split_exponent,
                    msb_in_token,
                    lsb_in_token: None,
                });
            }
            let lsb_bits = add_log2_ceil(split_exponent - msb_in_token) as usize;
            let lsb_in_token = bitstream.read_bits(lsb_bits)?;
            (msb_in_token, lsb_in_token)
        } else {
            (0u32, 0u32)
        };
        if lsb_in_token + msb_in_token > split_exponent {
            return Err(Error::InvalidIntegerConfig {
                split_exponent,
                msb_in_token,
                lsb_in_token: Some(lsb_in_token),
            });
        }
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
    clusters: Vec<u8>,           // num_dist, [0, num_clusters)
    configs: Vec<IntegerConfig>, // num_clusters
    code: Coder,
}

impl DecoderInner {
    fn parse(bitstream: &mut Bitstream, num_dist: u32) -> CodingResult<Self> {
        let (num_clusters, clusters) = read_clusters(bitstream, num_dist)?;
        let use_prefix_code = bitstream.read_bool()?;
        let log_alphabet_size = if use_prefix_code {
            15
        } else {
            bitstream.read_bits(2)? + 5
        };
        let configs = (0..num_clusters)
            .map(|_| IntegerConfig::parse(bitstream, log_alphabet_size))
            .collect::<CodingResult<Vec<_>>>()?;
        let code = if use_prefix_code {
            let counts = (0..num_clusters)
                .map(|_| -> CodingResult<_> {
                    let count = if bitstream.read_bool()? {
                        let n = bitstream.read_bits(4)? as usize;
                        1 + (1 << n) + bitstream.read_bits(n)?
                    } else {
                        1
                    };
                    if count > 1 << 15 {
                        return Err(Error::InvalidPrefixHistogram);
                    }
                    Ok(count)
                })
                .collect::<CodingResult<Vec<_>>>()?;
            let dist = counts
                .into_iter()
                .map(|count| prefix::Histogram::parse(bitstream, count))
                .collect::<CodingResult<Vec<_>>>()?;
            Coder::PrefixCode(Arc::new(dist))
        } else {
            let dist = (0..num_clusters)
                .map(|_| ans::Histogram::parse(bitstream, log_alphabet_size))
                .collect::<CodingResult<Vec<_>>>()?;
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

    #[inline]
    fn single_token(&self, cluster: u8) -> Option<u32> {
        let single_symbol = self.code.single_symbol(cluster)?;
        let IntegerConfig { split, .. } = self.configs[cluster as usize];
        (single_symbol < split).then_some(single_symbol)
    }

    #[inline]
    fn read_varint_with_multiplier_clustered(
        &mut self,
        bitstream: &mut Bitstream,
        cluster: u8,
    ) -> CodingResult<u32> {
        let token = self.code.read_symbol(bitstream, cluster)?;
        Ok(self.read_uint_prefilled(bitstream, &self.configs[cluster as usize], token))
    }

    fn read_varint_with_multiplier_clustered_lz77(
        &mut self,
        bitstream: &mut Bitstream,
        cluster: u8,
        dist_multiplier: u32,
        state: &mut Lz77State,
        min_symbol: u32,
        min_length: u32,
    ) -> CodingResult<u32> {
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
            let token = self.code.read_symbol(bitstream, cluster)?;
            if token >= min_symbol {
                if state.num_decoded == 0 {
                    tracing::error!("LZ77 repeat symbol encountered without decoding any symbols");
                    return Err(Error::UnexpectedLz77Repeat);
                }

                let lz_dist_cluster = self.lz_dist_cluster();

                let num_to_copy =
                    self.read_uint_prefilled(bitstream, &state.lz_len_conf, token - min_symbol);
                let Some(num_to_copy) = num_to_copy.checked_add(min_length) else {
                    tracing::error!(num_to_copy, min_length, "LZ77 num_to_copy overflow");
                    return Err(Error::InvalidLz77Symbol);
                };
                state.num_to_copy = num_to_copy;

                let token = self.code.read_symbol(bitstream, lz_dist_cluster)?;
                let distance = self.read_uint_prefilled(
                    bitstream,
                    &self.configs[lz_dist_cluster as usize],
                    token,
                );
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
                r = self.read_uint_prefilled(bitstream, &self.configs[cluster as usize], token);
            }
        }
        let offset = (state.num_decoded & 0xfffff) as usize;
        if state.window.len() <= offset {
            state.window.push(r);
        } else {
            state.window[offset] = r;
        }
        state.num_decoded += 1;
        Ok(r)
    }

    #[inline]
    fn read_uint_prefilled(
        &self,
        bitstream: &mut Bitstream,
        config: &IntegerConfig,
        token: u32,
    ) -> u32 {
        let &IntegerConfig {
            split_exponent,
            split,
            msb_in_token,
            lsb_in_token,
            ..
        } = config;
        if token < split {
            return token;
        }

        let n = split_exponent - (msb_in_token + lsb_in_token)
            + ((token - split) >> (msb_in_token + lsb_in_token));
        // n < 32.
        let n = n & 31;
        let rest_bits = bitstream.peek_bits_prefilled(n as usize) as u64;
        bitstream.consume_bits(n as usize).ok();

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
    fn read_symbol(&mut self, bitstream: &mut Bitstream, cluster: u8) -> CodingResult<u32> {
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
                    *state = bitstream.read_bits(32)?;
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

    fn begin(&mut self, bitstream: &mut Bitstream) -> CodingResult<()> {
        match self {
            Self::PrefixCode(_) => Ok(()),
            Self::Ans { state, initial, .. } => {
                *state = bitstream.read_bits(32)?;
                *initial = false;
                Ok(())
            }
        }
    }

    fn finalize(&self) -> CodingResult<()> {
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

/// Read a distribution clustering from the bitstream.
pub fn read_clusters(bitstream: &mut Bitstream, num_dist: u32) -> CodingResult<(u32, Vec<u8>)> {
    if num_dist == 1 {
        return Ok((1, vec![0u8]));
    }

    let cluster = if bitstream.read_bool()? {
        // simple dist
        let nbits = bitstream.read_bits(2)? as usize;
        (0..num_dist)
            .map(|_| bitstream.read_bits(nbits).map(|b| b as u8))
            .collect::<std::result::Result<Vec<_>, _>>()?
    } else {
        let use_mtf = bitstream.read_bool()?;
        let mut decoder = if num_dist <= 2 {
            Decoder::parse_assume_no_lz77(bitstream, 1)?
        } else {
            Decoder::parse(bitstream, 1)?
        };
        decoder.begin(bitstream)?;
        let mut ret = (0..num_dist)
            .map(|_| -> CodingResult<_> {
                let b = decoder.read_varint(bitstream, 0)?;
                u8::try_from(b).map_err(|_| Error::InvalidCluster(b))
            })
            .collect::<CodingResult<Vec<_>>>()?;
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
        ret
    };

    let num_clusters = *cluster.iter().max().unwrap() as u32 + 1;
    let set = cluster
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>();
    let num_expected_clusters = num_clusters;
    let num_actual_clusters = set.len() as u32;
    if num_actual_clusters != num_expected_clusters {
        tracing::error!(
            num_expected_clusters,
            num_actual_clusters,
            "distribution cluster has a hole"
        );
        Err(Error::ClusterHole {
            num_expected_clusters,
            num_actual_clusters,
        })
    } else {
        Ok((num_clusters, cluster))
    }
}
