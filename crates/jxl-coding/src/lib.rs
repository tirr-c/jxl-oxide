use std::io::Read;

use jxl_bitstream::{Bitstream, read_bits};

mod ans;
mod error;
mod prefix;

pub use error::Error;
pub type Result<T> = std::result::Result<T, Error>;

pub struct Decoder {
    lz77: Lz77,
    num_clusters: u32,
    clusters: Vec<u8>, // num_dist, [0, num_clusters)
    configs: Vec<IntegerConfig>, // num_clusters
    code: Coder,
}

impl Decoder {
    pub fn parse<R: std::io::Read>(bitstream: &mut Bitstream<R>, num_dist: u32) -> Result<Self> {
        let lz77 = Lz77::parse(bitstream)?;
        Self::parse_inner(bitstream, num_dist, lz77)
    }

    fn parse_assume_no_lz77<R: std::io::Read>(bitstream: &mut Bitstream<R>, num_dist: u32) -> Result<Self> {
        let lz77_enabled = read_bits!(bitstream, Bool)?;
        if lz77_enabled {
            return Err(Error::Lz77NotAllowed);
        }
        Self::parse_inner(bitstream, num_dist, Lz77::Disabled)
    }

    fn parse_inner<R: std::io::Read>(bitstream: &mut Bitstream<R>, num_dist: u32, lz77: Lz77) -> Result<Self> {
        let num_dist = if let Lz77::Disabled = &lz77 {
            num_dist
        } else {
            num_dist + 1
        };
        let (num_clusters, clusters) = Self::read_clusters(bitstream, num_dist)?;
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
            let dist = (0..num_clusters)
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
                    prefix::Histogram::parse(bitstream, count)
                })
                .collect::<Result<Vec<_>>>()?;
            Coder::PrefixCode(dist)
        } else {
            let dist = todo!();
            Coder::Ans {
                dist,
                state: 0,
                initial: true,
            }
        };
        Ok(Self {
            lz77,
            num_clusters,
            clusters,
            configs,
            code,
        })
    }

    fn read_clusters<R: std::io::Read>(bitstream: &mut Bitstream<R>, num_dist: u32) -> Result<(u32, Vec<u8>)> {
        if num_dist == 1 {
            return Ok((1, vec![0u8]));
        }

        let is_simple = read_bits!(bitstream, Bool)?;
        todo!()
    }

    pub fn read_varint<R: std::io::Read>(&mut self, bitstream: &mut Bitstream<R>) -> Result<u32> {
        todo!()
    }
}

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

struct Lz77State {
    lz_len_conf: IntegerConfig,
    window: Vec<u32>,
    num_to_copy: u32,
    copy_pos: u32,
    num_decoded: u32,
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

struct IntegerConfig {
    split_exponent: u32,
    msb_in_token: u32,
    lsb_in_token: u32,
}

impl IntegerConfig {
    fn parse<R: std::io::Read>(bitstream: &mut Bitstream<R>, log_alphabet_size: u32) -> Result<Self> {
        let split_exponent_bits = log_alphabet_size.next_power_of_two().trailing_zeros();
        let split_exponent = bitstream.read_bits(split_exponent_bits)?;
        let (msb_in_token, lsb_in_token) = if split_exponent != log_alphabet_size {
            let msb_bits = split_exponent.next_power_of_two().trailing_zeros();
            let msb_in_token = bitstream.read_bits(msb_bits)?;
            let lsb_bits = (split_exponent - msb_in_token).next_power_of_two().trailing_zeros();
            let lsb_in_token = bitstream.read_bits(lsb_bits)?;
            (msb_in_token, lsb_in_token)
        } else {
            (0u32, 0u32)
        };
        Ok(Self {
            split_exponent,
            msb_in_token,
            lsb_in_token,
        })
    }

    pub fn split(&self) -> u32 {
        1 << self.split_exponent
    }
}

#[derive(Debug)]
enum Coder {
    PrefixCode(Vec<prefix::Histogram>),
    Ans {
        dist: Vec<ans::Histogram>,
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
}
