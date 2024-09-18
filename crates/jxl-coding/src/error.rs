#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    Bitstream(jxl_bitstream::Error),
    Lz77NotAllowed,
    InvalidAnsHistogram,
    InvalidAnsStream,
    InvalidIntegerConfig {
        split_exponent: u32,
        msb_in_token: u32,
        lsb_in_token: Option<u32>,
    },
    InvalidPermutation,
    InvalidPrefixHistogram,
    PrefixSymbolTooLarge(usize),
    InvalidCluster(u32),
    ClusterHole {
        num_expected_clusters: u32,
        num_actual_clusters: u32,
    },
    UnexpectedLz77Repeat,
    InvalidLz77Symbol,
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Bitstream(err) => Some(err),
            _ => None,
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bitstream(err) => write!(f, "error from bitstream: {}", err),
            Self::Lz77NotAllowed => write!(f, "LZ77-enabled decoder when it is not allowed"),
            Self::InvalidAnsHistogram => write!(f, "invalid ANS distribution"),
            Self::InvalidAnsStream => write!(f, "ANS stream verification failed"),
            Self::InvalidIntegerConfig {
                split_exponent,
                msb_in_token,
                lsb_in_token,
            } => write!(f, "invalid hybrid integer configuration; {} + {msb_in_token} > {split_exponent}", lsb_in_token.unwrap_or(0)),
            Self::InvalidPermutation => write!(f, "invalid permutation"),
            Self::InvalidPrefixHistogram => write!(f, "invalid Brotli prefix code"),
            Self::PrefixSymbolTooLarge(size) => write!(f, "prefix code symbol too large ({size})"),
            Self::InvalidCluster(id) => write!(f, "invalid cluster ID {id}"),
            Self::ClusterHole {
                num_expected_clusters,
                num_actual_clusters,
            } => write!(f, "distribution cluster has a hole; expected {num_expected_clusters}, actual {num_actual_clusters}"),
            Self::UnexpectedLz77Repeat => write!(
                f,
                "LZ77 repeat symbol encountered without decoding any symbols"
            ),
            Self::InvalidLz77Symbol => write!(f, "Invalid LZ77 symbol"),
        }
    }
}

impl From<jxl_bitstream::Error> for Error {
    fn from(err: jxl_bitstream::Error) -> Self {
        Self::Bitstream(err)
    }
}

impl Error {
    pub fn unexpected_eof(&self) -> bool {
        if let Error::Bitstream(e) = self {
            return e.unexpected_eof();
        }
        false
    }
}
