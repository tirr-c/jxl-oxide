#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    Bitstream(jxl_bitstream::Error),
    Lz77NotAllowed,
    InvalidPrefixHistogram,
    InvalidAnsHistogram,
    InvalidAnsStream,
    InvalidPermutation,
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
            Self::InvalidPrefixHistogram => write!(f, "invalid Brotli prefix code"),
            Self::InvalidAnsHistogram => write!(f, "invalid ANS distribution"),
            Self::InvalidAnsStream => write!(f, "ANS stream verification failed"),
            Self::InvalidPermutation => write!(f, "invalid permutation"),
        }
    }
}

impl From<jxl_bitstream::Error> for Error {
    fn from(err: jxl_bitstream::Error) -> Self {
        Self::Bitstream(err)
    }
}
