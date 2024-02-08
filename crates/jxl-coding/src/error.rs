#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    Bitstream(jxl_bitstream::Error),
    Lz77NotAllowed,
    InvalidAnsHistogram,
    InvalidAnsStream,
    InvalidIntegerConfig,
    InvalidPermutation,
    InvalidPrefixHistogram,
    UnexpectedLz77Repeat,
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
            Self::InvalidIntegerConfig => write!(f, "invalid hybrid integer configuration"),
            Self::InvalidPermutation => write!(f, "invalid permutation"),
            Self::InvalidPrefixHistogram => write!(f, "invalid Brotli prefix code"),
            Self::UnexpectedLz77Repeat => write!(
                f,
                "LZ77 repeat symbol encountered without decoding any symbols"
            ),
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
