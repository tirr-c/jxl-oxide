#[derive(Debug)]
pub enum Error {
    Bitstream(jxl_bitstream::Error),
    Decoder(jxl_coding::Error),
    Modular(jxl_modular::Error),
    InvalidTocPermutation,
    IncompleteFrameData {
        field: &'static str,
    },
}

impl From<jxl_bitstream::Error> for Error {
    fn from(err: jxl_bitstream::Error) -> Self {
        Self::Bitstream(err)
    }
}

impl From<jxl_coding::Error> for Error {
    fn from(err: jxl_coding::Error) -> Self {
        Self::Decoder(err)
    }
}

impl From<jxl_modular::Error> for Error {
    fn from(err: jxl_modular::Error) -> Self {
        Self::Modular(err)
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bitstream(err) => write!(f, "bitstream error: {}", err),
            Self::Decoder(err) => write!(f, "entropy decoder error: {}", err),
            Self::Modular(err) => write!(f, "modular stream error: {}", err),
            Self::InvalidTocPermutation => write!(f, "invalid TOC permutation"),
            Self::IncompleteFrameData { field } => write!(f, "incomplete frame data: {} is missing", field),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Bitstream(err) => Some(err),
            Self::Decoder(err) => Some(err),
            Self::Modular(err) => Some(err),
            _ => None,
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;
