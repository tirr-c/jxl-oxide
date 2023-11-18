#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// Decoded MA tree is invalid.
    InvalidMaTree,
    /// Global MA tree is requested but not available.
    GlobalMaTreeNotAvailable,
    /// Decoded Rct transform parameters are invalid.
    InvalidRctParams,
    /// Decoded Palette transform parameters are invalid.
    InvalidPaletteParams,
    /// Decoded Squeeze transform parameters are invalid.
    InvalidSqueezeParams,
    Bitstream(jxl_bitstream::Error),
    Decoder(jxl_coding::Error),
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

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use Error::*;

        match self {
            Self::InvalidMaTree => write!(f, "invalid meta-adaptive tree"),
            Self::GlobalMaTreeNotAvailable => {
                write!(f, "global meta-adaptive tree requested but unavailable")
            }
            Self::InvalidRctParams => write!(f, "invalid Rct transform parameters"),
            Self::InvalidPaletteParams => write!(f, "invalid Palette transform parameters"),
            Self::InvalidSqueezeParams => write!(f, "invalid Squeeze transform parameters"),
            Bitstream(err) => write!(f, "bitstream error: {}", err),
            Decoder(err) => write!(f, "entropy decoder error: {}", err),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        use Error::*;

        match self {
            Bitstream(err) => Some(err),
            Decoder(err) => Some(err),
            _ => None,
        }
    }
}

impl Error {
    pub fn unexpected_eof(&self) -> bool {
        match self {
            Error::Bitstream(e) => e.unexpected_eof(),
            Error::Decoder(e) => e.unexpected_eof(),
            _ => false,
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;
