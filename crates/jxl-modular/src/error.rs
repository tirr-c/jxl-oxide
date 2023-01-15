#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    InvalidMaTree,
    GlobalMaTreeNotAvailable,
    InvalidPaletteParams,
    InvalidSqueezeParams,
    PropertyNotFound {
        num_properties: usize,
        property_ref: usize,
    },
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
            Self::GlobalMaTreeNotAvailable => write!(f, "global meta-adaptive tree requested but unavailable"),
            Self::InvalidPaletteParams => write!(f, "invalid Palette transform parameters"),
            Self::InvalidSqueezeParams => write!(f, "invalid Squeeze transform parameters"),
            Self::PropertyNotFound { num_properties, property_ref } => {
                write!(f, "property {} not found ({} given)", property_ref, num_properties)
            },
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

pub type Result<T> = std::result::Result<T, Error>;
