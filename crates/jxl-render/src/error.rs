#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    Bitstream(jxl_bitstream::Error),
    Decoder(jxl_coding::Error),
    Frame(jxl_frame::Error),
    Color(jxl_color::Error),
    IncompleteFrame,
    InvalidReference(u32),
    NotReady,
    NotSupported(&'static str),
    TooLargeEstimatedArea(u64),
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

impl From<jxl_frame::Error> for Error {
    fn from(err: jxl_frame::Error) -> Self {
        Self::Frame(err)
    }
}

impl From<jxl_color::Error> for Error {
    fn from(err: jxl_color::Error) -> Self {
        Self::Color(err)
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use Error::*;

        match self {
            Bitstream(err) => write!(f, "bitstream error: {}", err),
            Decoder(err) => write!(f, "entropy decoder error: {}", err),
            Frame(err) => write!(f, "frame error: {}", err),
            Color(err) => write!(f, "color management error: {err}"),
            IncompleteFrame => write!(f, "frame data is incomplete"),
            InvalidReference(idx) => write!(f, "invalid reference {idx}"),
            NotReady => write!(f, "image is not ready to be rendered"),
            NotSupported(msg) => write!(f, "not supported: {}", msg),
            TooLargeEstimatedArea(area) => write!(f, "Too large estimated area for splines: {}", area),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        use Error::*;

        match self {
            Bitstream(err) => Some(err),
            Decoder(err) => Some(err),
            Frame(err) => Some(err),
            Color(err) => Some(err),
            _ => None,
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;
