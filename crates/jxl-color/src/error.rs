#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    Bitstream(jxl_bitstream::Error),
    Decoder(jxl_coding::Error),
    InvalidIccStream(&'static str),
    IccProfileEmbedded,
    InvalidEnumColorspace,
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
            Bitstream(err) => write!(f, "bitstream error: {}", err),
            Decoder(err) => write!(f, "entropy decoder error: {}", err),
            InvalidIccStream(s) => write!(f, "invalid ICC stream: {s}"),
            IccProfileEmbedded => write!(f, "embedded ICC profile is signalled, use it instead"),
            InvalidEnumColorspace => write!(f, "unknown colorspace without embedded ICC profile"),
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
        if let Error::Bitstream(e) | Error::Decoder(jxl_coding::Error::Bitstream(e)) = self {
            return e.unexpected_eof();
        }
        false
    }
}

pub type Result<T> = std::result::Result<T, Error>;
