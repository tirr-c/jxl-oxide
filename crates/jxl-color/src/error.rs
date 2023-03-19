#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    Bitstream(jxl_bitstream::Error),
    Decoder(jxl_coding::Error),
    InvalidIccStream(&'static str),
    #[cfg(feature = "icc")]
    Cms(lcms2::Error),
    #[cfg(feature = "icc")]
    IccProfileEmbedded,
    #[cfg(feature = "icc")]
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

#[cfg(feature = "icc")]
impl From<lcms2::Error> for Error {
    fn from(err: lcms2::Error) -> Self {
        Self::Cms(err)
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use Error::*;

        match self {
            Bitstream(err) => write!(f, "bitstream error: {}", err),
            Decoder(err) => write!(f, "entropy decoder error: {}", err),
            InvalidIccStream(s) => write!(f, "invalid ICC stream: {s}"),
            #[cfg(feature = "icc")]
            Cms(err) => write!(f, "LCMS2 error: {err}"),
            #[cfg(feature = "icc")]
            IccProfileEmbedded => write!(f, "embedded ICC profile is signalled, use it instead"),
            #[cfg(feature = "icc")]
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
            #[cfg(feature = "icc")]
            Cms(err) => Some(err),
            _ => None,
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;
