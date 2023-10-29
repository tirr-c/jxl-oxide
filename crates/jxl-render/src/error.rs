#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    Bitstream(jxl_bitstream::Error),
    Decoder(jxl_coding::Error),
    Frame(jxl_frame::Error),
    Color(jxl_color::Error),
    IncompleteFrame,
    UninitializedLfFrame(u32),
    InvalidReference(u32),
    NotReady,
    NotSupported(&'static str),
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
            UninitializedLfFrame(lf_level) => write!(f, "uninitialized LF frame for level {lf_level}"),
            InvalidReference(idx) => write!(f, "invalid reference {idx}"),
            NotReady => write!(f, "image is not ready to be rendered"),
            NotSupported(msg) => write!(f, "not supported: {}", msg),
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

impl Error {
    pub fn unexpected_eof(&self) -> bool {
        match self {
            Error::Bitstream(e) => e.unexpected_eof(),
            Error::Decoder(e) => e.unexpected_eof(),
            Error::Frame(e) => e.unexpected_eof(),
            Error::Color(e) => e.unexpected_eof(),
            _ => false,
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;
