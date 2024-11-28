#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    Bitstream(jxl_bitstream::Error),
    Decoder(jxl_coding::Error),
    Buffer(jxl_grid::Error),
    Modular(jxl_modular::Error),
    VarDct(jxl_vardct::Error),
    InvalidTocPermutation,
    IncompleteFrameData { field: &'static str },
    OutOfMemory,
    HadError,
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

impl From<jxl_grid::Error> for Error {
    fn from(err: jxl_grid::Error) -> Self {
        Self::Buffer(err)
    }
}

impl From<jxl_modular::Error> for Error {
    fn from(err: jxl_modular::Error) -> Self {
        Self::Modular(err)
    }
}

impl From<jxl_vardct::Error> for Error {
    fn from(err: jxl_vardct::Error) -> Self {
        Self::VarDct(err)
    }
}

impl From<std::collections::TryReserveError> for Error {
    fn from(_: std::collections::TryReserveError) -> Self {
        Self::OutOfMemory
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bitstream(err) => write!(f, "bitstream error: {}", err),
            Self::Decoder(err) => write!(f, "entropy decoder error: {}", err),
            Self::Buffer(err) => write!(f, "{}", err),
            Self::Modular(err) => write!(f, "modular stream error: {}", err),
            Self::VarDct(err) => write!(f, "vardct error: {}", err),
            Self::InvalidTocPermutation => write!(f, "invalid TOC permutation"),
            Self::IncompleteFrameData { field } => {
                write!(f, "incomplete frame data: {} is missing", field)
            }
            Self::OutOfMemory => write!(f, "out of memory"),
            Self::HadError => write!(f, "previous parsing errored"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Bitstream(err) => Some(err),
            Self::Decoder(err) => Some(err),
            Self::Buffer(err) => Some(err),
            Self::Modular(err) => Some(err),
            Self::VarDct(err) => Some(err),
            _ => None,
        }
    }
}

impl Error {
    /// Returns whether the error is caused by the unexpected EOF of the bitstream.
    pub fn unexpected_eof(&self) -> bool {
        let bitstream_err = match self {
            Self::Bitstream(b)
            | Self::Decoder(jxl_coding::Error::Bitstream(b))
            | Self::Modular(jxl_modular::Error::Decoder(jxl_coding::Error::Bitstream(b)))
            | Self::Modular(jxl_modular::Error::Bitstream(b))
            | Self::VarDct(jxl_vardct::Error::Bitstream(b))
            | Self::VarDct(jxl_vardct::Error::Decoder(jxl_coding::Error::Bitstream(b)))
            | Self::VarDct(jxl_vardct::Error::Modular(jxl_modular::Error::Bitstream(b)))
            | Self::VarDct(jxl_vardct::Error::Modular(jxl_modular::Error::Decoder(
                jxl_coding::Error::Bitstream(b),
            ))) => b,
            _ => return false,
        };
        if let jxl_bitstream::Error::Io(e) = bitstream_err {
            e.kind() == std::io::ErrorKind::UnexpectedEof
        } else {
            false
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;
