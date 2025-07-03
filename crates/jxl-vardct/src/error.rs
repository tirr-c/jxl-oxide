#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    Bitstream(jxl_bitstream::Error),
    Decoder(jxl_coding::Error),
    Buffer(jxl_grid::OutOfMemory),
    Modular(jxl_modular::Error),
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

impl From<jxl_grid::OutOfMemory> for Error {
    fn from(err: jxl_grid::OutOfMemory) -> Self {
        Self::Buffer(err)
    }
}

impl From<jxl_modular::Error> for Error {
    fn from(err: jxl_modular::Error) -> Self {
        Self::Modular(err)
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use Error::*;

        match self {
            Bitstream(err) => write!(f, "bitstream error: {err}"),
            Decoder(err) => write!(f, "entropy decoder error: {err}"),
            Buffer(err) => write!(f, "{err}"),
            Modular(err) => write!(f, "modular stream error: {err}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        use Error::*;

        match self {
            Bitstream(err) => Some(err),
            Decoder(err) => Some(err),
            Buffer(err) => Some(err),
            Modular(err) => Some(err),
        }
    }
}

impl Error {
    pub fn unexpected_eof(&self) -> bool {
        match self {
            Error::Bitstream(e) => e.unexpected_eof(),
            Error::Decoder(e) => e.unexpected_eof(),
            Error::Modular(e) => e.unexpected_eof(),
            _ => false,
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;
