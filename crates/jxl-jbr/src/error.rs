#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    Bitstream(jxl_bitstream::Error),
    Brotli(std::io::Error),
    InvalidData,
    HuffmanLookup,
    ReconstructionWrite(std::io::Error),
    // TODO: Move to jxl-oxide crate
    ReconstructionUnavailable,
    ReconstructionDataIncomplete,
    FrameDataIncomplete,
    FrameParse(jxl_frame::Error),
    IncompatibleFrame,
}

impl From<jxl_bitstream::Error> for Error {
    fn from(value: jxl_bitstream::Error) -> Self {
        Self::Bitstream(value)
    }
}

impl From<jxl_frame::Error> for Error {
    fn from(value: jxl_frame::Error) -> Self {
        Self::FrameParse(value)
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Bitstream(e) => write!(f, "failed to read reconstruction data: {e}"),
            Error::Brotli(e) => write!(f, "failed to decompress Brotli stream: {e}"),
            Error::InvalidData => write!(f, "invalid reconstruction data"),
            Error::HuffmanLookup => {
                write!(f, "invalid reconstruction data: Huffman code lookup failed")
            }
            Error::ReconstructionWrite(e) => write!(f, "failed to write data: {e}"),
            Error::ReconstructionUnavailable => write!(f, "reconstruction data not found"),
            Error::ReconstructionDataIncomplete => write!(f, "reconstruction data is incomplete"),
            Error::FrameDataIncomplete => write!(f, "JPEG XL frame data is incomplete"),
            Error::FrameParse(e) => write!(f, "error parsing JPEG XL frame: {e}"),
            Error::IncompatibleFrame => write!(
                f,
                "JPEG XL frame data is incompatible with reconstruction data"
            ),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Bitstream(e) => Some(e),
            Error::Brotli(e) => Some(e),
            Error::ReconstructionWrite(e) => Some(e),
            Error::FrameParse(e) => Some(e),
            _ => None,
        }
    }
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
