#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    NonZeroPadding,
    InvalidFloat,
    InvalidEnum {
        name: &'static str,
        value: u32,
    },
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => {
                write!(f, "I/O error: {}", e)
            },
            Self::NonZeroPadding => {
                write!(f, "PadZeroToByte() read non-zero bits")
            },
            Self::InvalidFloat => {
                write!(f, "F16() read NaN or Infinity")
            },
            Self::InvalidEnum { name, value } => {
                write!(f, "Enum({}) read invalid enum value of {}", name, value)
            },
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

pub type Result<T> = std::result::Result<T, Error>;
