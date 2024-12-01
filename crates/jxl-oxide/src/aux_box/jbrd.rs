use jxl_jbr::JpegBitstreamData;

pub enum Jbrd {
    Uninit(Vec<u8>),
    Init(JpegBitstreamData),
}

impl std::fmt::Debug for Jbrd {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Uninit(_) => write!(f, "Uninit(_)"),
            Self::Init(data) => f.debug_tuple("Init").field(data).finish(),
        }
    }
}

impl Jbrd {
    pub fn new() -> Self {
        Self::Uninit(Vec::new())
    }

    pub fn feed_bytes(&mut self, bytes: &[u8]) -> crate::Result<()> {
        match self {
            Self::Uninit(buf) => {
                buf.extend_from_slice(bytes);
                if let Some(data) = JpegBitstreamData::try_parse(buf)? {
                    *self = Self::Init(data);
                }
            }
            Self::Init(data) => {
                data.feed_bytes(bytes)?;
            }
        }

        Ok(())
    }

    pub fn finalize(&mut self) -> crate::Result<()> {
        match self {
            Jbrd::Uninit(_) => Err(jxl_jbr::Error::InvalidData.into()),
            Jbrd::Init(data) => data.finalize().map_err(From::from),
        }
    }

    pub fn data(&self) -> Option<&JpegBitstreamData> {
        match self {
            Self::Init(data) => Some(data),
            _ => None,
        }
    }
}
