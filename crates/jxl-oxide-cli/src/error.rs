#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    ReadJxl(Box<dyn std::error::Error + Send + Sync + 'static>),
    ReadIcc(std::io::Error),
    WriteIcc(std::io::Error),
    WriteImage(std::io::Error),
    Render(Box<dyn std::error::Error + Send + Sync + 'static>),
    Reconstruct(Box<dyn std::error::Error + Send + Sync + 'static>),
    #[cfg(feature = "__ffmpeg")]
    Ffmpeg {
        msg: Option<&'static str>,
        averror: Option<std::ffi::c_int>,
    },
}

impl Error {
    #[cfg(feature = "__ffmpeg")]
    #[inline]
    pub(crate) fn from_averror(averror: std::ffi::c_int) -> Self {
        Self::Ffmpeg {
            msg: None,
            averror: Some(averror),
        }
    }

    #[cfg(feature = "__ffmpeg")]
    #[inline]
    pub(crate) fn from_ffmpeg_msg(msg: &'static str) -> Self {
        Self::Ffmpeg {
            msg: Some(msg),
            averror: None,
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::ReadJxl(e) => write!(f, "failed reading JPEG XL image: {e}"),
            Error::ReadIcc(e) => write!(f, "failed reading ICC profile: {e}"),
            Error::WriteIcc(e) => write!(f, "failed writing ICC profile: {e}"),
            Error::WriteImage(e) => write!(f, "failed writing output image: {e}"),
            Error::Render(e) => write!(f, "failed to render image: {e}"),
            Error::Reconstruct(e) => write!(f, "failed to reconstruct: {e}"),
            #[cfg(feature = "__ffmpeg")]
            Error::Ffmpeg { msg, averror } => {
                write!(f, "FFmpeg error")?;
                let e = averror.map(rusty_ffmpeg::ffi::av_err2str);
                match (msg, e) {
                    (None, None) => {}
                    (Some(msg), None) => {
                        write!(f, ": {msg}")?;
                    }
                    (None, Some(e)) => {
                        write!(f, ": {e}")?;
                    }
                    (Some(msg), Some(e)) => {
                        write!(f, ": {msg} ({e})")?;
                    }
                }

                Ok(())
            }
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::ReadJxl(e) => Some(&**e),
            Error::ReadIcc(e) => Some(e),
            Error::WriteIcc(e) => Some(e),
            Error::WriteImage(e) => Some(e),
            Error::Render(e) => Some(&**e),
            Error::Reconstruct(e) => Some(&**e),
            #[cfg(feature = "__ffmpeg")]
            Error::Ffmpeg { .. } => None,
        }
    }
}
