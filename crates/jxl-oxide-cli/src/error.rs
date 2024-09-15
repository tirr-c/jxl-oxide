#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    ReadJxl(Box<dyn std::error::Error + Send + Sync + 'static>),
    ReadIcc(std::io::Error),
    WriteIcc(std::io::Error),
    WriteImage(std::io::Error),
    Render(Box<dyn std::error::Error + Send + Sync + 'static>),
    #[cfg(feature = "__ffmpeg")]
    Ffmpeg(std::ffi::c_int),
    #[cfg(feature = "__ffmpeg")]
    WriteVideo(&'static str),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::ReadJxl(e) => write!(f, "failed reading JPEG XL image: {e}"),
            Error::ReadIcc(e) => write!(f, "failed reading ICC profile: {e}"),
            Error::WriteIcc(e) => write!(f, "failed writing ICC profile: {e}"),
            Error::WriteImage(e) => write!(f, "failed writing output image: {e}"),
            Error::Render(e) => write!(f, "failed to render image: {e}"),
            #[cfg(feature = "__ffmpeg")]
            Error::Ffmpeg(averror) => {
                let e = rusty_ffmpeg::ffi::av_err2str(*averror);
                write!(f, "FFmpeg error: {e}")
            }
            #[cfg(feature = "__ffmpeg")]
            Error::WriteVideo(e) => write!(f, "failed to write video: {e}"),
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
            #[cfg(feature = "__ffmpeg")]
            Error::Ffmpeg(_) | Error::WriteVideo(_) => None,
        }
    }
}
