use std::io::Write;

use brotli_decompressor::DecompressorWriter;

use crate::Result;

mod exif;
mod jbrd;

pub use exif::*;
pub use jbrd::*;

#[derive(Debug, Default)]
pub struct AuxBoxReader {
    data: DataKind,
    done: bool,
}

#[derive(Default)]
enum DataKind {
    #[default]
    Init,
    NoData,
    Raw(Vec<u8>),
    Brotli(Box<DecompressorWriter<Vec<u8>>>),
}

impl std::fmt::Debug for DataKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Init => write!(f, "Init"),
            Self::NoData => write!(f, "NoData"),
            Self::Raw(buf) => f
                .debug_tuple("Raw")
                .field(&format_args!("{} byte(s)", buf.len()))
                .finish(),
            Self::Brotli(_) => f.debug_tuple("Brotli").finish(),
        }
    }
}

impl AuxBoxReader {
    pub(super) fn new() -> Self {
        Self::default()
    }

    pub(super) fn ensure_brotli(&mut self) -> Result<()> {
        if self.done {
            return Ok(());
        }

        match self.data {
            DataKind::Init => {
                let writer = DecompressorWriter::new(Vec::<u8>::new(), 4096);
                self.data = DataKind::Brotli(Box::new(writer));
            }
            DataKind::NoData | DataKind::Raw(_) => {
                panic!();
            }
            DataKind::Brotli(_) => {}
        }
        Ok(())
    }
}

impl AuxBoxReader {
    pub fn feed_data(&mut self, data: &[u8]) -> Result<()> {
        if self.done {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "cannot feed into finalized box",
            )
            .into());
        }

        match self.data {
            DataKind::Init => {
                self.data = DataKind::Raw(data.to_vec());
            }
            DataKind::NoData => {
                unreachable!();
            }
            DataKind::Raw(ref mut buf) => {
                buf.extend_from_slice(data);
            }
            DataKind::Brotli(ref mut writer) => {
                writer.write_all(data)?;
            }
        }
        Ok(())
    }

    pub fn finalize(&mut self) -> Result<()> {
        if self.done {
            return Ok(());
        }

        if let DataKind::Brotli(ref mut writer) = self.data {
            writer.flush()?;
            writer.close()?;
        }

        match std::mem::replace(&mut self.data, DataKind::NoData) {
            DataKind::Init | DataKind::NoData => {}
            DataKind::Raw(buf) => self.data = DataKind::Raw(buf),
            DataKind::Brotli(writer) => {
                let inner = writer.into_inner().inspect_err(|_| {
                    tracing::warn!("Brotli decompressor reported an error");
                });
                let buf = inner.unwrap_or_else(|buf| buf);
                self.data = DataKind::Raw(buf);
            }
        }

        self.done = true;
        Ok(())
    }
}

impl AuxBoxReader {
    pub fn is_done(&self) -> bool {
        self.done
    }

    pub fn data(&self) -> AuxBoxData {
        if !self.is_done() {
            return AuxBoxData::Decoding;
        }

        match &self.data {
            DataKind::Init | DataKind::Brotli(_) => AuxBoxData::Decoding,
            DataKind::NoData => AuxBoxData::NotFound,
            DataKind::Raw(buf) => AuxBoxData::Data(buf),
        }
    }
}

pub enum AuxBoxData<'data> {
    Data(&'data [u8]),
    Decoding,
    NotFound,
}

impl std::fmt::Debug for AuxBoxData<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Data(buf) => f
                .debug_tuple("Data")
                .field(&format_args!("{} byte(s)", buf.len()))
                .finish(),
            Self::Decoding => write!(f, "Decoding"),
            Self::NotFound => write!(f, "NotFound"),
        }
    }
}
