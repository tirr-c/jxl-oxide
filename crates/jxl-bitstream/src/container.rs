pub mod box_header;
pub mod parse;

use box_header::*;
pub use parse::ParseEvent;
use parse::*;

/// Wrapper that detects container format from underlying reader.
#[derive(Debug, Default)]
pub struct ContainerDetectingReader {
    state: DetectState,
    jxlp_index_state: JxlpIndexState,
    previous_consumed_bytes: usize,
}

#[derive(Debug, Default)]
enum DetectState {
    #[default]
    WaitingSignature,
    WaitingBoxHeader,
    WaitingJxlpIndex(ContainerBoxHeader),
    InAuxBox {
        #[allow(unused)]
        header: ContainerBoxHeader,
        bytes_left: Option<usize>,
    },
    InCodestream {
        kind: BitstreamKind,
        bytes_left: Option<usize>,
    },
}

/// Structure of the decoded bitstream.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum BitstreamKind {
    /// Decoder can't determine structure of the bitstream.
    Unknown,
    /// Bitstream is a direct JPEG XL codestream without box structure.
    BareCodestream,
    /// Bitstream is a JPEG XL container with box structure.
    Container,
    /// Bitstream is not a valid JPEG XL image.
    Invalid,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Default)]
enum JxlpIndexState {
    #[default]
    Initial,
    SingleJxlc,
    Jxlp(u32),
    JxlpFinished,
}

impl ContainerDetectingReader {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn kind(&self) -> BitstreamKind {
        match self.state {
            DetectState::WaitingSignature => BitstreamKind::Unknown,
            DetectState::WaitingBoxHeader
            | DetectState::WaitingJxlpIndex(..)
            | DetectState::InAuxBox { .. } => BitstreamKind::Container,
            DetectState::InCodestream { kind, .. } => kind,
        }
    }

    /// Feeds bytes to the parser, and receives parser events.
    ///
    /// The parser might not consume all of the buffer. Use [`previous_consumed_bytes`] to get how
    /// many bytes are consumed. Bytes not consumed by the parser should be fed into the parser
    /// again.
    ///
    /// [`previous_consumed_bytes`]: ContainerDetectingReader::previous_consumed_bytes
    pub fn feed_bytes<'inner, 'buf>(
        &'inner mut self,
        input: &'buf [u8],
    ) -> ParseEvents<'inner, 'buf> {
        ParseEvents::new(self, input)
    }

    /// Get how many bytes are consumed by the previous call to [`feed_bytes`].
    ///
    /// Bytes not consumed by the parser should be fed into the parser again.
    ///
    /// [`feed_bytes`]: ContainerDetectingReader::feed_bytes
    pub fn previous_consumed_bytes(&self) -> usize {
        self.previous_consumed_bytes
    }
}
