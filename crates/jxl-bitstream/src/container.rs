//! Types for JPEG XL container format.

mod box_header;
mod parse;

pub use box_header::*;
pub use parse::*;

/// Parser that detects the kind of bitstream and emits parser events.
#[derive(Debug, Default)]
pub struct ContainerParser {
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
        header: ContainerBoxHeader,
        brotli_box_type: Option<ContainerBoxType>,
        bytes_left: Option<usize>,
    },
    InCodestream {
        kind: BitstreamKind,
        bytes_left: Option<usize>,
        pending_no_more_aux_box: bool,
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

impl ContainerParser {
    /// Creates a new parser.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the kind of this bitstream currently recognized.
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
    /// [`previous_consumed_bytes`]: Self::previous_consumed_bytes
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
    /// [`feed_bytes`]: Self::feed_bytes
    pub fn previous_consumed_bytes(&self) -> usize {
        self.previous_consumed_bytes
    }
}
