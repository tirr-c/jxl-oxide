pub mod box_header;

use box_header::*;

use crate::error::Error;

/// Wrapper that detects container format from underlying reader.
#[derive(Default)]
pub struct ContainerDetectingReader {
    state: DetectState,
    buf: Vec<u8>,
    codestream: Vec<u8>,
    aux_boxes: Vec<(ContainerBoxType, Vec<u8>)>,
    jxlp_index_state: JxlpIndexState,
}

impl std::fmt::Debug for ContainerDetectingReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContainerDetectingReader")
            .field("state", &self.state)
            .field("jxlp_index_state", &self.jxlp_index_state)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Default)]
enum DetectState {
    #[default]
    WaitingSignature,
    WaitingBoxHeader,
    WaitingJxlpIndex(ContainerBoxHeader),
    InAuxBox {
        header: ContainerBoxHeader,
        data: Vec<u8>,
        bytes_left: Option<usize>,
    },
    InCodestream {
        kind: BitstreamKind,
        bytes_left: Option<usize>,
    },
    Done(BitstreamKind),
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

struct ConcatSlice<'first, 'second> {
    slices: (&'first [u8], &'second [u8]),
    ptr: usize,
}

impl<'first, 'second> ConcatSlice<'first, 'second> {
    fn new(slice0: &'first [u8], slice1: &'second [u8]) -> Self {
        Self {
            slices: (slice0, slice1),
            ptr: 0,
        }
    }

    fn len(&self) -> usize {
        self.slices.0.len() + self.slices.1.len()
    }

    fn remaining_slices(&self) -> (&'first [u8], &'second [u8]) {
        let (slice0, slice1) = self.slices;
        let total_len = self.len();
        let ptr = self.ptr;
        if ptr >= total_len {
            (&[], &[])
        } else if let Some(second_slice_ptr) = ptr.checked_sub(slice0.len()) {
            (&[], &slice1[second_slice_ptr..])
        } else {
            (&slice0[ptr..], slice1)
        }
    }

    fn advance(&mut self, bytes: usize) {
        self.ptr += bytes;
    }

    fn peek<'out>(&self, out_buf: &'out mut [u8]) -> &'out mut [u8] {
        let (slice0, slice1) = self.remaining_slices();
        let total_len = slice0.len() + slice1.len();

        let out_bytes = out_buf.len().min(total_len);
        let out_buf = &mut out_buf[..out_bytes];

        if out_bytes <= slice0.len() {
            out_buf.copy_from_slice(&slice0[..out_bytes]);
        } else {
            let (out_first, out_second) = out_buf.split_at_mut(slice0.len());
            out_first.copy_from_slice(slice0);
            out_second.copy_from_slice(&slice1[..out_second.len()]);
        }

        out_buf
    }

    fn fill_vec(&mut self, max_bytes: Option<usize>, v: &mut Vec<u8>) -> usize {
        let (slice0, slice1) = self.remaining_slices();
        let total_len = slice0.len() + slice1.len();

        let out_bytes = max_bytes.unwrap_or(usize::MAX).min(total_len);
        v.reserve(out_bytes);

        if out_bytes <= slice0.len() {
            v.extend_from_slice(&slice0[..out_bytes]);
        } else {
            let second_slice_len = out_bytes - slice0.len();
            v.extend_from_slice(slice0);
            v.extend_from_slice(&slice1[..second_slice_len]);
        }

        self.advance(out_bytes);
        out_bytes
    }
}

impl ContainerDetectingReader {
    const CODESTREAM_SIG: [u8; 2] = [0xff, 0x0a];
    const CONTAINER_SIG: [u8; 12] = [0, 0, 0, 0xc, b'J', b'X', b'L', b' ', 0xd, 0xa, 0x87, 0xa];

    pub fn new() -> Self {
        Self::default()
    }

    pub fn kind(&self) -> BitstreamKind {
        match self.state {
            DetectState::WaitingSignature => BitstreamKind::Unknown,
            DetectState::WaitingBoxHeader
            | DetectState::WaitingJxlpIndex(..)
            | DetectState::InAuxBox { .. } => BitstreamKind::Container,
            DetectState::InCodestream { kind, .. } | DetectState::Done(kind) => kind,
        }
    }

    pub fn feed_bytes(&mut self, input: &[u8]) -> Result<(), Error> {
        let state = &mut self.state;
        let mut reader = ConcatSlice::new(&self.buf, input);

        loop {
            match state {
                DetectState::WaitingSignature => {
                    let mut signature_buf = [0u8; 12];
                    let buf = reader.peek(&mut signature_buf);
                    if buf.starts_with(&Self::CODESTREAM_SIG) {
                        tracing::trace!("Codestream signature found");
                        *state = DetectState::InCodestream {
                            kind: BitstreamKind::BareCodestream,
                            bytes_left: None,
                        };
                    } else if buf.starts_with(&Self::CONTAINER_SIG) {
                        tracing::trace!("Container signature found");
                        *state = DetectState::WaitingBoxHeader;
                        reader.advance(Self::CONTAINER_SIG.len());
                    } else if !Self::CODESTREAM_SIG.starts_with(buf)
                        && !Self::CONTAINER_SIG.starts_with(buf)
                    {
                        tracing::debug!(?buf, "Invalid signature");
                        *state = DetectState::InCodestream {
                            kind: BitstreamKind::Invalid,
                            bytes_left: None,
                        };
                    } else {
                        break;
                    }
                }
                DetectState::WaitingBoxHeader => match ContainerBoxHeader::parse(&reader)? {
                    HeaderParseResult::Done {
                        header,
                        header_size: size,
                    } => {
                        reader.advance(size);
                        let tbox = header.box_type();
                        if tbox == ContainerBoxType::CODESTREAM {
                            match self.jxlp_index_state {
                                JxlpIndexState::Initial => {
                                    self.jxlp_index_state = JxlpIndexState::SingleJxlc;
                                }
                                JxlpIndexState::SingleJxlc => {
                                    tracing::debug!("Duplicate jxlc box found");
                                    return Err(Error::InvalidBox);
                                }
                                JxlpIndexState::Jxlp(_) | JxlpIndexState::JxlpFinished => {
                                    tracing::debug!("Found jxlc box instead of jxlp box");
                                    return Err(Error::InvalidBox);
                                }
                            }

                            *state = DetectState::InCodestream {
                                kind: BitstreamKind::Container,
                                bytes_left: header.box_size().map(|x| x as usize),
                            };
                        } else if tbox == ContainerBoxType::PARTIAL_CODESTREAM {
                            if let Some(box_size) = header.box_size() {
                                if box_size < 4 {
                                    return Err(Error::InvalidBox);
                                }
                            }

                            match &mut self.jxlp_index_state {
                                JxlpIndexState::Initial => {
                                    self.jxlp_index_state = JxlpIndexState::Jxlp(0);
                                }
                                JxlpIndexState::Jxlp(index) => {
                                    *index += 1;
                                }
                                JxlpIndexState::SingleJxlc => {
                                    tracing::debug!("jxlp box found after jxlc box");
                                    return Err(Error::InvalidBox);
                                }
                                JxlpIndexState::JxlpFinished => {
                                    tracing::debug!("found another jxlp box after the final one");
                                    return Err(Error::InvalidBox);
                                }
                            }

                            *state = DetectState::WaitingJxlpIndex(header);
                        } else {
                            let bytes_left = header.box_size().map(|x| x as usize);
                            *state = DetectState::InAuxBox {
                                header,
                                data: Vec::new(),
                                bytes_left,
                            };
                        }
                    }
                    HeaderParseResult::NeedMoreData => break,
                },
                DetectState::WaitingJxlpIndex(header) => {
                    let mut buf = [0u8; 4];
                    reader.peek(&mut buf);
                    if buf.len() < 4 {
                        break;
                    }

                    let index = u32::from_be_bytes(buf);
                    reader.advance(4);
                    let is_last = index & 0x80000000 != 0;
                    let index = index & 0x7fffffff;

                    match self.jxlp_index_state {
                        JxlpIndexState::Jxlp(expected_index) if expected_index == index => {
                            if is_last {
                                self.jxlp_index_state = JxlpIndexState::JxlpFinished;
                            }
                        }
                        JxlpIndexState::Jxlp(expected_index) => {
                            tracing::debug!(
                                expected_index,
                                actual_index = index,
                                "Out-of-order jxlp box found",
                            );
                            return Err(Error::InvalidBox);
                        }
                        state => {
                            tracing::debug!(?state, "invalid jxlp index state in WaitingJxlpIndex");
                            unreachable!("invalid jxlp index state in WaitingJxlpIndex");
                        }
                    }

                    *state = DetectState::InCodestream {
                        kind: BitstreamKind::Container,
                        bytes_left: header.box_size().map(|x| x as usize - 4),
                    };
                }
                DetectState::InCodestream {
                    bytes_left: None, ..
                } => {
                    reader.fill_vec(None, &mut self.codestream);
                    break;
                }
                DetectState::InCodestream {
                    bytes_left: Some(bytes_left),
                    ..
                } => {
                    let bytes_written = reader.fill_vec(Some(*bytes_left), &mut self.codestream);
                    *bytes_left -= bytes_written;
                    if *bytes_left == 0 {
                        *state = DetectState::WaitingBoxHeader;
                    } else {
                        break;
                    }
                }
                DetectState::InAuxBox {
                    data,
                    bytes_left: None,
                    ..
                } => {
                    reader.fill_vec(None, data);
                    break;
                }
                DetectState::InAuxBox {
                    header,
                    data,
                    bytes_left: Some(bytes_left),
                } => {
                    let bytes_written = reader.fill_vec(Some(*bytes_left), data);
                    *bytes_left -= bytes_written;
                    if *bytes_left == 0 {
                        self.aux_boxes
                            .push((header.box_type(), std::mem::take(data)));
                        *state = DetectState::WaitingBoxHeader;
                    } else {
                        break;
                    }
                }
                DetectState::Done(_) => break,
            }
        }

        let (buf_slice, input_slice) = reader.remaining_slices();
        if buf_slice.is_empty() {
            self.buf.clear();
        } else {
            let remaining_buf_from = self.buf.len() - buf_slice.len();
            self.buf.drain(..remaining_buf_from);
        }
        self.buf.reserve(input_slice.len());
        self.buf.extend_from_slice(input_slice);
        Ok(())
    }

    pub fn take_bytes(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.codestream)
    }

    pub fn finish(&mut self) {
        if let DetectState::InAuxBox { header, data, .. } = &mut self.state {
            self.aux_boxes
                .push((header.box_type(), std::mem::take(data)));
        }
        self.state = DetectState::Done(self.kind());
    }
}
