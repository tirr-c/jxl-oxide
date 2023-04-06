use std::io::prelude::*;

use super::container::*;

pub struct ContainerDetectingReader<R> {
    detected: bool,
    buffer: std::io::Cursor<Vec<u8>>,
    box_header: Option<(ContainerBoxHeader, u64)>,
    aux_boxes: Vec<(ContainerBoxType, Vec<u8>)>,
    inner: R,
}

impl<R> ContainerDetectingReader<R> {
    pub fn new(reader: R) -> Self {
        Self {
            detected: false,
            buffer: std::io::Cursor::new(Vec::new()),
            box_header: None,
            aux_boxes: Vec::new(),
            inner: reader,
        }
    }
}

impl<R: Read> ContainerDetectingReader<R> {
    fn find_next_codestream(&mut self) -> std::io::Result<bool> {
        loop {
            let header = ContainerBoxHeader::parse(&mut self.inner)?;
            tracing::trace!(header = format_args!("{:?}", header));
            let box_type = header.box_type();
            if box_type == ContainerBoxType::CODESTREAM {
                self.box_header = Some((header, 0));
                return Ok(true);
            }
            if box_type == ContainerBoxType::PARTIAL_CODESTREAM {
                let mut index = [0u8; 4];
                self.inner.read_exact(&mut index)?;
                let index = u32::from_be_bytes(index);
                tracing::trace!(index = index & 0x7fffffff, is_last = index & 0x80000000 != 0);
                // TODO: check index
                self.box_header = Some((header, 4));
                return Ok(true);
            }

            let size = header.size();
            let buf = if let Some(size) = size {
                let mut buf = vec![0u8; size as usize];
                self.inner.read_exact(&mut buf)?;
                buf
            } else {
                let mut buf = Vec::new();
                self.inner.read_to_end(&mut buf)?;
                buf
            };
            self.aux_boxes.push((box_type, buf));
            if header.is_last() {
                return Ok(false);
            }
        }
    }
}

impl<R: Read> Read for ContainerDetectingReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if !self.detected {
            let mut buffer = vec![0u8; 2];
            self.detected = true;
            self.inner.read_exact(&mut buffer)?;
            if *buffer != [0, 0] {
                if *buffer != [0xff, 0xa] {
                    tracing::error!("Invalid codestream signature");
                } else {
                    tracing::debug!("Codestream signature found");
                }
                self.buffer = std::io::Cursor::new(buffer);
                return self.buffer.read(buf);
            }
            buffer.resize(12, 0);
            self.inner.read_exact(&mut buffer[2..])?;
            if *buffer != [0, 0, 0, 0xc, b'J', b'X', b'L', b' ', 0xd, 0xa, 0x87, 0xa] {
                tracing::error!("Invalid container signature");
                self.buffer = std::io::Cursor::new(buffer);
                return self.buffer.read(buf);
            }

            tracing::debug!("Container signature found");

            let found = self.find_next_codestream()?;
            if !found {
                return Ok(0);
            }
        }

        let count = self.buffer.read(buf)?;
        if count > 0 {
            return Ok(count);
        }

        if let Some((header, offset)) = &mut self.box_header {
            let size = header.size();
            if let Some(size) = size {
                let max_len = size - *offset;
                if max_len == 0 {
                    if header.is_last() {
                        return Ok(0);
                    }
                    let found = self.find_next_codestream()?;
                    if !found {
                        return Ok(0);
                    }
                    return self.read(buf);
                }

                let read_len = buf.len().min(max_len as usize);
                let count = self.inner.read(&mut buf[..read_len])?;
                *offset += count as u64;
                Ok(count)
            } else {
                self.inner.read(buf)
            }
        } else {
            self.inner.read(buf)
        }
    }
}
