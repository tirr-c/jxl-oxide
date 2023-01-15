use std::io::Read;

use jxl_bitstream::{Bitstream, Bundle, header::Headers};
use jxl_frame::{Frame, header::FrameType};

mod error;
pub use error::{Error, Result};

#[derive(Debug)]
pub struct RenderContext<'a> {
    image_header: &'a Headers,
    frames: Vec<Frame<'a>>,
    lf_frame: Vec<usize>,
    reference: Vec<usize>,
}

impl<'a> RenderContext<'a> {
    pub fn new(image_header: &'a Headers) -> Self {
        Self {
            image_header,
            frames: Vec::new(),
            lf_frame: vec![usize::MAX; 4],
            reference: vec![usize::MAX; 4],
        }
    }

    fn metadata(&self) -> &'a jxl_bitstream::header::ImageMetadata {
        &self.image_header.metadata
    }

    fn xyb_encoded(&self) -> bool {
        self.image_header.metadata.xyb_encoded
    }

    fn preserve_frame(&mut self, frame: Frame<'a>) {
        let header = frame.header();
        let idx = self.frames.len();
        let is_last = header.is_last;

        if !is_last && (header.duration == 0 || header.save_as_reference != 0) && header.frame_type != FrameType::LfFrame {
            let ref_idx = header.save_as_reference as usize;
            self.reference[ref_idx] = idx;
        }
        if header.lf_level != 0 {
            let lf_idx = header.lf_level as usize - 1;
            self.lf_frame[lf_idx] = idx;
        }
        self.frames.push(frame);
    }
}

impl RenderContext<'_> {
    pub fn width(&self) -> u32 {
        self.image_header.size.width
    }

    pub fn height(&self) -> u32 {
        self.image_header.size.height
    }
}

impl RenderContext<'_> {
    #[cfg(feature = "mt")]
    pub fn load_all_frames<R: Read + Send>(&mut self, bitstream: &mut Bitstream<R>) -> Result<()> {
        let image_header = self.image_header;

        loop {
            bitstream.zero_pad_to_byte()?;
            let mut frame = Frame::parse(bitstream, image_header)?;
            let header = frame.header();
            let is_last = header.is_last;
            eprintln!("Decoding {}x{} frame (upsampling={}, lf_level={})", header.width, header.height, header.upsampling, header.lf_level);

            frame.load_all_par(bitstream)?;
            frame.complete()?;

            let toc = frame.toc();
            let bookmark = toc.bookmark() + (toc.total_byte_size() * 8);
            self.preserve_frame(frame);
            if is_last {
                break;
            }

            bitstream.skip_to_bookmark(bookmark)?;
        }
        Ok(())
    }

    #[cfg(not(feature = "mt"))]
    pub fn load_all_frames<R: Read>(&mut self, bitstream: &mut Bitstream<R>) -> Result<()> {
        let image_header = self.image_header;

        loop {
            bitstream.zero_pad_to_byte()?;
            let mut frame = Frame::parse(bitstream, image_header)?;
            let header = frame.header();
            let is_last = header.is_last;
            eprintln!("Decoding {}x{} frame (upsampling={}, lf_level={})", header.width, header.height, header.upsampling, header.lf_level);

            frame.load_all(bitstream)?;
            frame.complete()?;

            self.preserve_frame(frame);
            if is_last {
                break;
            }

            bitstream.skip_to_bookmark(bookmark)?;
        }
        Ok(())
    }

    pub fn tmp_rgba_be_interleaved<F>(&self, f: F) -> Result<()>
    where
        F: FnMut(&[u8]) -> jxl_frame::Result<()>,
    {
        let frame = self.frames.last().unwrap();
        frame.rgba_be_interleaved(f)?;
        Ok(())
    }
}
