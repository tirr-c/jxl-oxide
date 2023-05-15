use std::{
    fs::File,
    io::Read,
    path::Path,
};

use jxl_bitstream::{Bitstream, Bundle, ContainerDetectingReader};
use jxl_frame::Frame;
use jxl_grid::SimpleGrid;
use jxl_image::ImageHeader;
use jxl_render::{RenderContext, FrameBuffer};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Debug)]
pub struct JxlImage<R> {
    bitstream: Bitstream<ContainerDetectingReader<R>>,
    image_header: ImageHeader,
    icc: Vec<u8>,
}

impl<R: Read> JxlImage<R> {
    pub fn from_reader(reader: R) -> Result<Self> {
        let mut bitstream = Bitstream::new_detect(reader);
        let image_header = ImageHeader::parse(&mut bitstream, ())?;

        let icc = if image_header.metadata.colour_encoding.want_icc {
            tracing::debug!("Image has ICC profile");
            let icc = jxl_color::icc::read_icc(&mut bitstream)?;
            jxl_color::icc::decode_icc(&icc)?
        } else {
            jxl_color::icc::colour_encoding_to_icc(&image_header.metadata.colour_encoding)?
        };

        if image_header.metadata.preview.is_some() {
            tracing::debug!("Skipping preview frame");
            bitstream.zero_pad_to_byte()?;

            let frame = jxl_frame::Frame::parse(&mut bitstream, &image_header)?;
            let toc = frame.toc();
            let bookmark = toc.bookmark() + (toc.total_byte_size() * 8);
            bitstream.skip_to_bookmark(bookmark)?;
        }

        Ok(Self {
            bitstream,
            image_header,
            icc,
        })
    }

    #[inline]
    pub fn image_header(&self) -> &ImageHeader {
        &self.image_header
    }

    #[inline]
    pub fn desired_icc(&self) -> &[u8] {
        &self.icc
    }

    #[inline]
    pub fn renderer(&mut self) -> JxlRenderer<'_, R> {
        let ctx = RenderContext::new(&self.image_header);
        JxlRenderer {
            bitstream: &mut self.bitstream,
            icc: &self.icc,
            ctx,
            crop_region: None,
        }
    }
}

impl JxlImage<File> {
    #[inline]
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        Self::from_reader(file)
    }
}

#[derive(Debug)]
pub struct JxlRenderer<'img, R> {
    bitstream: &'img mut Bitstream<ContainerDetectingReader<R>>,
    icc: &'img [u8],
    ctx: RenderContext<'img>,
    crop_region: Option<CropInfo>,
}

impl<R: Read> JxlRenderer<'_, R> {
    #[inline]
    pub fn set_crop_region(&mut self, crop_region: Option<CropInfo>) -> &mut Self {
        self.crop_region = crop_region;
        self
    }

    #[inline]
    pub fn crop_region(&self) -> Option<CropInfo> {
        self.crop_region
    }

    #[inline]
    fn crop_region_flattened(&self) -> Option<(u32, u32, u32, u32)> {
        self.crop_region.map(|info| (info.left, info.top, info.width, info.height))
    }

    pub fn render_next_frame(&mut self) -> Result<RenderResult<'_>> {
        let region = self.crop_region_flattened();
        let result = self.ctx.load_until_keyframe(self.bitstream, false, region)?;
        match result {
            jxl_frame::ProgressiveResult::NeedMoreData => Ok(RenderResult::NeedMoreData),
            jxl_frame::ProgressiveResult::FrameComplete => {
                let keyframe_idx = self.ctx.loaded_keyframes() - 1;
                let mut fb = self.ctx.render_keyframe_cropped(keyframe_idx, region)?;
                todo!()
            },
            _ => unreachable!(),
        }
    }
}

#[derive(Debug)]
pub enum RenderResult<'f> {
    Done(Render<'f>),
    NeedMoreData,
    NoMoreFrames,
}

#[derive(Debug)]
pub struct Render<'f> {
    index: usize,
    frame: Frame<'f>,
    fb: FrameBuffer,
    extra_frames: Vec<SimpleGrid<f32>>,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct CropInfo {
    pub width: u32,
    pub height: u32,
    pub left: u32,
    pub top: u32,
}
