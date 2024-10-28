use std::io::prelude::*;

use image::error::{DecodingError, ImageFormatHint};
use image::{ColorType, ImageError, ImageResult};
use jxl_grid::AllocTracker;

use crate::{InitializeResult, JxlImage};

pub struct JxlDecoder<R> {
    reader: R,
    image: JxlImage,
    current_memory_limit: usize,
    buf: Vec<u8>,
    buf_valid: usize,
}

impl<R: Read> JxlDecoder<R> {
    fn init_image(
        builder: crate::JxlImageBuilder,
        reader: &mut R,
        buf: &mut [u8],
        buf_valid: &mut usize,
    ) -> crate::Result<JxlImage> {
        let mut uninit = builder.build_uninit();

        let image = loop {
            let count = reader.read(&mut buf[*buf_valid..])?;
            if count == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "reader ended before parsing image header",
                )
                .into());
            }
            *buf_valid += count;
            let consumed = uninit.feed_bytes(&buf[..*buf_valid])?;
            buf.copy_within(consumed..*buf_valid, 0);
            *buf_valid -= consumed;

            match uninit.try_init()? {
                InitializeResult::NeedMoreData(x) => {
                    uninit = x;
                }
                InitializeResult::Initialized(x) => {
                    break x;
                }
            }
        };

        Ok(image)
    }

    pub fn with_default_threadpool(mut reader: R) -> ImageResult<Self> {
        let current_memory_limit = usize::MAX;
        let builder =
            JxlImage::builder().alloc_tracker(AllocTracker::with_limit(current_memory_limit));

        let mut buf = vec![0u8; 4096];
        let mut buf_valid = 0usize;
        let image = Self::init_image(builder, &mut reader, &mut buf, &mut buf_valid)
            .map_err(|e| ImageError::Decoding(DecodingError::new(ImageFormatHint::Unknown, e)))?;

        let mut decoder = Self {
            reader,
            image,
            current_memory_limit,
            buf,
            buf_valid,
        };

        decoder.load_until_first_keyframe().map_err(|e| {
            ImageError::Decoding(DecodingError::new(
                ImageFormatHint::PathExtension("jxl".into()),
                e,
            ))
        })?;

        // Convert CMYK to sRGB
        if decoder.image.pixel_format().has_black() {
            decoder
                .image
                .request_color_encoding(jxl_color::EnumColourEncoding::srgb(
                    jxl_color::RenderingIntent::Relative,
                ));
        }

        Ok(decoder)
    }

    fn load_until_first_keyframe(&mut self) -> crate::Result<()> {
        while self.image.ctx.loaded_keyframes() == 0 {
            let count = self.reader.read(&mut self.buf[self.buf_valid..])?;
            if count == 0 {
                break;
            }
            self.buf_valid += count;
            let consumed = self.image.feed_bytes(&self.buf[..self.buf_valid])?;
            self.buf.copy_within(consumed..self.buf_valid, 0);
            self.buf_valid -= consumed;
        }

        if self.image.frame_by_keyframe(0).is_none() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "reader ended before parsing first frame",
            )
            .into());
        }

        Ok(())
    }

    fn read_image_inner(&mut self, buf: &mut [u8]) -> crate::Result<()> {
        let render = if self.image.num_loaded_keyframes() > 0 {
            self.image.render_frame(0)
        } else {
            self.image.render_loading_frame()
        };
        let render = render.map_err(|e| {
            ImageError::Decoding(DecodingError::new(
                ImageFormatHint::PathExtension("jxl".into()),
                e,
            ))
        })?;
        let stream = render.stream();

        let image_header = self.image.image_header();
        let first_frame = self.image.frame_by_keyframe(0).unwrap();
        let first_frame_header = first_frame.header();

        let is_float = image_header.metadata.xyb_encoded
            || matches!(
                image_header.metadata.bit_depth,
                jxl_image::BitDepth::FloatSample { .. }
            )
            || first_frame_header.encoding == jxl_frame::header::Encoding::VarDct;
        let need_16bit = image_header.metadata.bit_depth.bits_per_sample() > 8;

        if is_float {
            stream_to_buf::<f32>(stream, buf);
        } else if need_16bit {
            stream_to_buf::<u16>(stream, buf);
        } else {
            stream_to_buf::<u8>(stream, buf);
        }

        Ok(())
    }
}

impl<R: Read> image::ImageDecoder for JxlDecoder<R> {
    fn dimensions(&self) -> (u32, u32) {
        (self.image.width(), self.image.height())
    }

    fn color_type(&self) -> image::ColorType {
        let pixel_format = self.image.pixel_format();
        let image_header = self.image.image_header();
        let first_frame = self.image.frame_by_keyframe(0).unwrap();
        let first_frame_header = first_frame.header();

        let is_float = image_header.metadata.xyb_encoded
            || matches!(
                image_header.metadata.bit_depth,
                jxl_image::BitDepth::FloatSample { .. }
            )
            || first_frame_header.encoding == jxl_frame::header::Encoding::VarDct;
        let need_16bit = image_header.metadata.bit_depth.bits_per_sample() > 8;

        match (
            pixel_format.is_grayscale(),
            pixel_format.has_alpha(),
            is_float,
            need_16bit,
        ) {
            (false, false, false, false) => ColorType::Rgb8,
            (false, false, false, true) => ColorType::Rgb16,
            (false, false, true, _) => ColorType::Rgb32F,
            (false, true, false, false) => ColorType::Rgba8,
            (false, true, false, true) => ColorType::Rgba16,
            (false, true, true, _) => ColorType::Rgba32F,
            (true, false, false, false) => ColorType::L8,
            (true, false, false, true) | (true, false, true, _) => ColorType::L16,
            (true, true, false, false) => ColorType::La8,
            (true, true, false, true) | (true, true, true, _) => ColorType::La16,
        }
    }

    fn read_image(mut self, buf: &mut [u8]) -> ImageResult<()>
    where
        Self: Sized,
    {
        self.read_image_inner(buf).map_err(|e| {
            ImageError::Decoding(DecodingError::new(
                ImageFormatHint::PathExtension("jxl".into()),
                e,
            ))
        })
    }

    fn read_image_boxed(mut self: Box<Self>, buf: &mut [u8]) -> ImageResult<()> {
        self.read_image_inner(buf).map_err(|e| {
            ImageError::Decoding(DecodingError::new(
                ImageFormatHint::PathExtension("jxl".into()),
                e,
            ))
        })
    }

    fn icc_profile(&mut self) -> ImageResult<Option<Vec<u8>>> {
        Ok(Some(self.image.rendered_icc()))
    }

    fn set_limits(&mut self, limits: image::Limits) -> ImageResult<()> {
        use image::error::{LimitError, LimitErrorKind};

        if let Some(max_width) = limits.max_image_width {
            if self.image.width() > max_width {
                return Err(ImageError::Limits(LimitError::from_kind(
                    LimitErrorKind::DimensionError,
                )));
            }
        }

        if let Some(max_height) = limits.max_image_height {
            if self.image.height() > max_height {
                return Err(ImageError::Limits(LimitError::from_kind(
                    LimitErrorKind::DimensionError,
                )));
            }
        }

        let alloc_tracker = self.image.ctx.alloc_tracker();
        match (alloc_tracker, limits.max_alloc) {
            (Some(tracker), max_alloc) => {
                let new_memory_limit = max_alloc.map(|x| x as usize).unwrap_or(usize::MAX);
                if new_memory_limit > self.current_memory_limit {
                    tracker.expand_limit(new_memory_limit - self.current_memory_limit);
                } else {
                    tracker
                        .shrink_limit(self.current_memory_limit - new_memory_limit)
                        .map_err(|_| {
                            ImageError::Limits(LimitError::from_kind(
                                LimitErrorKind::InsufficientMemory,
                            ))
                        })?;
                }

                self.current_memory_limit = new_memory_limit;
            }
            (None, None) => {}
            (None, Some(_)) => {
                return Err(ImageError::Limits(LimitError::from_kind(
                    LimitErrorKind::Unsupported {
                        limits,
                        supported: image::LimitSupport::default(),
                    },
                )));
            }
        }

        Ok(())
    }
}

fn stream_to_buf<Sample: crate::FrameBufferSample>(
    mut stream: crate::ImageStream<'_>,
    buf: &mut [u8],
) {
    let stride =
        stream.width() as usize * stream.channels() as usize * std::mem::size_of::<Sample>();
    assert_eq!(buf.len(), stride * stream.height() as usize);

    if let Ok(buf) = bytemuck::try_cast_slice_mut::<u8, Sample>(buf) {
        stream.write_to_buffer(buf);
    } else {
        let mut row = Vec::with_capacity(stream.width() as usize);
        row.fill_with(Sample::default);
        for buf_row in buf.chunks_exact_mut(stride) {
            stream.write_to_buffer(&mut row);

            let row = bytemuck::cast_slice::<Sample, u8>(&row);
            buf_row.copy_from_slice(row);
        }
    }
}
