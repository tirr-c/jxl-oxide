use std::io::prelude::*;

use image::error::{DecodingError, ImageFormatHint};
use image::{ColorType, ImageError, ImageResult};
use jxl_grid::AllocTracker;

use crate::{AuxBoxData, CropInfo, InitializeResult, JxlImage};

/// JPEG XL decoder which implements [`ImageDecoder`][image::ImageDecoder].
///
/// # Supported features
///
/// Currently `JxlDecoder` supports following features:
/// - Returning images of 8-bit, 16-bit integer and 32-bit float samples
/// - RGB or luma-only images, with or without alpha
/// - Returning ICC profiles via `icc_profile`
/// - Returning Exif metadata via `exif_metadata`
/// - Setting decoder limits (caveat: memory limits are not strict)
/// - Cropped decoding with [`ImageDecoderRect`][image::ImageDecoderRect]
/// - (When `lcms2` feature is enabled) Converting CMYK images to sRGB color space
///
/// Some features are planned but not implemented yet:
/// - Decoding animations
///
/// # Note about color management
///
/// `JxlDecoder` doesn't do color management by itself (except for CMYK images, which will be
/// converted to sRGB color space if `lcms2` is available). Consumers should apply appropriate
/// color transforms using ICC profile returned by [`icc_profile()`], otherwise colors may be
/// inaccurate.
///
/// # Examples
///
/// Converting JPEG XL image to PNG:
///
/// ```no_run
/// use image::{DynamicImage, ImageDecoder};
/// use jxl_oxide::integration::JxlDecoder;
///
/// # type Result<T, E = Box<dyn std::error::Error>> = std::result::Result<T, E>;
/// # fn do_color_transform(_: &mut DynamicImage, _: Vec<u8>) -> Result<()> { Ok(()) }
/// # fn main() -> Result<()> {
/// // Read and decode a JPEG XL image.
/// let file = std::fs::File::open("image.jxl")?;
/// let mut decoder = JxlDecoder::new(file)?;
/// let icc = decoder.icc_profile()?;
/// let mut image = DynamicImage::from_decoder(decoder)?;
///
/// // Perform color transform using the ICC profile.
/// // Note that ICC profile will be always available for images decoded by `JxlDecoder`.
/// if let Some(icc) = icc {
///     do_color_transform(&mut image, icc)?;
/// }
///
/// // Save decoded image to PNG.
/// image.save("image.png")?;
/// # Ok(()) }
/// ```
///
/// [`icc_profile()`]: image::ImageDecoder::icc_profile
pub struct JxlDecoder<R> {
    reader: R,
    image: JxlImage,
    current_crop: CropInfo,
    current_memory_limit: usize,
    buf: Vec<u8>,
    buf_valid: usize,
}

impl<R: Read> JxlDecoder<R> {
    /// Initializes a decoder which reads from given image stream.
    ///
    /// Decoder will be initialized with default thread pool.
    pub fn new(reader: R) -> ImageResult<Self> {
        let builder = JxlImage::builder().alloc_tracker(AllocTracker::with_limit(usize::MAX));

        Self::init(builder, reader)
    }

    /// Initializes a decoder which reads from given image stream, with custom thread pool.
    pub fn with_thread_pool(reader: R, pool: crate::JxlThreadPool) -> ImageResult<Self> {
        let builder = JxlImage::builder()
            .pool(pool)
            .alloc_tracker(AllocTracker::with_limit(usize::MAX));

        Self::init(builder, reader)
    }

    fn init(builder: crate::JxlImageBuilder, mut reader: R) -> ImageResult<Self> {
        let mut buf = vec![0u8; 4096];
        let mut buf_valid = 0usize;
        let image = Self::init_image(builder, &mut reader, &mut buf, &mut buf_valid)
            .map_err(|e| ImageError::Decoding(DecodingError::new(ImageFormatHint::Unknown, e)))?;

        let crop = CropInfo {
            width: image.width(),
            height: image.height(),
            left: 0,
            top: 0,
        };

        let mut decoder = Self {
            reader,
            image,
            current_memory_limit: usize::MAX,
            current_crop: crop,
            buf,
            buf_valid,
        };

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

    fn load_until_condition(
        &mut self,
        mut predicate: impl FnMut(&JxlImage) -> crate::Result<bool>,
    ) -> crate::Result<()> {
        while !predicate(&self.image)? {
            let count = self.reader.read(&mut self.buf[self.buf_valid..])?;
            if count == 0 {
                break;
            }
            self.buf_valid += count;
            let consumed = self.image.feed_bytes(&self.buf[..self.buf_valid])?;
            self.buf.copy_within(consumed..self.buf_valid, 0);
            self.buf_valid -= consumed;
        }

        Ok(())
    }

    fn load_until_first_keyframe(&mut self) -> crate::Result<()> {
        self.load_until_condition(|image| Ok(image.ctx.loaded_frames() > 0))?;

        if self.image.frame_by_keyframe(0).is_none() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "reader ended before parsing first frame",
            )
            .into());
        }

        Ok(())
    }

    fn load_until_exif(&mut self) -> crate::Result<()> {
        self.load_until_condition(|image| Ok(!image.aux_boxes().first_exif()?.is_decoding()))
    }

    #[inline]
    fn is_float(&self) -> bool {
        use crate::BitDepth;

        let metadata = &self.image.image_header().metadata;
        matches!(
            metadata.bit_depth,
            BitDepth::FloatSample { .. }
                | BitDepth::IntegerSample {
                    bits_per_sample: 17..
                }
        )
    }

    #[inline]
    fn need_16bit(&self) -> bool {
        let metadata = &self.image.image_header().metadata;
        metadata.bit_depth.bits_per_sample() > 8
    }

    fn read_image_inner(
        &mut self,
        crop: CropInfo,
        buf: &mut [u8],
        buf_stride: Option<usize>,
    ) -> crate::Result<()> {
        if self.current_crop != crop {
            self.image.set_image_region(crop);
            self.current_crop = crop;
        }

        self.load_until_first_keyframe()?;

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

        let stride_base = stream.width() as usize * stream.channels() as usize;
        if self.is_float() && !self.image.pixel_format().is_grayscale() {
            let stride = buf_stride.unwrap_or(stride_base * std::mem::size_of::<f32>());
            stream_to_buf::<f32>(stream, buf, stride);
        } else if self.need_16bit() {
            let stride = buf_stride.unwrap_or(stride_base * std::mem::size_of::<u16>());
            stream_to_buf::<u16>(stream, buf, stride);
        } else {
            let stride = buf_stride.unwrap_or(stride_base * std::mem::size_of::<u8>());
            stream_to_buf::<u8>(stream, buf, stride);
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

        match (
            pixel_format.is_grayscale(),
            pixel_format.has_alpha(),
            self.is_float(),
            self.need_16bit(),
        ) {
            (false, false, false, false) => ColorType::Rgb8,
            (false, false, false, true) => ColorType::Rgb16,
            (false, false, true, _) => ColorType::Rgb32F,
            (false, true, false, false) => ColorType::Rgba8,
            (false, true, false, true) => ColorType::Rgba16,
            (false, true, true, _) => ColorType::Rgba32F,
            (true, false, _, false) => ColorType::L8,
            (true, false, _, true) => ColorType::L16,
            (true, true, _, false) => ColorType::La8,
            (true, true, _, true) => ColorType::La16,
        }
    }

    fn read_image(mut self, buf: &mut [u8]) -> ImageResult<()>
    where
        Self: Sized,
    {
        let crop = CropInfo {
            width: self.image.width(),
            height: self.image.height(),
            left: 0,
            top: 0,
        };

        self.read_image_inner(crop, buf, None).map_err(|e| {
            ImageError::Decoding(DecodingError::new(
                ImageFormatHint::PathExtension("jxl".into()),
                e,
            ))
        })
    }

    fn read_image_boxed(mut self: Box<Self>, buf: &mut [u8]) -> ImageResult<()> {
        let crop = CropInfo {
            width: self.image.width(),
            height: self.image.height(),
            left: 0,
            top: 0,
        };

        self.read_image_inner(crop, buf, None).map_err(|e| {
            ImageError::Decoding(DecodingError::new(
                ImageFormatHint::PathExtension("jxl".into()),
                e,
            ))
        })
    }

    fn icc_profile(&mut self) -> ImageResult<Option<Vec<u8>>> {
        Ok(Some(self.image.rendered_icc()))
    }

    fn exif_metadata(&mut self) -> ImageResult<Option<Vec<u8>>> {
        self.load_until_exif().map_err(|e| {
            ImageError::Decoding(DecodingError::new(
                ImageFormatHint::PathExtension("jxl".into()),
                e,
            ))
        })?;

        let aux_boxes = self.image.aux_boxes();
        let AuxBoxData::Data(exif) = aux_boxes.first_exif().unwrap() else {
            return Ok(None);
        };
        Ok(Some(exif.payload().to_vec()))
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

impl<R: Read> image::ImageDecoderRect for JxlDecoder<R> {
    fn read_rect(
        &mut self,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        buf: &mut [u8],
        row_pitch: usize,
    ) -> ImageResult<()> {
        let crop = CropInfo {
            width,
            height,
            left: x,
            top: y,
        };

        self.read_image_inner(crop, buf, Some(row_pitch))
            .map_err(|e| {
                ImageError::Decoding(DecodingError::new(
                    ImageFormatHint::PathExtension("jxl".into()),
                    e,
                ))
            })
    }
}

fn stream_to_buf<Sample: crate::FrameBufferSample>(
    mut stream: crate::ImageStream<'_>,
    buf: &mut [u8],
    buf_stride: usize,
) {
    let stride =
        stream.width() as usize * stream.channels() as usize * std::mem::size_of::<Sample>();
    assert!(buf_stride >= stride);
    assert_eq!(buf.len(), buf_stride * stream.height() as usize);

    if let Ok(buf) = bytemuck::try_cast_slice_mut::<u8, Sample>(buf) {
        if buf_stride == stride {
            stream.write_to_buffer(buf);
        } else {
            for buf_row in buf.chunks_exact_mut(buf_stride / std::mem::size_of::<Sample>()) {
                let buf_row = &mut buf_row[..stream.width() as usize];
                stream.write_to_buffer(buf_row);
            }
        }
    } else {
        let mut row = Vec::with_capacity(stream.width() as usize);
        row.fill_with(Sample::default);
        for buf_row in buf.chunks_exact_mut(stride) {
            stream.write_to_buffer(&mut row);

            let row = bytemuck::cast_slice::<Sample, u8>(&row);
            buf_row[..stride].copy_from_slice(row);
        }
    }
}
