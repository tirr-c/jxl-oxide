//! jxl-oxide is a JPEG XL decoder written in pure Rust. It's internally organized into a few
//! small crates. This crate acts as a blanket and provides a simple interface made from those
//! crates to decode the actual image.
//!
//! # Decoding an image
//!
//! Decoding a JPEG XL image starts with constructing [`JxlImage`]. First create a builder using
//! [`JxlImage::builder`], and use [`open`][JxlImageBuilder::open] to read a file:
//!
//! ```no_run
//! # use jxl_oxide::JxlImage;
//! let image = JxlImage::builder().open("input.jxl").expect("Failed to read image header");
//! println!("{:?}", image.image_header()); // Prints the image header
//! ```
//!
//! Or, if you're reading from a reader that implements [`Read`][std::io::Read], you can use
//! [`read`][JxlImageBuilder::read]:
//!
//! ```no_run
//! # use jxl_oxide::JxlImage;
//! # let reader = std::io::empty();
//! let image = JxlImage::builder().read(reader).expect("Failed to read image header");
//! println!("{:?}", image.image_header()); // Prints the image header
//! ```
//!
//! In async context, you'll probably want to feed byte buffers directly. In this case, create an
//! image struct with *uninitialized state* using [`build_uninit`][JxlImageBuilder::build_uninit],
//! and call [`feed_bytes`][UninitializedJxlImage::feed_bytes] and
//! [`try_init`][UninitializedJxlImage::try_init]:
//!
//! ```no_run
//! # struct StubReader(&'static [u8]);
//! # impl StubReader {
//! #     fn read(&self) -> StubReaderFuture { StubReaderFuture(self.0) }
//! # }
//! # struct StubReaderFuture(&'static [u8]);
//! # impl std::future::Future for StubReaderFuture {
//! #     type Output = jxl_oxide::Result<&'static [u8]>;
//! #     fn poll(
//! #         self: std::pin::Pin<&mut Self>,
//! #         cx: &mut std::task::Context<'_>,
//! #     ) -> std::task::Poll<Self::Output> {
//! #         std::task::Poll::Ready(Ok(self.0))
//! #     }
//! # }
//! #
//! # use jxl_oxide::{JxlImage, InitializeResult};
//! # async fn run() -> jxl_oxide::Result<()> {
//! # let reader = StubReader(&[
//! #   0xff, 0x0a, 0x30, 0x54, 0x10, 0x09, 0x08, 0x06, 0x01, 0x00, 0x78, 0x00,
//! #   0x4b, 0x38, 0x41, 0x3c, 0xb6, 0x3a, 0x51, 0xfe, 0x00, 0x47, 0x1e, 0xa0,
//! #   0x85, 0xb8, 0x27, 0x1a, 0x48, 0x45, 0x84, 0x1b, 0x71, 0x4f, 0xa8, 0x3e,
//! #   0x8e, 0x30, 0x03, 0x92, 0x84, 0x01,
//! # ]);
//! let mut uninit_image = JxlImage::builder().build_uninit();
//! let image = loop {
//!     uninit_image.feed_bytes(reader.read().await?);
//!     match uninit_image.try_init()? {
//!         InitializeResult::NeedMoreData(uninit) => {
//!             uninit_image = uninit;
//!         }
//!         InitializeResult::Initialized(image) => {
//!             break image;
//!         }
//!     }
//! };
//! println!("{:?}", image.image_header()); // Prints the image header
//! # Ok(())
//! # }
//! ```
//!
//! `JxlImage` parses the image header and embedded ICC profile (if there's any). Use
//! [`JxlImage::render_frame`] to render the image.
//!
//! ```no_run
//! # use jxl_oxide::Render;
//! use jxl_oxide::{JxlImage, RenderResult};
//!
//! # fn present_image(_: Render) {}
//! # fn main() -> jxl_oxide::Result<()> {
//! # let image = JxlImage::builder().open("input.jxl").unwrap();
//! for keyframe_idx in 0..image.num_loaded_keyframes() {
//!     let render = image.render_frame(keyframe_idx)?;
//!     present_image(render);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! # Color management
//! jxl-oxide has basic color management support, which enables color transformation between
//! well-known color encodings and parsing simple, matrix-based ICC profiles. However, jxl-oxide
//! alone does not support conversion to and from arbitrary ICC profiles, notably CMYK profiles.
//! This includes converting from embedded ICC profiles.
//!
//! Use [`JxlImage::request_color_encoding`] or [`JxlImage::request_icc`] to set color encoding of
//! rendered images. Conversion to and/or from ICC profiles may occur if you do this; in that case,
//! external CMS need to be set using [`JxlImage::set_cms`].
//!
//! ```no_run
//! # use jxl_oxide::{EnumColourEncoding, JxlImage, RenderingIntent};
//! # use jxl_oxide::NullCms as MyCustomCms;
//! # let reader = std::io::empty();
//! let mut image = JxlImage::builder().read(reader).expect("Failed to read image header");
//! image.set_cms(MyCustomCms);
//!
//! let color_encoding = EnumColourEncoding::display_p3(RenderingIntent::Perceptual);
//! image.request_color_encoding(color_encoding);
//! ```
//!
//! External CMS is set to Little CMS 2 by default if `lcms2` feature is enabled. You can
//! explicitly disable this by setting CMS to [`NullCms`].
//!
//! ```no_run
//! # use jxl_oxide::{JxlImage, NullCms};
//! # let reader = std::io::empty();
//! let mut image = JxlImage::builder().read(reader).expect("Failed to read image header");
//! image.set_cms(NullCms);
//! ```
//!
//! ## Not using `set_cms` for color management
//! If implementing `ColorManagementSystem` is difficult for your use case, color management can be
//! done separately using ICC profile of rendered images. [`JxlImage::rendered_icc`] returns ICC
//! profile for further processing.
//!
//! ```no_run
//! # use jxl_oxide::Render;
//! use jxl_oxide::{JxlImage, RenderResult};
//!
//! # fn present_image_with_cms(_: Render, _: &[u8]) {}
//! # fn main() -> jxl_oxide::Result<()> {
//! # let image = JxlImage::builder().open("input.jxl").unwrap();
//! let icc_profile = image.rendered_icc();
//! for keyframe_idx in 0..image.num_loaded_keyframes() {
//!     let render = image.render_frame(keyframe_idx)?;
//!     present_image_with_cms(render, &icc_profile);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! # Feature flags
//! - `rayon`: Enable multithreading with Rayon. (*default*)
//! - `image`: Enable integration with `image` crate.
//! - `lcms2`: Enable integration with Little CMS 2.
//! - `moxcms`: Enable integration with `moxcms` crate.

#![cfg_attr(docsrs, feature(doc_auto_cfg))]

use std::sync::Arc;

use jxl_bitstream::{Bitstream, ContainerParser, ParseEvent};
use jxl_frame::FrameContext;
use jxl_image::BitDepth;
use jxl_oxide_common::{Bundle, Name};
use jxl_render::ImageBuffer;
use jxl_render::ImageWithRegion;
use jxl_render::Region;
use jxl_render::{IndexedFrame, RenderContext};

pub use jxl_color::{ColorEncodingWithProfile, ColorManagementSystem, NullCms, PreparedTransform};
pub use jxl_frame::header as frame;
pub use jxl_frame::{Frame, FrameHeader};
pub use jxl_grid::{AlignedGrid, AllocTracker};
pub use jxl_image::color::{self, EnumColourEncoding, RenderingIntent};
pub use jxl_image::{self as image, ExtraChannelType, ImageHeader};
pub use jxl_jbr as jpeg_bitstream;
pub use jxl_threadpool::JxlThreadPool;

mod aux_box;
mod fb;
pub mod integration;
#[cfg(feature = "lcms2")]
mod lcms2;
#[cfg(feature = "moxcms")]
mod moxcms;

#[cfg(feature = "lcms2")]
pub use self::lcms2::Lcms2;
#[cfg(feature = "moxcms")]
pub use self::moxcms::Moxcms;
pub use aux_box::{AuxBoxData, AuxBoxList, RawExif};
pub use fb::{FrameBuffer, FrameBufferSample, ImageStream};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync + 'static>>;

#[cfg(feature = "rayon")]
fn default_pool() -> JxlThreadPool {
    JxlThreadPool::rayon_global()
}

#[cfg(not(feature = "rayon"))]
fn default_pool() -> JxlThreadPool {
    JxlThreadPool::none()
}

/// JPEG XL image decoder builder.
#[derive(Debug, Default)]
pub struct JxlImageBuilder {
    pool: Option<JxlThreadPool>,
    tracker: Option<AllocTracker>,
    force_wide_buffers: bool,
}

impl JxlImageBuilder {
    /// Sets a custom thread pool.
    pub fn pool(mut self, pool: JxlThreadPool) -> Self {
        self.pool = Some(pool);
        self
    }

    /// Sets an allocation tracker.
    pub fn alloc_tracker(mut self, tracker: AllocTracker) -> Self {
        self.tracker = Some(tracker);
        self
    }

    /// Force 32-bit Modular buffers when decoding.
    pub fn force_wide_buffers(mut self, force_wide_buffers: bool) -> Self {
        self.force_wide_buffers = force_wide_buffers;
        self
    }

    /// Consumes the builder, and creates an empty, uninitialized JPEG XL image decoder.
    pub fn build_uninit(self) -> UninitializedJxlImage {
        UninitializedJxlImage {
            pool: self.pool.unwrap_or_else(default_pool),
            tracker: self.tracker,
            reader: ContainerParser::new(),
            buffer: Vec::new(),
            aux_boxes: AuxBoxList::new(),
            force_wide_buffers: self.force_wide_buffers,
        }
    }

    /// Consumes the builder, and creates a JPEG XL image decoder by reading image from the reader.
    pub fn read(self, mut reader: impl std::io::Read) -> Result<JxlImage> {
        let mut uninit = self.build_uninit();
        let mut buf = vec![0u8; 4096];
        let mut buf_valid = 0usize;
        let mut image = loop {
            let count = reader.read(&mut buf[buf_valid..])?;
            if count == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "reader ended before parsing image header",
                )
                .into());
            }
            buf_valid += count;
            let consumed = uninit.feed_bytes(&buf[..buf_valid])?;
            buf.copy_within(consumed..buf_valid, 0);
            buf_valid -= consumed;

            match uninit.try_init()? {
                InitializeResult::NeedMoreData(x) => {
                    uninit = x;
                }
                InitializeResult::Initialized(x) => {
                    break x;
                }
            }
        };

        while !image.inner.end_of_image {
            let count = reader.read(&mut buf[buf_valid..])?;
            if count == 0 {
                break;
            }
            buf_valid += count;
            let consumed = image.feed_bytes(&buf[..buf_valid])?;
            buf.copy_within(consumed..buf_valid, 0);
            buf_valid -= consumed;
        }

        buf.truncate(buf_valid);
        image.finalize()?;
        Ok(image)
    }

    /// Consumes the builder, and creates a JPEG XL image decoder by reading image from the file.
    pub fn open(self, path: impl AsRef<std::path::Path>) -> Result<JxlImage> {
        let file = std::fs::File::open(path)?;
        self.read(file)
    }
}

/// Empty, uninitialized JPEG XL image.
///
/// # Examples
/// ```no_run
/// # fn read_bytes() -> jxl_oxide::Result<&'static [u8]> { Ok(&[]) }
/// # use jxl_oxide::{JxlImage, InitializeResult};
/// # fn main() -> jxl_oxide::Result<()> {
/// let mut uninit_image = JxlImage::builder().build_uninit();
/// let image = loop {
///     let buf = read_bytes()?;
///     uninit_image.feed_bytes(buf)?;
///     match uninit_image.try_init()? {
///         InitializeResult::NeedMoreData(uninit) => {
///             uninit_image = uninit;
///         }
///         InitializeResult::Initialized(image) => {
///             break image;
///         }
///     }
/// };
/// println!("{:?}", image.image_header());
/// # Ok(())
/// # }
/// ```
pub struct UninitializedJxlImage {
    pool: JxlThreadPool,
    tracker: Option<AllocTracker>,
    reader: ContainerParser,
    buffer: Vec<u8>,
    aux_boxes: AuxBoxList,
    force_wide_buffers: bool,
}

impl UninitializedJxlImage {
    /// Feeds more data into the decoder.
    ///
    /// Returns total consumed bytes from the buffer.
    pub fn feed_bytes(&mut self, buf: &[u8]) -> Result<usize> {
        for event in self.reader.feed_bytes(buf) {
            match event? {
                ParseEvent::BitstreamKind(_) => {}
                ParseEvent::Codestream(buf) => {
                    self.buffer.extend_from_slice(buf);
                }
                aux_box_event => {
                    self.aux_boxes.handle_event(aux_box_event)?;
                }
            }
        }
        Ok(self.reader.previous_consumed_bytes())
    }

    /// Returns the internal reader.
    #[inline]
    pub fn reader(&self) -> &ContainerParser {
        &self.reader
    }

    /// Try to initialize an image with the data fed into so far.
    ///
    /// # Returns
    /// - `Ok(InitializeResult::Initialized(_))` if the initialization was successful,
    /// - `Ok(InitializeResult::NeedMoreData(_))` if the data was not enough, and
    /// - `Err(_)` if there was a decode error during the initialization, meaning invalid bitstream
    ///   was given.
    pub fn try_init(mut self) -> Result<InitializeResult> {
        let mut bitstream = Bitstream::new(&self.buffer);
        let image_header = match ImageHeader::parse(&mut bitstream, ()) {
            Ok(x) => x,
            Err(e) if e.unexpected_eof() => {
                return Ok(InitializeResult::NeedMoreData(self));
            }
            Err(e) => {
                return Err(e.into());
            }
        };

        let embedded_icc = if image_header.metadata.colour_encoding.want_icc() {
            let icc = match jxl_color::icc::read_icc(&mut bitstream) {
                Ok(x) => x,
                Err(e) if e.unexpected_eof() => {
                    return Ok(InitializeResult::NeedMoreData(self));
                }
                Err(e) => {
                    return Err(e.into());
                }
            };
            tracing::debug!("Image has an embedded ICC profile");
            let icc = jxl_color::icc::decode_icc(&icc)?;
            Some(icc)
        } else {
            None
        };
        bitstream.zero_pad_to_byte()?;

        let image_header = Arc::new(image_header);
        let skip_bytes = if image_header.metadata.preview.is_some() {
            let frame = match Frame::parse(
                &mut bitstream,
                FrameContext {
                    image_header: image_header.clone(),
                    tracker: self.tracker.as_ref(),
                    pool: self.pool.clone(),
                },
            ) {
                Ok(x) => x,
                Err(e) if e.unexpected_eof() => {
                    return Ok(InitializeResult::NeedMoreData(self));
                }
                Err(e) => {
                    return Err(e.into());
                }
            };

            let bytes_read = bitstream.num_read_bits() / 8;
            let x = frame.toc().total_byte_size();
            if self.buffer.len() < bytes_read + x {
                return Ok(InitializeResult::NeedMoreData(self));
            }

            x
        } else {
            0usize
        };

        let bytes_read = bitstream.num_read_bits() / 8 + skip_bytes;
        self.buffer.drain(..bytes_read);

        let render_spot_color = !image_header.metadata.grayscale();

        let mut builder = RenderContext::builder().pool(self.pool.clone());
        if let Some(icc) = embedded_icc {
            builder = builder.embedded_icc(icc);
        }
        if let Some(tracker) = self.tracker {
            builder = builder.alloc_tracker(tracker);
        }
        builder = builder.force_wide_buffers(self.force_wide_buffers);
        #[cfg_attr(not(any(feature = "lcms2", feature = "moxcms")), allow(unused_mut))]
        let mut ctx = builder.build(image_header.clone())?;
        #[cfg(feature = "lcms2")]
        ctx.set_cms(Lcms2);
        #[cfg(all(not(feature = "lcms2"), feature = "moxcms"))]
        ctx.set_cms(Moxcms);

        let mut image = JxlImage {
            pool: self.pool.clone(),
            reader: self.reader,
            image_header,
            ctx: Box::new(ctx),
            render_spot_color,
            inner: JxlImageInner {
                end_of_image: false,
                buffer: Vec::new(),
                buffer_offset: bytes_read,
                frame_offsets: Vec::new(),
                aux_boxes: self.aux_boxes,
            },
        };
        image.inner.feed_bytes_inner(&mut image.ctx, &self.buffer)?;

        Ok(InitializeResult::Initialized(image))
    }
}

/// Initialization result from [`UninitializedJxlImage::try_init`].
pub enum InitializeResult {
    /// The data was not enough. Feed more data into the returned image.
    NeedMoreData(UninitializedJxlImage),
    /// The image is successfully initialized.
    Initialized(JxlImage),
}

/// JPEG XL image.
#[derive(Debug)]
pub struct JxlImage {
    pool: JxlThreadPool,
    reader: ContainerParser,
    image_header: Arc<ImageHeader>,
    ctx: Box<RenderContext>,
    render_spot_color: bool,
    inner: JxlImageInner,
}

/// # Constructors and data-feeding methods
impl JxlImage {
    /// Creates a decoder builder with default options.
    pub fn builder() -> JxlImageBuilder {
        JxlImageBuilder::default()
    }

    /// Reads an image from the reader with default options.
    pub fn read_with_defaults(reader: impl std::io::Read) -> Result<JxlImage> {
        Self::builder().read(reader)
    }

    /// Opens an image in the filesystem with default options.
    pub fn open_with_defaults(path: impl AsRef<std::path::Path>) -> Result<JxlImage> {
        Self::builder().open(path)
    }

    /// Feeds more data into the decoder.
    ///
    /// Returns total consumed bytes from the buffer.
    pub fn feed_bytes(&mut self, buf: &[u8]) -> Result<usize> {
        for event in self.reader.feed_bytes(buf) {
            match event? {
                ParseEvent::BitstreamKind(_) => {}
                ParseEvent::Codestream(buf) => {
                    self.inner.feed_bytes_inner(&mut self.ctx, buf)?;
                }
                aux_box_event => {
                    self.inner.aux_boxes.handle_event(aux_box_event)?;
                }
            }
        }
        Ok(self.reader.previous_consumed_bytes())
    }

    /// Signals the end of bitstream.
    ///
    /// This is automatically done if `open()` or `read()` is used to decode the image.
    pub fn finalize(&mut self) -> Result<()> {
        self.inner.aux_boxes.eof()?;
        Ok(())
    }
}

/// # Image and decoder metadata accessors
impl JxlImage {
    /// Returns the image header.
    #[inline]
    pub fn image_header(&self) -> &ImageHeader {
        &self.image_header
    }

    /// Returns the image width with orientation applied.
    #[inline]
    pub fn width(&self) -> u32 {
        self.image_header.width_with_orientation()
    }

    /// Returns the image height with orientation applied.
    #[inline]
    pub fn height(&self) -> u32 {
        self.image_header.height_with_orientation()
    }

    /// Returns the *original* ICC profile embedded in the image.
    #[inline]
    pub fn original_icc(&self) -> Option<&[u8]> {
        self.ctx.embedded_icc()
    }

    /// Returns the ICC profile that describes rendered images.
    ///
    /// The returned profile will change if different color encoding is specified using
    /// [`request_icc`][Self::request_icc] or
    /// [`request_color_encoding`][Self::request_color_encoding].
    pub fn rendered_icc(&self) -> Vec<u8> {
        let encoding = self.ctx.requested_color_encoding();
        match encoding.encoding() {
            color::ColourEncoding::Enum(encoding) => {
                jxl_color::icc::colour_encoding_to_icc(encoding)
            }
            color::ColourEncoding::IccProfile(_) => encoding.icc_profile().to_vec(),
        }
    }

    /// Returns the CICP tag of the color encoding of rendered images, if there's any.
    #[inline]
    pub fn rendered_cicp(&self) -> Option<[u8; 4]> {
        let encoding = self.ctx.requested_color_encoding();
        encoding.encoding().cicp()
    }

    /// Returns the pixel format of the rendered image.
    pub fn pixel_format(&self) -> PixelFormat {
        let encoding = self.ctx.requested_color_encoding();
        let is_grayscale = encoding.is_grayscale();
        let has_black = encoding.is_cmyk();
        let mut has_alpha = false;
        for ec_info in &self.image_header.metadata.ec_info {
            if ec_info.is_alpha() {
                has_alpha = true;
            }
        }

        match (is_grayscale, has_black, has_alpha) {
            (false, false, false) => PixelFormat::Rgb,
            (false, false, true) => PixelFormat::Rgba,
            (false, true, false) => PixelFormat::Cmyk,
            (false, true, true) => PixelFormat::Cmyka,
            (true, _, false) => PixelFormat::Gray,
            (true, _, true) => PixelFormat::Graya,
        }
    }

    /// Returns what HDR transfer function the image uses, if there's any.
    ///
    /// Returns `None` if the image is not HDR one.
    pub fn hdr_type(&self) -> Option<HdrType> {
        self.ctx.suggested_hdr_tf().and_then(|tf| match tf {
            color::TransferFunction::Pq => Some(HdrType::Pq),
            color::TransferFunction::Hlg => Some(HdrType::Hlg),
            _ => None,
        })
    }

    /// Returns whether the spot color channels will be rendered.
    #[inline]
    pub fn render_spot_color(&self) -> bool {
        self.render_spot_color
    }

    /// Sets whether the spot colour channels will be rendered.
    #[inline]
    pub fn set_render_spot_color(&mut self, render_spot_color: bool) -> &mut Self {
        if render_spot_color && self.image_header.metadata.grayscale() {
            tracing::warn!("Spot colour channels are not rendered on grayscale images");
            return self;
        }
        self.render_spot_color = render_spot_color;
        self
    }

    /// Returns the list of auxiliary boxes in the JPEG XL container.
    ///
    /// The list may contain Exif and XMP metadata.
    pub fn aux_boxes(&self) -> &AuxBoxList {
        &self.inner.aux_boxes
    }

    /// Returns the number of currently loaded keyframes.
    #[inline]
    pub fn num_loaded_keyframes(&self) -> usize {
        self.ctx.loaded_keyframes()
    }

    /// Returns the number of currently loaded frames, including frames that are not displayed
    /// directly.
    #[inline]
    pub fn num_loaded_frames(&self) -> usize {
        self.ctx.loaded_frames()
    }

    /// Returns whether the image is loaded completely, without missing animation keyframes or
    /// partially loaded frames.
    #[inline]
    pub fn is_loading_done(&self) -> bool {
        self.inner.end_of_image
    }

    /// Returns frame data by keyframe index.
    pub fn frame_by_keyframe(&self, keyframe_index: usize) -> Option<&IndexedFrame> {
        self.ctx.keyframe(keyframe_index)
    }

    /// Returns the frame header for the given keyframe index, or `None` if the keyframe does not
    /// exist.
    pub fn frame_header(&self, keyframe_index: usize) -> Option<&FrameHeader> {
        let frame = self.ctx.keyframe(keyframe_index)?;
        Some(frame.header())
    }

    /// Returns frame data by frame index, including frames that are not displayed directly.
    ///
    /// There are some situations where a frame is not displayed directly:
    /// - It may be marked as reference only, and meant to be only used by other frames.
    /// - It may contain LF image (which is 8x downsampled version) of another VarDCT frame.
    /// - Zero duration frame that is not the last frame of image is blended with following frames
    ///   and displayed together.
    pub fn frame(&self, frame_idx: usize) -> Option<&IndexedFrame> {
        self.ctx.frame(frame_idx)
    }

    /// Returns the offset of frame within codestream, in bytes.
    pub fn frame_offset(&self, frame_index: usize) -> Option<usize> {
        self.inner.frame_offsets.get(frame_index).copied()
    }

    /// Returns the thread pool used by the renderer.
    #[inline]
    pub fn pool(&self) -> &JxlThreadPool {
        &self.pool
    }

    /// Returns the internal reader.
    pub fn reader(&self) -> &ContainerParser {
        &self.reader
    }
}

/// # Color management methods
impl JxlImage {
    /// Sets color management system implementation to be used by the renderer.
    #[inline]
    pub fn set_cms(&mut self, cms: impl ColorManagementSystem + Send + Sync + 'static) {
        self.ctx.set_cms(cms);
    }

    /// Requests the decoder to render in specific color encoding, described by an ICC profile.
    ///
    /// # Errors
    /// This function will return an error if it cannot parse the ICC profile.
    pub fn request_icc(&mut self, icc_profile: &[u8]) -> Result<()> {
        self.ctx
            .request_color_encoding(ColorEncodingWithProfile::with_icc(icc_profile)?);
        Ok(())
    }

    /// Requests the decoder to render in specific color encoding, described by
    /// `EnumColourEncoding`.
    pub fn request_color_encoding(&mut self, color_encoding: EnumColourEncoding) {
        self.ctx
            .request_color_encoding(ColorEncodingWithProfile::new(color_encoding))
    }
}

/// # Rendering to image buffers
impl JxlImage {
    /// Renders the given keyframe.
    pub fn render_frame(&self, keyframe_index: usize) -> Result<Render> {
        self.render_frame_cropped(keyframe_index)
    }

    /// Renders the given keyframe with optional cropping region.
    pub fn render_frame_cropped(&self, keyframe_index: usize) -> Result<Render> {
        let image = self.ctx.render_keyframe(keyframe_index)?;

        let image_region = self
            .ctx
            .image_region()
            .apply_orientation(&self.image_header);
        let frame = self.ctx.keyframe(keyframe_index).unwrap();
        let frame_header = frame.header();
        let target_frame_region = image_region.translate(-frame_header.x0, -frame_header.y0);

        let is_cmyk = self.ctx.requested_color_encoding().is_cmyk();
        let result = Render {
            keyframe_index,
            name: frame_header.name.clone(),
            duration: frame_header.duration,
            orientation: self.image_header.metadata.orientation,
            image,
            extra_channels: self.convert_ec_info(),
            target_frame_region,
            color_bit_depth: self.image_header.metadata.bit_depth,
            is_cmyk,
            render_spot_color: self.render_spot_color,
        };
        Ok(result)
    }

    /// Renders the currently loading keyframe.
    pub fn render_loading_frame(&mut self) -> Result<Render> {
        self.render_loading_frame_cropped()
    }

    /// Renders the currently loading keyframe with optional cropping region.
    pub fn render_loading_frame_cropped(&mut self) -> Result<Render> {
        let (frame, image) = self.ctx.render_loading_keyframe()?;
        let frame_header = frame.header();
        let name = frame_header.name.clone();
        let duration = frame_header.duration;

        let image_region = self
            .ctx
            .image_region()
            .apply_orientation(&self.image_header);
        let frame = self
            .ctx
            .frame(self.ctx.loaded_frames())
            .or_else(|| self.ctx.frame(self.ctx.loaded_frames() - 1))
            .unwrap();
        let frame_header = frame.header();
        let target_frame_region = image_region.translate(-frame_header.x0, -frame_header.y0);

        let is_cmyk = self.ctx.requested_color_encoding().is_cmyk();
        let result = Render {
            keyframe_index: self.ctx.loaded_keyframes(),
            name,
            duration,
            orientation: self.image_header.metadata.orientation,
            image,
            extra_channels: self.convert_ec_info(),
            target_frame_region,
            color_bit_depth: self.image_header.metadata.bit_depth,
            is_cmyk,
            render_spot_color: self.render_spot_color,
        };
        Ok(result)
    }

    /// Returns current cropping region (region of interest).
    pub fn current_image_region(&self) -> CropInfo {
        let region = self.ctx.image_region();
        region.into()
    }

    /// Sets the cropping region (region of interest).
    ///
    /// Subsequent rendering methods will crop the image buffer according to the region.
    pub fn set_image_region(&mut self, region: CropInfo) -> &mut Self {
        self.ctx.request_image_region(region.into());
        self
    }
}

/// # JPEG bitstream reconstruction
impl JxlImage {
    /// Returns availability and validity of JPEG bitstream reconstruction data.
    pub fn jpeg_reconstruction_status(&self) -> JpegReconstructionStatus {
        match self.inner.aux_boxes.jbrd() {
            AuxBoxData::Data(jbrd) => {
                let header = jbrd.header();
                let Ok(exif) = self.inner.aux_boxes.first_exif() else {
                    return JpegReconstructionStatus::Invalid;
                };
                let xml = self.inner.aux_boxes.first_xml();

                if header.expected_icc_len() > 0 {
                    if !self.image_header.metadata.colour_encoding.want_icc() {
                        return JpegReconstructionStatus::Invalid;
                    } else if self.original_icc().is_none() {
                        return JpegReconstructionStatus::NeedMoreData;
                    }
                }
                if header.expected_exif_len() > 0 {
                    if exif.is_decoding() {
                        return JpegReconstructionStatus::NeedMoreData;
                    } else if exif.is_not_found() {
                        return JpegReconstructionStatus::Invalid;
                    }
                }
                if header.expected_xmp_len() > 0 {
                    if xml.is_decoding() {
                        return JpegReconstructionStatus::NeedMoreData;
                    } else if xml.is_not_found() {
                        return JpegReconstructionStatus::Invalid;
                    }
                }

                JpegReconstructionStatus::Available
            }
            AuxBoxData::Decoding => {
                if self.num_loaded_frames() >= 2 {
                    return JpegReconstructionStatus::Invalid;
                }
                let Some(frame) = self.frame(0) else {
                    return JpegReconstructionStatus::NeedMoreData;
                };
                let frame_header = frame.header();
                if frame_header.encoding != jxl_frame::header::Encoding::VarDct {
                    return JpegReconstructionStatus::Invalid;
                }
                if !frame_header.frame_type.is_normal_frame() {
                    return JpegReconstructionStatus::Invalid;
                }
                JpegReconstructionStatus::NeedMoreData
            }
            AuxBoxData::NotFound => JpegReconstructionStatus::Unavailable,
        }
    }

    /// Reconstructs JPEG bitstream and writes the image to writer.
    ///
    /// # Errors
    /// Returns an error if the reconstruction data is not available, incomplete or invalid, or
    /// if there was an error writing the image.
    ///
    /// Note that reconstruction may fail even if `jpeg_reconstruction_status` returned `Available`.
    pub fn reconstruct_jpeg(&self, jpeg_output: impl std::io::Write) -> Result<()> {
        let aux_boxes = &self.inner.aux_boxes;
        let jbrd = match aux_boxes.jbrd() {
            AuxBoxData::Data(jbrd) => jbrd,
            AuxBoxData::Decoding => {
                return Err(jxl_jbr::Error::ReconstructionDataIncomplete.into());
            }
            AuxBoxData::NotFound => {
                return Err(jxl_jbr::Error::ReconstructionUnavailable.into());
            }
        };
        if self.num_loaded_frames() == 0 {
            return Err(jxl_jbr::Error::FrameDataIncomplete.into());
        }

        let jbrd_header = jbrd.header();
        let expected_icc_len = jbrd_header.expected_icc_len();
        let expected_exif_len = jbrd_header.expected_exif_len();
        let expected_xmp_len = jbrd_header.expected_xmp_len();

        let icc = if expected_icc_len > 0 {
            self.original_icc().unwrap_or(&[])
        } else {
            &[]
        };

        let exif = if expected_exif_len > 0 {
            let b = aux_boxes.first_exif()?;
            b.map(|x| x.payload()).unwrap_or(&[])
        } else {
            &[]
        };

        let xmp = if expected_xmp_len > 0 {
            aux_boxes.first_xml().unwrap_or(&[])
        } else {
            &[]
        };

        let frame = self.frame(0).unwrap();
        jbrd.reconstruct(frame, icc, exif, xmp, &self.pool)?
            .write(jpeg_output)?;

        Ok(())
    }
}

/// # Private methods
impl JxlImage {
    fn convert_ec_info(&self) -> Vec<ExtraChannel> {
        self.image_header
            .metadata
            .ec_info
            .iter()
            .map(|ec_info| ExtraChannel {
                ty: ec_info.ty,
                name: ec_info.name.clone(),
                bit_depth: ec_info.bit_depth,
            })
            .collect()
    }
}

#[derive(Debug)]
struct JxlImageInner {
    end_of_image: bool,
    buffer: Vec<u8>,
    buffer_offset: usize,
    frame_offsets: Vec<usize>,
    aux_boxes: AuxBoxList,
}

impl JxlImageInner {
    fn feed_bytes_inner(&mut self, ctx: &mut RenderContext, mut buf: &[u8]) -> Result<()> {
        if buf.is_empty() {
            return Ok(());
        }

        if self.end_of_image {
            self.buffer.extend_from_slice(buf);
            return Ok(());
        }

        if let Some(loading_frame) = ctx.current_loading_frame() {
            debug_assert!(self.buffer.is_empty());
            let len = buf.len();
            buf = loading_frame.feed_bytes(buf)?;
            let count = len - buf.len();
            self.buffer_offset += count;

            if loading_frame.is_loading_done() {
                let is_last = loading_frame.header().is_last;
                ctx.finalize_current_frame();
                if is_last {
                    self.end_of_image = true;
                    self.buffer = buf.to_vec();
                    return Ok(());
                }
            }
            if buf.is_empty() {
                return Ok(());
            }
        }

        self.buffer.extend_from_slice(buf);
        let mut buf = &*self.buffer;
        while !buf.is_empty() {
            let mut bitstream = Bitstream::new(buf);
            let frame = match ctx.load_frame_header(&mut bitstream) {
                Ok(x) => x,
                Err(e) if e.unexpected_eof() => {
                    self.buffer = buf.to_vec();
                    return Ok(());
                }
                Err(e) => {
                    return Err(e.into());
                }
            };
            let frame_index = frame.index();
            assert_eq!(self.frame_offsets.len(), frame_index);
            self.frame_offsets.push(self.buffer_offset);

            let read_bytes = bitstream.num_read_bits() / 8;
            buf = &buf[read_bytes..];
            let len = buf.len();
            buf = frame.feed_bytes(buf)?;
            let read_bytes = read_bytes + (len - buf.len());
            self.buffer_offset += read_bytes;

            if frame.is_loading_done() {
                let is_last = frame.header().is_last;
                ctx.finalize_current_frame();
                if is_last {
                    self.end_of_image = true;
                    self.buffer = buf.to_vec();
                    return Ok(());
                }
            }
        }

        self.buffer.clear();
        Ok(())
    }
}

/// Pixel format of the rendered image.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum PixelFormat {
    /// Grayscale, single channel
    Gray,
    /// Grayscale with alpha, two channels
    Graya,
    /// RGB, three channels
    Rgb,
    /// RGB with alpha, four channels
    Rgba,
    /// CMYK, four channels
    Cmyk,
    /// CMYK with alpha, five channels
    Cmyka,
}

impl PixelFormat {
    /// Returns the number of channels of the image.
    #[inline]
    pub fn channels(self) -> usize {
        match self {
            PixelFormat::Gray => 1,
            PixelFormat::Graya => 2,
            PixelFormat::Rgb => 3,
            PixelFormat::Rgba => 4,
            PixelFormat::Cmyk => 4,
            PixelFormat::Cmyka => 5,
        }
    }

    /// Returns whether the image is grayscale.
    #[inline]
    pub fn is_grayscale(self) -> bool {
        matches!(self, Self::Gray | Self::Graya)
    }

    /// Returns whether the image has an alpha channel.
    #[inline]
    pub fn has_alpha(self) -> bool {
        matches!(
            self,
            PixelFormat::Graya | PixelFormat::Rgba | PixelFormat::Cmyka
        )
    }

    /// Returns whether the image has a black channel.
    #[inline]
    pub fn has_black(self) -> bool {
        matches!(self, PixelFormat::Cmyk | PixelFormat::Cmyka)
    }
}

/// HDR transfer function type, returned by [`JxlImage::hdr_type`].
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum HdrType {
    /// Perceptual quantizer.
    Pq,
    /// Hybrid log-gamma.
    Hlg,
}

/// The result of loading the keyframe.
#[derive(Debug)]
pub enum LoadResult {
    /// The frame is loaded with the given keyframe index.
    Done(usize),
    /// More data is needed to fully load the frame.
    NeedMoreData,
    /// No more frames are present.
    NoMoreFrames,
}

/// The result of loading and rendering the keyframe.
#[derive(Debug)]
pub enum RenderResult {
    /// The frame is rendered.
    Done(Render),
    /// More data is needed to fully render the frame.
    NeedMoreData,
    /// No more frames are present.
    NoMoreFrames,
}

/// The result of rendering a keyframe.
#[derive(Debug)]
pub struct Render {
    keyframe_index: usize,
    name: Name,
    duration: u32,
    orientation: u32,
    image: Arc<ImageWithRegion>,
    extra_channels: Vec<ExtraChannel>,
    target_frame_region: Region,
    color_bit_depth: BitDepth,
    is_cmyk: bool,
    render_spot_color: bool,
}

impl Render {
    /// Returns the keyframe index.
    #[inline]
    pub fn keyframe_index(&self) -> usize {
        self.keyframe_index
    }

    /// Returns the name of the frame.
    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns how many ticks this frame is presented.
    #[inline]
    pub fn duration(&self) -> u32 {
        self.duration
    }

    /// Returns the orientation of the image.
    #[inline]
    pub fn orientation(&self) -> u32 {
        self.orientation
    }

    /// Creates a stream that writes to borrowed buffer.
    ///
    /// The stream will include black and alpha channels, if exist, in addition to color channels.
    /// Orientation is applied.
    pub fn stream(&self) -> ImageStream {
        ImageStream::from_render(self, false)
    }

    /// Creates a stream that writes to borrowed buffer.
    ///
    /// The stream will include black channels if exist, but not alpha channels. Orientation is
    /// applied.
    pub fn stream_no_alpha(&self) -> ImageStream {
        ImageStream::from_render(self, true)
    }

    /// Creates a buffer with interleaved channels, with orientation applied.
    ///
    /// All extra channels are included. Use [`stream`](Render::stream) if only color, black and
    /// alpha channels are needed.
    #[inline]
    pub fn image_all_channels(&self) -> FrameBuffer {
        let fb: Vec<_> = self.image.buffer().iter().collect();
        let mut bit_depth = vec![self.color_bit_depth; self.image.color_channels()];
        for ec in &self.extra_channels {
            bit_depth.push(ec.bit_depth);
        }
        let regions: Vec<_> = self
            .image
            .regions_and_shifts()
            .iter()
            .map(|(region, _)| *region)
            .collect();

        FrameBuffer::from_grids(
            &fb,
            &bit_depth,
            &regions,
            self.target_frame_region,
            self.orientation,
        )
    }

    /// Creates a separate buffer by channel, with orientation applied.
    ///
    /// All extra channels are included.
    pub fn image_planar(&self) -> Vec<FrameBuffer> {
        let grids = self.image.buffer();
        let bit_depth_it = std::iter::repeat_n(self.color_bit_depth, self.image.color_channels())
            .chain(self.extra_channels.iter().map(|ec| ec.bit_depth));
        let region_it = self
            .image
            .regions_and_shifts()
            .iter()
            .map(|(region, _)| *region);

        bit_depth_it
            .zip(region_it)
            .zip(grids)
            .map(|((bit_depth, region), x)| {
                FrameBuffer::from_grids(
                    &[x],
                    &[bit_depth],
                    &[region],
                    self.target_frame_region,
                    self.orientation,
                )
            })
            .collect()
    }

    /// Returns the color channels.
    ///
    /// Orientation is not applied.
    #[inline]
    pub fn color_channels(&self) -> &[ImageBuffer] {
        let color_channels = self.image.color_channels();
        &self.image.buffer()[..color_channels]
    }

    /// Returns the extra channels, potentially including alpha and black channels.
    ///
    /// Orientation is not applied.
    #[inline]
    pub fn extra_channels(&self) -> (&[ExtraChannel], &[ImageBuffer]) {
        let color_channels = self.image.color_channels();
        (&self.extra_channels, &self.image.buffer()[color_channels..])
    }
}

/// Extra channel of the image.
#[derive(Debug)]
pub struct ExtraChannel {
    ty: ExtraChannelType,
    name: Name,
    bit_depth: BitDepth,
}

impl ExtraChannel {
    /// Returns the type of the extra channel.
    #[inline]
    pub fn ty(&self) -> ExtraChannelType {
        self.ty
    }

    /// Returns the name of the channel.
    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns `true` if the channel is a black channel of CMYK image.
    #[inline]
    pub fn is_black(&self) -> bool {
        matches!(self.ty, ExtraChannelType::Black)
    }

    /// Returns `true` if the channel is an alpha channel.
    #[inline]
    pub fn is_alpha(&self) -> bool {
        matches!(self.ty, ExtraChannelType::Alpha { .. })
    }

    /// Returns `true` if the channel is a spot colour channel.
    #[inline]
    pub fn is_spot_colour(&self) -> bool {
        matches!(self.ty, ExtraChannelType::SpotColour { .. })
    }
}

/// Cropping region information.
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq)]
pub struct CropInfo {
    pub width: u32,
    pub height: u32,
    pub left: u32,
    pub top: u32,
}

impl From<CropInfo> for jxl_render::Region {
    fn from(value: CropInfo) -> Self {
        Self {
            left: value.left as i32,
            top: value.top as i32,
            width: value.width,
            height: value.height,
        }
    }
}

impl From<jxl_render::Region> for CropInfo {
    fn from(value: jxl_render::Region) -> Self {
        Self {
            left: value.left.max(0) as u32,
            top: value.top.max(0) as u32,
            width: value.width,
            height: value.height,
        }
    }
}

/// Availability and validity of JPEG bitstream reconstruction data.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum JpegReconstructionStatus {
    /// JPEG bitstream reconstruction data is found. Actual reconstruction may or may not succeed.
    Available,
    /// Either JPEG bitstream reconstruction data or JPEG XL image data is invalid and cannot be
    /// used for actual reconstruction.
    Invalid,
    /// JPEG bitstream reconstruction data is not found. Result will *not* change.
    Unavailable,
    /// JPEG bitstream reconstruction data is not found. Result may change with more data.
    NeedMoreData,
}
