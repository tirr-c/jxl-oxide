//! jxl-oxide is a JPEG XL decoder written in pure Rust. It's internally organized into a few
//! small crates. This crate acts as a blanket and provides a simple interface made from those
//! crates to decode the actual image.
//!
//! # Decoding an image
//!
//! Decoding a JPEG XL image starts with constructing [`JxlImage`]. If you're reading a file, you
//! can use [`JxlImage::open`]:
//!
//! ```no_run
//! use jxl_oxide::JxlImage;
//!
//! let image = JxlImage::open("input.jxl").expect("Failed to read image header");
//! println!("{:?}", image.image_header()); // Prints the image header
//! ```
//!
//! Or, if you're reading from a reader that implements [`Read`][std::io::Read], you can use
//! [`JxlImage::from_reader`]:
//!
//! ```no_run
//! use jxl_oxide::JxlImage;
//!
//! # let reader = std::io::empty();
//! let image = JxlImage::from_reader(reader).expect("Failed to read image header");
//! println!("{:?}", image.image_header()); // Prints the image header
//! ```
//!
//! `JxlImage` parses the image header and embedded ICC profile (if there's any). Use
//! [`JxlImage::render_next_frame`], or [`JxlImage::load_next_frame`] followed by
//! [`JxlImage::render_frame`] to render the image. You might need to use
//! [`JxlImage::rendered_icc`] to do color management correctly.
//!
//! ```no_run
//! # use jxl_oxide::Render;
//! use jxl_oxide::{JxlImage, RenderResult};
//!
//! # fn present_image(_: Render) {}
//! # fn wait_for_data() {}
//! # fn main() -> jxl_oxide::Result<()> {
//! # let mut image = JxlImage::open("input.jxl").unwrap();
//! loop {
//!     let result = image.render_next_frame()?;
//!     match result {
//!         RenderResult::Done(render) => {
//!             present_image(render);
//!         },
//!         RenderResult::NeedMoreData => {
//!             wait_for_data();
//!         },
//!         RenderResult::NoMoreFrames => break,
//!     }
//! }
//! # Ok(())
//! # }
//! ```
use std::sync::Arc;

mod fb;

use bitstream::ContainerDetectingReader;
pub use jxl_bitstream as bitstream;
pub use jxl_color::header as color;
pub use jxl_image as image;
pub use jxl_frame::header as frame;

pub use jxl_bitstream::{Bitstream, Bundle};
use jxl_bitstream::Name;
pub use jxl_frame::{Frame, FrameHeader};
pub use jxl_grid::SimpleGrid;
pub use jxl_image::{ExtraChannelType, ImageHeader};
use jxl_render::RenderContext;

pub use fb::FrameBuffer;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync + 'static>>;

#[derive(Default)]
pub struct UninitializedJxlImage {
    reader: ContainerDetectingReader,
    buffer: Vec<u8>,
}

impl UninitializedJxlImage {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn feed_bytes(&mut self, buf: &[u8]) -> Result<()> {
        self.reader.feed_bytes(buf)?;
        self.buffer.extend(self.reader.take_bytes());
        Ok(())
    }

    pub fn try_init(mut self) -> Result<InitializeResult> {
        let mut bitstream = Bitstream::new(&self.buffer);
        let image_header = match ImageHeader::parse(&mut bitstream, ()) {
            Ok(x) => x,
            Err(e) if e.unexpected_eof() => {
                return Ok(InitializeResult::NeedMoreData(self));
            },
            Err(e) => {
                return Err(e.into());
            },
        };

        let embedded_icc = if image_header.metadata.colour_encoding.want_icc {
            tracing::debug!("Image has an embedded ICC profile");
            let icc = match jxl_color::icc::read_icc(&mut bitstream) {
                Ok(x) => x,
                Err(e) if e.unexpected_eof() => {
                    return Ok(InitializeResult::NeedMoreData(self));
                },
                Err(e) => {
                    return Err(e.into());
                },
            };
            let icc = jxl_color::icc::decode_icc(&icc)?;
            Some(icc)
        } else {
            None
        };
        bitstream.zero_pad_to_byte()?;

        let image_header = Arc::new(image_header);
        let skip_bytes = if image_header.metadata.preview.is_some() {
            let frame = match Frame::parse(&mut bitstream, image_header.clone()) {
                Ok(x) => x,
                Err(e) if e.unexpected_eof() => {
                    return Ok(InitializeResult::NeedMoreData(self));
                },
                Err(e) => {
                    return Err(e.into());
                },
            };

            let bytes_read = bitstream.num_read_bits() / 8;
            let x = frame.toc().total_byte_size() as usize;
            if self.buffer.len() < bytes_read + x {
                return Ok(InitializeResult::NeedMoreData(self));
            }

            x
        } else {
            0usize
        };

        let bytes_read = bitstream.num_read_bits() / 8 + skip_bytes;
        self.buffer.drain(..bytes_read);

        let render_spot_colour = !image_header.metadata.grayscale();

        let mut image = JxlImage {
            image_header: image_header.clone(),
            embedded_icc,
            ctx: RenderContext::new(image_header),
            render_spot_colour,
            end_of_image: false,
            buffer: Vec::new(),
        };
        image.feed_bytes(&self.buffer)?;

        Ok(InitializeResult::Initialized(image))
    }
}

pub enum InitializeResult {
    NeedMoreData(UninitializedJxlImage),
    Initialized(JxlImage),
}

/// JPEG XL image.
#[derive(Debug)]
pub struct JxlImage {
    image_header: Arc<ImageHeader>,
    embedded_icc: Option<Vec<u8>>,
    ctx: RenderContext,
    render_spot_colour: bool,
    end_of_image: bool,
    buffer: Vec<u8>,
}

impl JxlImage {
    pub fn new_uninit() -> UninitializedJxlImage {
        UninitializedJxlImage::new()
    }

    pub fn from_reader(mut reader: impl std::io::Read) -> Result<Self> {
        let mut uninit = Self::new_uninit();
        let mut buf = vec![0u8; 4096];
        let mut image = loop {
            let count = reader.read(&mut buf)?;
            if count == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "reader ended before parsing image header",
                ).into());
            }
            let buf = &buf[..count];
            uninit.feed_bytes(buf)?;

            match uninit.try_init()? {
                InitializeResult::NeedMoreData(x) => {
                    uninit = x;
                },
                InitializeResult::Initialized(x) => {
                    break x;
                },
            }
        };

        while !image.end_of_image {
            let count = reader.read(&mut buf)?;
            if count == 0 {
                break;
            }
            let buf = &buf[..count];
            image.feed_bytes(buf)?;
        }

        Ok(image)
    }

    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        Self::from_reader(file)
    }
}

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

    /// Returns the embedded ICC profile.
    ///
    /// It does *not* describe the colorspace of rendered images. Use
    /// [`rendered_icc`][Self::rendered_icc] to do color management.
    #[inline]
    pub fn embedded_icc(&self) -> Option<&[u8]> {
        self.embedded_icc.as_deref()
    }

    /// Returns the ICC profile that describes rendered images.
    ///
    /// - If the image is XYB encoded, and the ICC profile is embedded, then the profile describes
    ///   linear sRGB or linear grayscale colorspace.
    /// - Else, if the ICC profile is embedded, then the embedded profile is returned.
    /// - Else, the profile describes the color encoding signalled in the image header.
    pub fn rendered_icc(&self) -> Vec<u8> {
        create_rendered_icc(&self.image_header.metadata, self.embedded_icc.as_deref())
    }

    /// Returns the pixel format of the rendered image.
    pub fn pixel_format(&self) -> PixelFormat {
        let is_grayscale = self.image_header.metadata.grayscale();
        let mut has_black = false;
        let mut has_alpha = false;
        for ec_info in &self.image_header.metadata.ec_info {
            if ec_info.is_alpha() {
                has_alpha = true;
            }
            if ec_info.is_black() {
                has_black = true;
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

    /// Sets whether the spot colour channels will be rendered.
    #[inline]
    pub fn set_render_spot_colour(&mut self, render_spot_colour: bool) -> &mut Self {
        if render_spot_colour && self.image_header.metadata.grayscale() {
            tracing::warn!("Spot colour channels are not rendered on grayscale images");
            return self;
        }
        self.render_spot_colour = render_spot_colour;
        self
    }

    /// Returns whether the spot color channels will be rendered.
    #[inline]
    pub fn render_spot_colour(&self) -> bool {
        self.render_spot_colour
    }
}

impl JxlImage {
    pub fn feed_bytes(&mut self, mut buf: &[u8]) -> Result<()> {
        if self.end_of_image {
            self.buffer.extend_from_slice(buf);
            return Ok(());
        }

        if let Some(loading_frame) = self.ctx.current_loading_frame() {
            debug_assert!(self.buffer.is_empty());
            buf = loading_frame.feed_bytes(buf);
            if loading_frame.is_loading_done() {
                let is_last = loading_frame.header().is_last;
                self.ctx.finalize_current_frame();
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
            let frame = match self.ctx.load_frame_header(&mut bitstream) {
                Ok(x) => x,
                Err(e) if e.unexpected_eof() => {
                    self.buffer = buf.to_vec();
                    return Ok(());
                },
                Err(e) => {
                    return Err(e.into());
                },
            };
            let read_bytes = bitstream.num_read_bits() / 8;
            buf = &buf[read_bytes..];
            buf = frame.feed_bytes(buf);

            if frame.is_loading_done() {
                let is_last = frame.header().is_last;
                self.ctx.finalize_current_frame();
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

    pub fn try_take_buffer(&mut self) -> Option<Vec<u8>> {
        if self.end_of_image {
            Some(std::mem::take(&mut self.buffer))
        } else {
            None
        }
    }

    #[inline]
    pub fn is_loading_done(&self) -> bool {
        self.end_of_image
    }

    /// Returns the frame header for the given keyframe index, or `None` if the keyframe does not
    /// exist.
    pub fn frame_header(&self, keyframe_index: usize) -> Option<&FrameHeader> {
        let frame = self.ctx.keyframe(keyframe_index)?;
        Some(frame.header())
    }

    /// Returns the number of currently loaded keyframes.
    pub fn num_loaded_keyframes(&self) -> usize {
        self.ctx.loaded_keyframes()
    }

    /// Renders the given keyframe with optional cropping region.
    pub fn render_frame_cropped(&mut self, keyframe_index: usize, image_region: Option<CropInfo>) -> Result<Render> {
        let mut grids = self.ctx.render_keyframe(keyframe_index, image_region.map(From::from))?;
        let mut grids = grids.take_buffer();

        let color_channels = if self.image_header.metadata.grayscale() { 1 } else { 3 };
        let mut color_channels: Vec<_> = grids.drain(..color_channels).collect();
        let extra_channels: Vec<_> = grids
            .into_iter()
            .zip(&self.image_header.metadata.ec_info)
            .map(|(grid, ec_info)| ExtraChannel {
                ty: ec_info.ty,
                name: ec_info.name.clone(),
                grid,
            })
        .collect();

        if self.render_spot_colour {
            for ec in &extra_channels {
                if ec.is_spot_colour() {
                    jxl_render::render_spot_colour(&mut color_channels, &ec.grid, &ec.ty)?;
                }
            }
        }

        let frame = self.ctx.keyframe(keyframe_index).unwrap();
        let frame_header = frame.header();
        let result = Render {
            keyframe_index,
            name: frame_header.name.clone(),
            duration: frame_header.duration,
            orientation: self.image_header.metadata.orientation,
            color_channels,
            extra_channels,
        };

        self.end_of_image = frame_header.is_last;
        Ok(result)
    }

    /// Renders the given keyframe.
    pub fn render_frame(&mut self, keyframe_index: usize) -> Result<Render> {
        self.render_frame_cropped(keyframe_index, None)
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum PixelFormat {
    Gray,
    Graya,
    Rgb,
    Rgba,
    Cmyk,
    Cmyka,
}

impl PixelFormat {
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

    #[inline]
    pub fn has_alpha(self) -> bool {
        matches!(self, PixelFormat::Graya | PixelFormat::Rgba | PixelFormat::Cmyka)
    }

    #[inline]
    pub fn has_black(self) -> bool {
        matches!(self, PixelFormat::Cmyk | PixelFormat::Cmyka)
    }
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
    color_channels: Vec<SimpleGrid<f32>>,
    extra_channels: Vec<ExtraChannel>,
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

    /// Creates a buffer with interleaved channels, with orientation applied.
    ///
    /// Extra channels other than black and alpha are not included.
    #[inline]
    pub fn image(&self) -> FrameBuffer {
        let mut fb: Vec<_> = self.color_channels.clone();

        // Find black
        for ec in &self.extra_channels {
            if ec.is_black() {
                fb.push(ec.grid.clone());
                break;
            }
        }
        // Find alpha
        for ec in &self.extra_channels {
            if ec.is_alpha() {
                fb.push(ec.grid.clone());
                break;
            }
        }

        FrameBuffer::from_grids(&fb, self.orientation)
    }

    /// Creates a separate buffer by channel, with orientation applied.
    ///
    /// All extra channels are included.
    pub fn image_planar(&self) -> Vec<FrameBuffer> {
        self.color_channels
            .iter()
            .chain(self.extra_channels.iter().map(|x| &x.grid))
            .map(|x| FrameBuffer::from_grids(std::slice::from_ref(x), self.orientation))
            .collect()
    }

    /// Returns the color channels.
    ///
    /// Orientation is not applied.
    #[inline]
    pub fn color_channels(&self) -> &[SimpleGrid<f32>] {
        &self.color_channels
    }

    /// Returns the mutable slice to the color channels.
    ///
    /// Orientation is not applied.
    #[inline]
    pub fn color_channels_mut(&mut self) -> &mut [SimpleGrid<f32>] {
        &mut self.color_channels
    }

    /// Returns the extra channels, potentially including alpha and black channels.
    ///
    /// Orientation is not applied.
    #[inline]
    pub fn extra_channels(&self) -> &[ExtraChannel] {
        &self.extra_channels
    }

    /// Returns the mutable slice to the extra channels, potentially including alpha and black
    /// channels.
    ///
    /// Orientation is not applied.
    #[inline]
    pub fn extra_channels_mut(&mut self) -> &mut [ExtraChannel] {
        &mut self.extra_channels
    }
}

/// Extra channel of the image.
#[derive(Debug)]
pub struct ExtraChannel {
    ty: ExtraChannelType,
    name: Name,
    grid: SimpleGrid<f32>,
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

    /// Returns the sample grid of the channel.
    #[inline]
    pub fn grid(&self) -> &SimpleGrid<f32> {
        &self.grid
    }

    /// Returns the mutable sample grid of the channel.
    #[inline]
    pub fn grid_mut(&mut self) -> &mut SimpleGrid<f32> {
        &mut self.grid
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
#[derive(Debug, Default, Copy, Clone)]
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

fn create_rendered_icc(metadata: &image::ImageMetadata, embedded_icc: Option<&[u8]>) -> Vec<u8> {
    if !metadata.xyb_encoded {
        if let Some(icc) = embedded_icc {
            return icc.to_vec();
        }
    }

    jxl_color::icc::colour_encoding_to_icc(&metadata.colour_encoding)
}
