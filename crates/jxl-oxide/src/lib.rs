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
use std::{
    fs::File,
    io::Read,
    path::Path,
    sync::Arc,
};

mod fb;

pub use jxl_bitstream as bitstream;
pub use jxl_color::header as color;
pub use jxl_image as image;
pub use jxl_frame::header as frame;

pub use jxl_bitstream::{Bitstream, Bundle};
use jxl_bitstream::{ContainerDetectingReader, Name};
pub use jxl_frame::{Frame, FrameHeader};
pub use jxl_grid::SimpleGrid;
pub use jxl_image::{ExtraChannelType, ImageHeader};
use jxl_render::RenderContext;

pub use fb::FrameBuffer;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync + 'static>>;

/// JPEG XL image.
#[derive(Debug)]
pub struct JxlImage<R> {
    bitstream: Bitstream<ContainerDetectingReader<R>>,
    image_header: Arc<ImageHeader>,
    embedded_icc: Option<Vec<u8>>,
    ctx: RenderContext,
    render_spot_colour: bool,
    end_of_image: bool,
}

impl<R: Read> JxlImage<R> {
    /// Creates a `JxlImage` from the reader.
    pub fn from_reader(reader: R) -> Result<Self> {
        let mut bitstream = Bitstream::new_detect(reader);
        let image_header = Arc::new(ImageHeader::parse(&mut bitstream, ())?);

        let embedded_icc = image_header.metadata.colour_encoding.want_icc.then(|| {
            tracing::debug!("Image has an embedded ICC profile");
            let icc = jxl_color::icc::read_icc(&mut bitstream)?;
            jxl_color::icc::decode_icc(&icc)
        }).transpose()?;

        if image_header.metadata.preview.is_some() {
            tracing::debug!("Skipping preview frame");
            bitstream.zero_pad_to_byte()?;

            let frame = Frame::parse(&mut bitstream, image_header.clone())?;
            let toc = frame.toc();
            let bookmark = toc.bookmark() + (toc.total_byte_size() * 8);
            bitstream.skip_to_bookmark(bookmark)?;
        }

        let render_spot_colour = !image_header.metadata.grayscale();

        Ok(Self {
            bitstream,
            image_header: image_header.clone(),
            embedded_icc,
            ctx: RenderContext::new(image_header),
            render_spot_colour,
            end_of_image: false,
        })
    }
}

impl JxlImage<File> {
    /// Creates a `JxlImage` from the filesystem.
    #[inline]
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        Self::from_reader(file)
    }
}

impl<R> JxlImage<R> {
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

impl<R: Read> JxlImage<R> {
    /// Loads the next keyframe, and returns the result with the keyframe index.
    ///
    /// Unlike [`render_next_frame`][Self::render_next_frame], this method does not render the
    /// loaded frame. Use [`render_frame`][Self::render_frame] to render the loaded frame.
    pub fn load_next_frame(&mut self) -> Result<LoadResult> {
        if self.end_of_image {
            return Ok(LoadResult::NoMoreFrames);
        }

        self.ctx.load_until_keyframe(&mut self.bitstream)?;

        let keyframe_index = self.ctx.loaded_keyframes() - 1;
        self.end_of_image = self.frame_header(keyframe_index).unwrap().is_last;
        Ok(LoadResult::Done(keyframe_index))
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

    /// Loads and renders the next keyframe with optional cropping region.
    pub fn render_next_frame_cropped(&mut self, image_region: Option<CropInfo>) -> Result<RenderResult> {
        let load_result = self.load_next_frame()?;
        match load_result {
            LoadResult::Done(keyframe_index) => {
                let render = self.render_frame_cropped(keyframe_index, image_region)?;
                Ok(RenderResult::Done(render))
            },
            LoadResult::NeedMoreData => Ok(RenderResult::NeedMoreData),
            LoadResult::NoMoreFrames => Ok(RenderResult::NoMoreFrames),
        }
    }

    /// Loads and renders the next keyframe.
    pub fn render_next_frame(&mut self) -> Result<RenderResult> {
        self.render_next_frame_cropped(None)
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
