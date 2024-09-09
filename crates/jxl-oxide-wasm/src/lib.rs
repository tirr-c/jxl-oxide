use jxl_oxide::{
    color::ColourEncoding, EnumColourEncoding, InitializeResult, JxlImage, PixelFormat, Render,
    RenderingIntent, UninitializedJxlImage,
};
use wasm_bindgen::prelude::*;

#[cfg(feature = "dev")]
#[wasm_bindgen(start)]
fn start() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    console_log::init().ok();
}

#[wasm_bindgen(js_name = JxlImage)]
pub struct WasmJxlImage {
    inner: WasmJxlImageInner,
    force_srgb: bool,
}

enum WasmJxlImageInner {
    Uninit(UninitializedJxlImage),
    Init(JxlImage),
}

impl Default for WasmJxlImage {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "web")]
fn is_hdr_supported() -> bool {
    let global_scope = web_sys::js_sys::global();
    let user_agent = if let Some(window) = global_scope.dyn_ref::<web_sys::Window>() {
        window.navigator().user_agent()
    } else if let Some(window) = global_scope.dyn_ref::<web_sys::WorkerGlobalScope>() {
        window.navigator().user_agent()
    } else {
        return true;
    };

    let Ok(user_agent) = user_agent else {
        return true;
    };

    !user_agent.contains(" Firefox/")
}

#[cfg(not(feature = "web"))]
fn is_hdr_supported() -> bool {
    true
}

/// Return the version of jxl-oxide-wasm.
#[wasm_bindgen]
pub fn version() -> String {
    String::from(env!("CARGO_PKG_VERSION"))
}

#[wasm_bindgen(js_class = JxlImage)]
impl WasmJxlImage {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        let inner = WasmJxlImageInner::Uninit(JxlImage::builder().build_uninit());
        Self {
            inner,
            force_srgb: !is_hdr_supported(),
        }
    }

    #[wasm_bindgen(getter = forceSrgb)]
    pub fn force_srgb(&self) -> bool {
        self.force_srgb
    }

    #[wasm_bindgen(setter = forceSrgb)]
    pub fn set_force_srgb(&mut self, force_srgb: bool) {
        self.force_srgb = force_srgb;
    }

    #[wasm_bindgen(js_name = feedBytes)]
    pub fn feed_bytes(&mut self, bytes: &[u8]) -> Result<(), String> {
        match &mut self.inner {
            WasmJxlImageInner::Uninit(image) => image.feed_bytes(bytes),
            WasmJxlImageInner::Init(image) => image.feed_bytes(bytes),
        }
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    #[wasm_bindgen(js_name = tryInit)]
    pub fn try_init(&mut self) -> Result<bool, String> {
        if let WasmJxlImageInner::Uninit(image) = &mut self.inner {
            let uninit_image = std::mem::replace(image, JxlImage::builder().build_uninit());
            let result = uninit_image.try_init().map_err(|e| e.to_string())?;
            match result {
                InitializeResult::NeedMoreData(uninit_image) => {
                    *image = uninit_image;
                    Ok(false)
                }
                InitializeResult::Initialized(mut image) => {
                    let tagged_color_encoding = &image.image_header().metadata.colour_encoding;
                    let xyb_encoded = image.image_header().metadata.xyb_encoded;
                    if let ColourEncoding::Enum(color) = tagged_color_encoding {
                        if xyb_encoded {
                            if self.force_srgb {
                                image.request_color_encoding(EnumColourEncoding::srgb(
                                    RenderingIntent::Relative,
                                ));
                            } else if color.is_hdr() {
                                image.request_color_encoding(EnumColourEncoding::bt2100_pq(
                                    RenderingIntent::Perceptual,
                                ));
                            }
                        }
                    }
                    self.inner = WasmJxlImageInner::Init(image);
                    Ok(true)
                }
            }
        } else {
            Ok(true)
        }
    }

    #[wasm_bindgen(getter = loaded)]
    pub fn is_loading_done(&self) -> bool {
        match &self.inner {
            WasmJxlImageInner::Uninit(_) => false,
            WasmJxlImageInner::Init(image) => image.is_loading_done(),
        }
    }

    #[wasm_bindgen(getter = numLoadedKeyframes)]
    pub fn num_loaded_keyframes(&self) -> u32 {
        match &self.inner {
            WasmJxlImageInner::Uninit(_) => 0,
            WasmJxlImageInner::Init(image) => image.num_loaded_keyframes() as u32,
        }
    }

    #[wasm_bindgen(getter = animated)]
    pub fn is_animation(&self) -> Option<bool> {
        match &self.inner {
            WasmJxlImageInner::Uninit(_) => None,
            WasmJxlImageInner::Init(image) => {
                Some(image.image_header().metadata.animation.is_some())
            }
        }
    }

    #[wasm_bindgen(getter = numLoops)]
    pub fn num_loops(&self) -> Option<u32> {
        match &self.inner {
            WasmJxlImageInner::Uninit(_) => None,
            WasmJxlImageInner::Init(image) => Some(
                image
                    .image_header()
                    .metadata
                    .animation
                    .as_ref()
                    .map(|anim| anim.num_loops)
                    .unwrap_or(0),
            ),
        }
    }

    pub fn render(&mut self, keyframe_idx: Option<u32>) -> Result<RenderResult, String> {
        let image = match &mut self.inner {
            WasmJxlImageInner::Uninit(_) => return Err(String::from("image not initialized")),
            WasmJxlImageInner::Init(image) => image,
        };
        let (tps_numer, tps_denom) =
            if let Some(animation) = &image.image_header().metadata.animation {
                (animation.tps_numerator, animation.tps_denominator)
            } else {
                (0, 0)
            };

        let frame_idx = keyframe_idx.unwrap_or(0) as usize;
        let frame = if image.num_loaded_keyframes() == frame_idx {
            image.render_loading_frame()
        } else {
            image.render_frame(frame_idx)
        }
        .map_err(|e| e.to_string())?;

        let duration = frame.duration();
        let (frame_duration_numer, frame_duration_denom) = if frame.duration() == 0xffffffff {
            (duration, 0)
        } else {
            (tps_numer * duration, tps_denom)
        };

        let pixfmt = image.pixel_format();
        let cicp = image.rendered_cicp();
        let icc = if cicp.is_some() {
            Vec::new()
        } else {
            image.rendered_icc()
        };

        let has_embedded_icc = image.original_icc().is_some();
        let metadata = &image.image_header().metadata;
        let bit_depth = metadata.bit_depth.bits_per_sample();
        let need_high_precision = !metadata.xyb_encoded && has_embedded_icc && bit_depth > 8;
        Ok(RenderResult {
            image: frame,
            pixfmt,
            need_high_precision,
            icc,
            cicp,
            frame_duration_numer,
            frame_duration_denom,
        })
    }
}

#[wasm_bindgen]
pub struct RenderResult {
    image: Render,
    pixfmt: PixelFormat,
    need_high_precision: bool,
    icc: Vec<u8>,
    cicp: Option<[u8; 4]>,
    frame_duration_numer: u32,
    frame_duration_denom: u32,
}

#[wasm_bindgen]
impl RenderResult {
    #[wasm_bindgen(getter = durationNumerator)]
    pub fn frame_duration_numer(&self) -> u32 {
        self.frame_duration_numer
    }

    #[wasm_bindgen(getter = durationDenominator)]
    pub fn frame_duration_denom(&self) -> u32 {
        self.frame_duration_denom
    }

    #[wasm_bindgen(getter = duration)]
    pub fn frame_duration(&self) -> f64 {
        if self.frame_duration_denom == 0 {
            if self.frame_duration_numer == 0 {
                0.0
            } else {
                f64::INFINITY
            }
        } else {
            self.frame_duration_numer as f64 / self.frame_duration_denom as f64
        }
    }

    #[wasm_bindgen(getter = iccProfile)]
    pub fn icc_profile(&self) -> Vec<u8> {
        self.icc.clone()
    }

    #[wasm_bindgen(js_name = encodeToPng)]
    pub fn into_png(self) -> Result<Vec<u8>, String> {
        let image = self.image;
        let mut stream = image.stream();
        let mut fb = jxl_oxide::FrameBuffer::new(
            stream.width() as usize,
            stream.height() as usize,
            stream.channels() as usize,
        );
        stream.write_to_buffer(fb.buf_mut());

        let mut out = Vec::new();
        let mut encoder = png::Encoder::new(&mut out, stream.width(), stream.height());
        let color = match self.pixfmt {
            PixelFormat::Gray => png::ColorType::Grayscale,
            PixelFormat::Graya => png::ColorType::GrayscaleAlpha,
            PixelFormat::Rgb => png::ColorType::Rgb,
            PixelFormat::Rgba => png::ColorType::Rgba,
            PixelFormat::Cmyk | PixelFormat::Cmyka => {
                return Err(String::from("unsupported colorspace"))
            }
        };
        let depth = if self.need_high_precision {
            png::BitDepth::Sixteen
        } else {
            png::BitDepth::Eight
        };
        encoder.set_color(color);
        encoder.set_depth(depth);

        let mut writer = encoder.write_header().map_err(|e| e.to_string())?;

        if let Some(cicp) = self.cicp {
            writer
                .write_chunk(png::chunk::ChunkType([b'c', b'I', b'C', b'P']), &cicp)
                .map_err(|e| e.to_string())?;
        } else if !self.icc.is_empty() {
            let compressed_icc = miniz_oxide::deflate::compress_to_vec_zlib(&self.icc, 7);
            let mut iccp_chunk_data = vec![b'0', 0, 0];
            iccp_chunk_data.extend(compressed_icc);
            writer
                .write_chunk(png::chunk::iCCP, &iccp_chunk_data)
                .map_err(|e| e.to_string())?;
        }

        if self.need_high_precision {
            let mut buf = vec![0u8; fb.width() * fb.height() * fb.channels() * 2];
            for (b, s) in buf.chunks_exact_mut(2).zip(fb.buf()) {
                let w = (*s * 65535.0 + 0.5).clamp(0.0, 65535.0) as u16;
                let [b0, b1] = w.to_be_bytes();
                b[0] = b0;
                b[1] = b1;
            }
            writer.write_image_data(&buf).map_err(|e| e.to_string())?;
        } else {
            let mut buf = vec![0u8; fb.width() * fb.height() * fb.channels()];
            for (b, s) in buf.iter_mut().zip(fb.buf()) {
                *b = (*s * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            }
            writer.write_image_data(&buf).map_err(|e| e.to_string())?;
        }

        writer.finish().map_err(|e| e.to_string())?;
        Ok(out)
    }
}
