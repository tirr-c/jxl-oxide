use std::io::prelude::*;
use std::path::Path;

use jxl_oxide::{JxlImage, JxlThreadPool, Render};

use crate::commands::progressive::*;
use crate::{Error, Result};

#[cfg(feature = "__ffmpeg")]
mod mp4;
mod png_seq;

enum OutputType {
    PngSequence(png_seq::Context),
    #[cfg(feature = "__ffmpeg")]
    Mp4(mp4::Context),
}

impl OutputType {
    fn png_sequence(output_dir: impl AsRef<Path>) -> Result<Self> {
        png_seq::Context::new(output_dir).map(Self::PngSequence)
    }

    #[cfg(feature = "__ffmpeg")]
    fn mp4(output_path: impl AsRef<Path>, font: &str) -> Result<Self> {
        mp4::Context::new(output_path, font).map(Self::Mp4)
    }
}

impl OutputType {
    fn prepare_image(&self, image: &mut JxlImage) {
        #[cfg(feature = "__ffmpeg")]
        use jxl_oxide::{EnumColourEncoding, RenderingIntent};

        match self {
            Self::PngSequence(_) => {
                let _ = image;
            }
            #[cfg(feature = "__ffmpeg")]
            Self::Mp4(_) => {
                if image.is_hdr() {
                    image.request_color_encoding(EnumColourEncoding::bt2100_pq(
                        RenderingIntent::Relative,
                    ));
                } else {
                    image.request_color_encoding(EnumColourEncoding::srgb(
                        RenderingIntent::Relative,
                    ));
                }
            }
        }
    }

    fn add_empty_frame(
        &mut self,
        image: &JxlImage,
        description: impl std::fmt::Display,
    ) -> Result<()> {
        match self {
            OutputType::PngSequence(ctx) => ctx.add_empty_frame(image, description),
            #[cfg(feature = "__ffmpeg")]
            OutputType::Mp4(ctx) => ctx.add_empty_frame(image, description),
        }
    }

    fn add_frame(
        &mut self,
        image: &JxlImage,
        render: &Render,
        description: impl std::fmt::Display,
    ) -> Result<()> {
        match self {
            OutputType::PngSequence(ctx) => ctx.add_frame(image, render, description),
            #[cfg(feature = "__ffmpeg")]
            OutputType::Mp4(ctx) => ctx.add_frame(image, render, description),
        }
    }

    fn finalize(&mut self) -> Result<()> {
        match self {
            OutputType::PngSequence(_) => Ok(()),
            #[cfg(feature = "__ffmpeg")]
            OutputType::Mp4(ctx) => ctx.finalize(),
        }
    }
}

struct LoadProgress {
    iter: usize,
    unit_step: usize,
    current_step: usize,
    bytes_read: usize,
    total_bytes: usize,
}

impl LoadProgress {
    fn new(unit_step: usize, total_bytes: usize) -> Self {
        Self {
            iter: 0,
            unit_step,
            current_step: unit_step,
            bytes_read: 0,
            total_bytes,
        }
    }

    #[inline]
    fn add_iter(&mut self, bytes_read: usize) {
        self.bytes_read += bytes_read;
        self.iter += 1;
    }

    #[inline]
    fn double_step(&mut self) -> usize {
        self.current_step = (self.current_step * 2).min(self.step_cap());
        self.current_step
    }

    #[inline]
    fn iter(&self) -> usize {
        self.iter
    }

    #[inline]
    fn current_step(&self) -> usize {
        self.current_step
    }

    #[inline]
    fn step_cap(&self) -> usize {
        self.total_bytes.div_ceil(self.unit_step * 100) * self.unit_step
    }

    #[inline]
    fn progress(&self) -> f64 {
        self.bytes_read as f64 / self.total_bytes as f64
    }
}

impl std::fmt::Display for LoadProgress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let step = self.current_step;
        if self.total_bytes > 0 {
            let percentage = self.progress() * 100.0;
            write!(f, "{percentage:.2}\\% loaded ({step} bytes/frame)")
        } else {
            write!(f, "{step} bytes/frame")
        }
    }
}

pub fn handle_progressive(args: ProgressiveArgs) -> Result<()> {
    let _guard = tracing::trace_span!("Handle progressive subcommand").entered();

    #[cfg(feature = "rayon")]
    let pool = JxlThreadPool::rayon(args.num_threads);
    #[cfg(not(feature = "rayon"))]
    let pool = JxlThreadPool::none();

    let mut uninit_image = JxlImage::builder().pool(pool.clone()).build_uninit();

    let mut input = std::fs::File::open(&args.input).map_err(|e| Error::ReadJxl(e.into()))?;
    let total_bytes = input.metadata().map(|meta| meta.len()).unwrap_or(0);

    let mut progress = LoadProgress::new(args.step as usize, total_bytes as usize);
    let mut buf = vec![0u8; progress.current_step()];

    let output_dir = args.output.as_deref();
    let mut output_ctx = match output_dir {
        Some(path) => Some(match path.extension() {
            #[cfg(feature = "__ffmpeg")]
            Some(ext) if ext == "mp4" => OutputType::mp4(path, &args.font)?,
            #[cfg(not(feature = "__ffmpeg"))]
            Some(ext) if ext == "mp4" => {
                tracing::warn!("FFmpeg support is not active; will output as PNG sequence");
                OutputType::png_sequence(path)?
            }
            _ => OutputType::png_sequence(path)?,
        }),
        None => {
            tracing::info!("No output path specified, skipping output encoding");
            None
        }
    };

    let mut empty_frame_desc = Vec::<String>::new();
    let mut image = loop {
        let current_iter = progress.iter();

        let buf = fill_buf(&mut input, &mut buf)?;
        if buf.is_empty() {
            tracing::warn!("Partial image");
            return Ok(());
        }
        progress.add_iter(buf.len());

        uninit_image.feed_bytes(buf).map_err(Error::ReadJxl)?;
        let init_result = uninit_image.try_init().map_err(Error::ReadJxl)?;
        match init_result {
            jxl_oxide::InitializeResult::NeedMoreData(image) => uninit_image = image,
            jxl_oxide::InitializeResult::Initialized(mut image) => {
                if let Some(output) = &mut output_ctx {
                    output.prepare_image(&mut image);

                    for mut description in empty_frame_desc {
                        description.push_str("\nno frames loaded");
                        output.add_empty_frame(&image, description)?;
                    }
                }
                run_once(&mut image, &progress, output_ctx.as_mut())?;
                break image;
            }
        }

        empty_frame_desc.push(progress.to_string());
        tracing::info!("Iteration {current_iter} didn't produce an image");
    };

    loop {
        let current_iter = progress.iter();
        if current_iter >= 120 && current_iter % 60 == 0 {
            let prev_step = progress.current_step();
            let new_step = progress.double_step();
            if prev_step != new_step {
                buf.resize(new_step, 0);
                tracing::info!(new_step, "Increasing step size");
            }
        }

        let buf = fill_buf(&mut input, &mut buf)?;
        if buf.is_empty() {
            break;
        }
        progress.add_iter(buf.len());

        image.feed_bytes(buf).map_err(Error::ReadJxl)?;
        run_once(&mut image, &progress, output_ctx.as_mut())?;
    }

    if !image.is_loading_done() {
        tracing::warn!("Partial image");
    }

    if let Some(output_ctx) = &mut output_ctx {
        output_ctx.finalize()?;
    } else {
        tracing::info!("No output path specified, skipping output encoding");
    }

    Ok(())
}

fn run_once(
    image: &mut JxlImage,
    progress: &LoadProgress,
    output_ctx: Option<&mut OutputType>,
) -> Result<()> {
    match (image.num_loaded_keyframes(), image.is_loading_done()) {
        (0, false) | (1, true) => {}
        _ => {
            tracing::error!("Progressive decoding doesn't support animation");
            return Err(Error::ReadJxl(
                "Progressive decoding doesn't support animation".into(),
            ));
        }
    }

    let current_iter = progress.iter() - 1;

    let loaded_frames = image.num_loaded_frames();
    let last_frame = image.frame(loaded_frames).or_else(|| {
        loaded_frames
            .checked_sub(1)
            .and_then(|idx| image.frame(idx))
    });
    let load_position_str = if let Some(frame) = last_frame {
        let idx = frame.index();
        if let Some(current_group) = frame.current_loading_group() {
            format!("frame #{idx} decoding {:?}", current_group.kind)
        } else {
            format!("frame #{idx} fully loaded")
        }
    } else {
        String::from("no frames loaded")
    };

    fn run_inner(image: &mut JxlImage) -> Result<Option<Render>> {
        Ok(if image.is_loading_done() {
            Some(image.render_frame_cropped(0).map_err(Error::Render)?)
        } else {
            image.render_loading_frame_cropped().ok()
        })
    }

    let mut output = None;

    let decode_start = std::time::Instant::now();
    #[cfg(feature = "rayon")]
    if let Some(rayon_pool) = image.pool().clone().as_rayon_pool() {
        output = rayon_pool.install(|| run_inner(image))?;
    }

    if output.is_none() {
        output = run_inner(image)?;
    }

    let elapsed = decode_start.elapsed();
    let elapsed_seconds = elapsed.as_secs_f64();
    let elapsed_msecs = elapsed_seconds * 1000.0;

    let Some(render) = output else {
        tracing::info!(
            "Iteration {current_iter} took {elapsed_msecs:.2} ms; didn't produce an image"
        );
        if let Some(output_ctx) = output_ctx {
            output_ctx.add_empty_frame(image, format_args!("{progress}\n{load_position_str}"))?;
        }
        return Ok(());
    };

    tracing::info!("Iteration {current_iter} took {elapsed_msecs:.2} ms");
    if let Some(output_ctx) = output_ctx {
        output_ctx.add_frame(
            image,
            &render,
            format_args!("{progress}\n{load_position_str}"),
        )?;
    }

    Ok(())
}

fn fill_buf<R: Read>(mut reader: R, buf: &mut [u8]) -> Result<&mut [u8]> {
    let mut bytes_read = 0;
    loop {
        let cnt = reader
            .read(&mut buf[bytes_read..])
            .map_err(|e| Error::ReadJxl(e.into()))?;
        if cnt == 0 {
            break;
        }
        bytes_read += cnt;
    }

    Ok(&mut buf[..bytes_read])
}
