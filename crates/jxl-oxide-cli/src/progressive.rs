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

pub fn handle_progressive(args: ProgressiveArgs) -> Result<()> {
    let _guard = tracing::trace_span!("Handle progressive subcommand").entered();

    #[cfg(feature = "rayon")]
    let pool = JxlThreadPool::rayon(args.num_threads);
    #[cfg(not(feature = "rayon"))]
    let pool = JxlThreadPool::none();

    let mut uninit_image = JxlImage::builder().pool(pool.clone()).build_uninit();

    let mut input = std::fs::File::open(&args.input).map_err(|e| Error::ReadJxl(e.into()))?;
    let total_bytes = input.metadata().map(|meta| meta.len()).unwrap_or(0);

    let init_step = args.step as usize;
    let step_cap = total_bytes.div_ceil(args.step * 100) as usize * init_step;

    let mut step = init_step;
    let mut buf = vec![0u8; step];
    let mut idx = 0usize;
    let mut bytes_read = 0usize;

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

    let mut image = loop {
        let current_iter = idx;

        let buf = fill_buf(&mut input, &mut buf)?;
        if buf.is_empty() {
            tracing::warn!("Partial image");
            return Ok(());
        }
        bytes_read += buf.len();

        uninit_image.feed_bytes(buf).map_err(Error::ReadJxl)?;
        idx += 1;
        let init_result = uninit_image.try_init().map_err(Error::ReadJxl)?;
        match init_result {
            jxl_oxide::InitializeResult::NeedMoreData(image) => uninit_image = image,
            jxl_oxide::InitializeResult::Initialized(mut image) => {
                if let Some(output) = &mut output_ctx {
                    output.prepare_image(&mut image);

                    let step = args.step as usize;
                    for idx in 0..current_iter {
                        let mut description = progress_desc(step * idx, step, total_bytes as usize);
                        description.push_str("no frames loaded");
                        output.add_empty_frame(&image, description)?;
                    }
                }
                run_once(
                    &mut image,
                    current_iter,
                    bytes_read,
                    step,
                    total_bytes as usize,
                    output_ctx.as_mut(),
                )?;
                break image;
            }
        }

        tracing::info!("Iteration {current_iter} didn't produce an image");
    };

    loop {
        let current_iter = idx;
        if current_iter >= 120 && current_iter % 60 == 0 && step < step_cap {
            step = (step * 2).min(step_cap);
            buf.resize(step, 0);
            tracing::info!(step, "Increasing step size");
        }

        let buf = fill_buf(&mut input, &mut buf)?;
        if buf.is_empty() {
            break;
        }
        bytes_read += buf.len();

        image.feed_bytes(buf).map_err(Error::ReadJxl)?;
        idx += 1;
        run_once(
            &mut image,
            current_iter,
            bytes_read,
            step,
            total_bytes as usize,
            output_ctx.as_mut(),
        )?;
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

fn progress_desc(byte_offset: usize, step: usize, total_bytes: usize) -> String {
    use std::fmt::Write;

    let mut description = String::new();
    if total_bytes > 0 {
        let progress = byte_offset as f64 / total_bytes as f64;
        let percentage = progress * 100.0;
        writeln!(
            &mut description,
            "{percentage:.2}\\% loaded ({step} bytes/frame)"
        )
        .ok();
    } else {
        writeln!(&mut description, "{step} bytes/frame").ok();
    }

    description
}

fn run_once(
    image: &mut JxlImage,
    current_iter: usize,
    byte_offset: usize,
    step: usize,
    total_bytes: usize,
    output_ctx: Option<&mut OutputType>,
) -> Result<()> {
    use std::fmt::Write;

    match (image.num_loaded_keyframes(), image.is_loading_done()) {
        (0, false) | (1, true) => {}
        _ => {
            tracing::error!("Progressive decoding doesn't support animation");
            return Err(Error::ReadJxl(
                "Progressive decoding doesn't support animation".into(),
            ));
        }
    }

    let mut description = progress_desc(byte_offset, step, total_bytes);

    let loaded_frames = image.num_loaded_frames();
    let last_frame = image.frame(loaded_frames).or_else(|| {
        loaded_frames
            .checked_sub(1)
            .and_then(|idx| image.frame(idx))
    });
    if let Some(frame) = last_frame {
        let idx = frame.index();
        let frame_offset = image.frame_offset(idx).unwrap();
        let offset_within_frame = byte_offset - frame_offset;
        let toc = frame.toc();
        let current_group = toc
            .iter_bitstream_order()
            .take_while(|group| offset_within_frame >= group.offset)
            .last();

        if let Some(current_group) = current_group {
            if offset_within_frame >= current_group.offset + current_group.size as usize {
                write!(&mut description, "frame #{idx} fully loaded").ok();
            } else {
                write!(
                    &mut description,
                    "frame #{idx} decoding {:?}",
                    current_group.kind
                )
                .ok();
            }
        } else {
            write!(&mut description, "frame #{idx} parsing header").ok();
        }
    } else {
        write!(&mut description, "no frames loaded").ok();
    }

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
            output_ctx.add_empty_frame(image, description)?;
        }
        return Ok(());
    };

    tracing::info!("Iteration {current_iter} took {elapsed_msecs:.2} ms");
    if let Some(output_ctx) = output_ctx {
        output_ctx.add_frame(image, &render, description)?;
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
