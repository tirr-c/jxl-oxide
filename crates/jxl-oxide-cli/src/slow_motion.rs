use std::io::prelude::*;

use jxl_oxide::{EnumColourEncoding, JxlImage, Render, RenderingIntent};

use crate::commands::slow_motion::*;
use crate::output::Mp4FileEncoder;
use crate::{Error, Result};

struct LoadProgress {
    iter: usize,
    encoded_frames: usize,
    bytes_read: usize,
    total_bytes: usize,
}

impl LoadProgress {
    fn new(total_bytes: usize) -> Self {
        Self {
            iter: 0,
            encoded_frames: 0,
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
    fn add_encoded_frame(&mut self) {
        self.encoded_frames += 1;
    }

    #[inline]
    fn iter(&self) -> usize {
        self.iter
    }

    #[inline]
    fn progress(&self) -> f64 {
        self.bytes_read as f64 / self.total_bytes as f64
    }
}

impl std::fmt::Display for LoadProgress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let bytes_read = self.bytes_read;
        let total_bytes = self.total_bytes;
        if total_bytes > 0 {
            let percentage = self.progress() * 100.0;
            write!(f, "{percentage:.2}\\% loaded\n{bytes_read} / {total_bytes}")
        } else {
            write!(f, "{bytes_read} bytes loaded")
        }
    }
}

pub fn handle_slow_motion(args: SlowMotionArgs) -> Result<()> {
    let _guard = tracing::trace_span!("Handle slow-motion subcommand").entered();

    let pool = crate::create_thread_pool(args.num_threads);

    let bytes_per_frame = args.bytes_per_frame as usize;

    let mut uninit_image = JxlImage::builder().pool(pool.clone()).build_uninit();

    let input = std::fs::File::open(&args.input).map_err(|e| Error::ReadJxl(e.into()))?;
    let total_bytes = input.metadata().map(|meta| meta.len()).unwrap_or(0);

    let mut progress = LoadProgress::new(total_bytes as usize);
    let mut input = std::io::BufReader::new(input);
    let mut buf = vec![0u8; 4096];
    let mut buf_valid = 0usize;

    let mut encoder = Mp4FileEncoder::new(args.output, None, None, &args.font)?;

    let mut image = loop {
        let current_iter = progress.iter();

        let filled_buf = fill_buf(&mut input, &mut buf[buf_valid..][..bytes_per_frame])?;
        if filled_buf.is_empty() {
            tracing::warn!("Partial image");
            return Ok(());
        }
        progress.add_iter(filled_buf.len());
        buf_valid += filled_buf.len();

        let consumed = uninit_image
            .feed_bytes(&buf[..buf_valid])
            .map_err(Error::ReadJxl)?;
        buf.copy_within(consumed..buf_valid, 0);
        buf_valid -= consumed;

        let init_result = uninit_image.try_init().map_err(Error::ReadJxl)?;
        match init_result {
            jxl_oxide::InitializeResult::NeedMoreData(image) => uninit_image = image,
            jxl_oxide::InitializeResult::Initialized(mut image) => {
                tracing::info!("Image initialized at iteration {current_iter}");

                let intent = RenderingIntent::Relative;
                let encoding = match image.hdr_type() {
                    Some(jxl_oxide::HdrType::Pq) => EnumColourEncoding::bt2100_pq(intent),
                    Some(jxl_oxide::HdrType::Hlg) => EnumColourEncoding::bt2100_hlg(intent),
                    None => EnumColourEncoding::srgb(intent),
                };
                image.request_color_encoding(encoding);

                run_once(&mut image, &mut progress, &mut encoder)?;
                break image;
            }
        }
    };

    while progress.encoded_frames < 900 {
        tracing::trace!(progress.bytes_read);
        let filled_buf = fill_buf(&mut input, &mut buf[buf_valid..][..bytes_per_frame])?;
        if filled_buf.is_empty() {
            if !image.is_loading_done() {
                tracing::warn!("Partial image");
            }
            break;
        }
        progress.add_iter(filled_buf.len());
        buf_valid += filled_buf.len();

        let consumed = image
            .feed_bytes(&buf[..buf_valid])
            .map_err(Error::ReadJxl)?;
        buf.copy_within(consumed..buf_valid, 0);
        buf_valid -= consumed;

        run_once(&mut image, &mut progress, &mut encoder)?;
    }

    encoder.finalize_skip_still()?;
    Ok(())
}

fn run_once(
    image: &mut JxlImage,
    progress: &mut LoadProgress,
    encoder: &mut Mp4FileEncoder,
) -> Result<()> {
    fn run_inner(image: &mut JxlImage) -> Result<Option<Render>> {
        Ok(if image.is_loading_done() {
            Some(image.render_frame_cropped(0).map_err(Error::Render)?)
        } else {
            image.render_loading_frame_cropped().ok()
        })
    }

    let current_iter = progress.iter() - 1;
    let mut encoded = false;
    let mut output = None;

    let decode_start = std::time::Instant::now();
    #[cfg(feature = "rayon")]
    if let Some(rayon_pool) = image.pool().clone().as_rayon_pool() {
        output = rayon_pool.install(|| run_inner(image))?;
        encoded = true;
    }
    if !encoded {
        output = run_inner(image)?;
    }

    let elapsed = decode_start.elapsed();
    let elapsed_seconds = elapsed.as_secs_f64();
    let elapsed_msecs = elapsed_seconds * 1000.0;

    let Some(render) = output else {
        return Ok(());
    };

    tracing::info!("Iteration {current_iter} took {elapsed_msecs:.2} ms");
    encoder.add_frame(image, &render, &*progress)?;
    progress.add_encoded_frame();

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
