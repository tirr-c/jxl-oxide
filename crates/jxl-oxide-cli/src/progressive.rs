use std::io::prelude::*;
use std::path::Path;

use jxl_oxide::{JxlImage, JxlThreadPool, Render};

use crate::commands::progressive::*;
use crate::{output, Error, Result};

pub fn handle_progressive(args: ProgressiveArgs) -> Result<()> {
    let _guard = tracing::trace_span!("Handle progressive subcommand").entered();

    #[cfg(feature = "rayon")]
    let pool = JxlThreadPool::rayon(args.num_threads);
    #[cfg(not(feature = "rayon"))]
    let pool = JxlThreadPool::none();

    let mut uninit_image = JxlImage::builder().pool(pool.clone()).build_uninit();

    let mut input = std::fs::File::open(&args.input).map_err(|e| Error::ReadJxl(e.into()))?;
    let mut buf = vec![0u8; args.step as usize];
    let mut idx = 0usize;

    let output_dir = args.output.as_deref();
    if let Some(output_dir) = output_dir {
        std::fs::create_dir_all(output_dir).map_err(Error::WriteImage)?;
    } else {
        tracing::info!("No output path specified, skipping output encoding");
    }

    let mut image = loop {
        let current_iter = idx;

        let buf = fill_buf(&mut input, &mut buf)?;
        if buf.is_empty() {
            tracing::warn!("Partial image");
            return Ok(());
        }

        uninit_image.feed_bytes(buf).map_err(Error::ReadJxl)?;
        idx += 1;
        let init_result = uninit_image.try_init().map_err(Error::ReadJxl)?;
        match init_result {
            jxl_oxide::InitializeResult::NeedMoreData(image) => uninit_image = image,
            jxl_oxide::InitializeResult::Initialized(mut image) => {
                run_once(&mut image, current_iter, output_dir)?;
                break image;
            }
        }

        tracing::info!("Iteration {current_iter} didn't produce an image");
    };

    loop {
        let current_iter = idx;

        let buf = fill_buf(&mut input, &mut buf)?;
        if buf.is_empty() {
            break;
        }

        image.feed_bytes(buf).map_err(Error::ReadJxl)?;
        idx += 1;
        run_once(&mut image, current_iter, output_dir)?;
    }

    if !image.is_loading_done() {
        tracing::warn!("Partial image");
    }

    if output_dir.is_none() {
        tracing::info!("No output path specified, skipping output encoding");
    }

    Ok(())
}

fn run_once(image: &mut JxlImage, current_iter: usize, output_dir: Option<&Path>) -> Result<()> {
    match (image.num_loaded_keyframes(), image.is_loading_done()) {
        (0, false) | (1, true) => {}
        _ => {
            tracing::error!("Progressive decoding doesn't support animation");
            return Err(Error::ReadJxl(
                "Progressive decoding doesn't support animation".into(),
            ));
        }
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
        return Ok(());
    };
    tracing::info!("Iteration {current_iter} took {elapsed_msecs:.2} ms");

    let Some(output_dir) = output_dir else {
        return Ok(());
    };

    let mut output_path = output_dir.to_owned();
    output_path.push(format!("frame{current_iter}.png"));
    let output = std::fs::File::create(output_path).map_err(Error::WriteImage)?;

    let width = image.width();
    let height = image.height();
    output::write_png(
        output,
        image,
        &[render],
        image.pixel_format(),
        None,
        width,
        height,
    )
    .map_err(Error::WriteImage)
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
