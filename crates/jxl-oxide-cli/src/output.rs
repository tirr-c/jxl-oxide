use std::io::prelude::*;

use jxl_oxide::{FrameBuffer, JxlImage, PixelFormat, Render};

pub(crate) fn write_png<W: Write>(
    output: W,
    image: &JxlImage,
    keyframes: &[Render],
    pixfmt: PixelFormat,
    force_bit_depth: Option<png::BitDepth>,
    width: u32,
    height: u32,
) -> std::io::Result<()> {
    // Color encoding information
    let source_icc = image.rendered_icc();
    let cicp = image.rendered_cicp();
    let metadata = &image.image_header().metadata;

    let mut encoder = png::Encoder::new(output, width, height);

    let color_type = match pixfmt {
        PixelFormat::Gray => png::ColorType::Grayscale,
        PixelFormat::Graya => png::ColorType::GrayscaleAlpha,
        PixelFormat::Rgb => png::ColorType::Rgb,
        PixelFormat::Rgba => png::ColorType::Rgba,
        _ => {
            tracing::error!("Cannot output CMYK PNG");
            panic!();
        }
    };
    encoder.set_color(color_type);

    let bit_depth = force_bit_depth.unwrap_or(if metadata.bit_depth.bits_per_sample() > 8 {
        png::BitDepth::Sixteen
    } else {
        png::BitDepth::Eight
    });
    let sixteen_bits = bit_depth == png::BitDepth::Sixteen;
    if sixteen_bits {
        encoder.set_depth(png::BitDepth::Sixteen);
    } else {
        encoder.set_depth(png::BitDepth::Eight);
    }

    if let Some(animation) = &metadata.animation {
        let num_plays = animation.num_loops;
        encoder
            .set_animated(keyframes.len() as u32, num_plays)
            .unwrap();
    }

    encoder.validate_sequence(true);

    let mut writer = encoder.write_header()?;

    tracing::debug!("Embedding ICC profile");
    let compressed_icc = miniz_oxide::deflate::compress_to_vec_zlib(&source_icc, 7);
    let mut iccp_chunk_data = vec![b'0', 0, 0];
    iccp_chunk_data.extend(compressed_icc);
    writer.write_chunk(png::chunk::iCCP, &iccp_chunk_data)?;

    if let Some(cicp) = cicp {
        tracing::debug!(cicp = format_args!("{:?}", cicp), "Writing cICP chunk");
        writer.write_chunk(png::chunk::ChunkType([b'c', b'I', b'C', b'P']), &cicp)?;
    }

    tracing::debug!("Writing image data");
    for keyframe in keyframes {
        if let Some(animation) = &metadata.animation {
            let duration = keyframe.duration();
            let numer = animation.tps_denominator * duration;
            let denom = animation.tps_numerator;
            let (numer, denom) = if numer >= 0x10000 || denom >= 0x10000 {
                if duration == 0xffffffff {
                    tracing::warn!(numer, denom, "Writing multi-page image in APNG");
                } else {
                    tracing::warn!(numer, denom, "Frame duration is not representable in APNG");
                }
                let duration = (numer as f32 / denom as f32) * 65535.0;
                (duration as u16, 0xffffu16)
            } else {
                (numer as u16, denom as u16)
            };
            writer.set_frame_delay(numer, denom).unwrap();
        }

        let mut stream = keyframe.stream();
        let mut fb = FrameBuffer::new(
            stream.width() as usize,
            stream.height() as usize,
            stream.channels() as usize,
        );
        stream.write_to_buffer(fb.buf_mut());

        if sixteen_bits {
            let mut buf = vec![0u8; fb.width() * fb.height() * fb.channels() * 2];
            for (b, s) in buf.chunks_exact_mut(2).zip(fb.buf()) {
                let w = (*s * 65535.0 + 0.5).clamp(0.0, 65535.0) as u16;
                let [b0, b1] = w.to_be_bytes();
                b[0] = b0;
                b[1] = b1;
            }
            writer.write_image_data(&buf)?;
        } else {
            let mut buf = vec![0u8; fb.width() * fb.height() * fb.channels()];
            for (b, s) in buf.iter_mut().zip(fb.buf()) {
                *b = (*s * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            }
            writer.write_image_data(&buf)?;
        }
    }

    writer.finish()?;
    Ok(())
}

pub(crate) fn write_npy<W: Write>(
    output: W,
    keyframes: &[Render],
    width: u32,
    height: u32,
) -> std::io::Result<()> {
    let channels = {
        let first_frame = keyframes.first().unwrap();
        first_frame.color_channels().len() + first_frame.extra_channels().0.len()
    };

    let mut output = std::io::BufWriter::new(output);
    output.write_all(b"\x93NUMPY\x01\x00").unwrap();
    let header = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, {}, {}, {}), }}\n",
        keyframes.len(),
        height,
        width,
        channels,
    );
    output.write_all(&(header.len() as u16).to_le_bytes())?;
    output.write_all(header.as_bytes())?;

    tracing::debug!("Writing image data");
    for keyframe in keyframes {
        let fb = keyframe.image_all_channels();
        for sample in fb.buf() {
            output.write_all(&sample.to_bits().to_le_bytes())?;
        }
    }

    output.flush()?;
    Ok(())
}
