use lcms2::Profile;

use jxl_bitstream::{Bundle, header::Headers};
use jxl_render::RenderContext;

fn read_numpy(mut r: impl std::io::Read) -> Vec<f32> {
    let mut magic = [0u8; 6];
    let mut version = [0u8; 2];
    let mut meta_len = [0u8; 2];

    r.read_exact(&mut magic).unwrap();
    r.read_exact(&mut version).unwrap();
    r.read_exact(&mut meta_len).unwrap();
    assert_eq!(&magic, b"\x93NUMPY");

    let meta_len = u16::from_le_bytes(meta_len) as usize;
    let mut meta = vec![0u8; meta_len];
    r.read_exact(&mut meta).unwrap();

    let mut out = Vec::new();
    let mut buf = [0u8; 12];
    while r.read_exact(&mut buf).is_ok() {
        let mut val = [0u8; 4];
        for c in buf.chunks_exact(4) {
            val.copy_from_slice(c);
            let x = u32::from_le_bytes(val);
            out.push(f32::from_bits(x));
        }
    }

    out
}

fn download_object_with_cache(hash: &str, ext: &str) -> Vec<u8> {
    let url = format!("https://storage.googleapis.com/storage/v1/b/jxl-conformance/o/objects%2F{hash}?alt=media");
    let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/cache");
    path.push(hash);
    path.set_extension(ext);

    if let Ok(buf) = std::fs::read(&path) {
        buf
    } else {
        let bytes = reqwest::blocking::get(url)
            .and_then(|resp| resp.error_for_status())
            .and_then(|resp| resp.bytes())
            .expect("Cannot download reference image");
        std::fs::write(path, &bytes).ok();
        bytes.to_vec()
    }
}

fn run_test(
    mut bitstream: jxl_bitstream::Bitstream<std::fs::File>,
    target_icc: Vec<u8>,
    expected: Vec<f32>,
    expected_peak_error: f32,
) -> (f32, f32) {
    let target_profile = Profile::new_icc(&target_icc).expect("failed to parse ICC profile");

    let headers = Headers::parse(&mut bitstream, ()).expect("Failed to read headers");

    let mut render = RenderContext::new(&headers);
    render.read_icc_if_exists(&mut bitstream).expect("failed to decode ICC");

    if headers.metadata.have_preview {
        bitstream.zero_pad_to_byte().expect("Zero-padding failed");

        let frame = jxl_frame::Frame::parse(&mut bitstream, &headers)
            .expect("Failed to read frame header");

        let toc = frame.toc();
        let bookmark = toc.bookmark() + (toc.total_byte_size() * 8);
        bitstream.skip_to_bookmark(bookmark).expect("Failed to skip");
    }

    render
        .load_all_frames(&mut bitstream)
        .expect("failed to load frames");
    let mut fb = render.render_cropped(None).expect("failed to render");

    let source_profile = if headers.metadata.xyb_encoded {
        fb.yxb_to_srgb_linear(&headers.metadata);
        fb.srgb_linear_to_standard();
        Profile::new_srgb()
    } else if headers.metadata.colour_encoding.is_srgb() {
        Profile::new_srgb()
    } else {
        todo!()
    };

    let width = fb.width() as usize;
    let height = fb.height() as usize;
    let channels = fb.channel_buffers();
    assert_eq!(channels.len(), 3);

    let expected_len = expected.len();
    let actual_len = width * height * channels.len();
    assert_eq!(expected_len, actual_len);

    let mut interleaved_buffer = vec![[0.0f32; 3]; width * height];
    for y in 0..height {
        for x in 0..width {
            let out = &mut interleaved_buffer[x + y * width];
            for (&ch, out) in channels.iter().zip(out) {
                *out = ch[x + y * width];
            }
        }
    }

    let pixfmt = lcms2::PixelFormat::RGB_FLT;
    let transform = lcms2::Transform::new(
        &source_profile,
        pixfmt,
        &target_profile,
        pixfmt,
        lcms2::Intent::Perceptual,
    ).expect("failed to create transform");
    transform.transform_in_place(&mut interleaved_buffer);

    let mut sum_se = vec![0.0f32; channels.len()];
    let mut peak_error = 0.0f32;
    for y in 0..height {
        for x in 0..width {
            let pixel = &interleaved_buffer[x + y * width];
            for (c, (&output, sum_se)) in pixel.iter().zip(&mut sum_se).enumerate() {
                let reference = expected[c + (x + y * width) * channels.len()];
                let abs_error = (output - reference).abs();
                if abs_error >= expected_peak_error {
                    eprintln!("abs_error is larger than max peak_error, at (x={x}, y={y}, c={c}), reference={reference}, actual={output}");
                }
                peak_error = peak_error.max(abs_error);
                *sum_se += abs_error * abs_error;
            }
        }
    }

    let mut max_rmse = 0.0f32;
    for se in sum_se {
        let rmse = (se / (width * height) as f32).sqrt();
        max_rmse = max_rmse.max(rmse);
    }

    eprintln!("peak_error = {}", peak_error);
    eprintln!("max_rmse = {}", max_rmse);

    (peak_error, max_rmse)
}

macro_rules! conformance_test {
    ($(#[$attr:meta])* $name:ident ($npy_hash:literal, $icc_hash:literal, $peak_error:expr, $max_rmse:expr $(,)? )) => {
        #[test]
        $(#[$attr])*
        fn $name() {
            let buf = download_object_with_cache($npy_hash, "npy");
            let target_icc = download_object_with_cache($icc_hash, "icc");

            let expected = read_numpy(std::io::Cursor::new(buf));

            let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            path.push("tests/conformance/testcases");
            path.push(stringify!($name));
            path.push("input.jxl");
            let file = std::fs::File::open(path).expect("Failed to open file");
            let bitstream = jxl_bitstream::Bitstream::new(file);

            let (peak_error, max_rmse) = run_test(bitstream, target_icc, expected, $peak_error);

            assert!(peak_error < $peak_error);
            assert!(max_rmse < $max_rmse);
        }
    };
}

conformance_test! {
    bicycles(
        "6f71d8ca122872e7d850b672e7fb46b818c2dfddacd00b3934fe70aa8e0b327e",
        "80a1d9ea2892c89ab10a05fcbd1d752069557768fac3159ecd91c33be0d74a19",
        0.000976562,
        0.000976562,
    )
}

conformance_test! {
    delta_palette(
        "952b9e16aa0ae23df38c6b358cb4835b5f9479838f6855b96845ea54b0528c1f",
        "80a1d9ea2892c89ab10a05fcbd1d752069557768fac3159ecd91c33be0d74a19",
        0.000976562,
        0.000976562,
    )
}
