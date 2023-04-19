use lcms2::Profile;

use jxl_bitstream::Bundle;
use jxl_render::RenderContext;
use jxl_image::Headers;

fn read_numpy(mut r: impl std::io::Read, channels: usize) -> Vec<f32> {
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
    let mut buf = vec![0u8; channels * 4];
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

fn run_test<R: std::io::Read>(
    mut bitstream: jxl_bitstream::Bitstream<R>,
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
        .load_all_frames(&mut bitstream, false)
        .expect("failed to load frames");
    let fb = render.render_cropped(None).expect("failed to render");

    let source_profile = {
        if headers.metadata.colour_encoding.want_icc {
            Profile::new_icc(render.icc()).unwrap()
        } else {
            let icc = jxl_color::icc::colour_encoding_to_icc(&headers.metadata.colour_encoding).unwrap();
            Profile::new_icc(&icc).unwrap()
        }
    };

    let mut grids = fb.into_iter().map(From::from).collect::<Vec<_>>();
    let mut fb = jxl_image::FrameBuffer::from_grids(&grids[..3], headers.size.width as usize, headers.size.height as usize).unwrap();
    let width = fb.width();
    let height = fb.height();
    let pixfmt = lcms2::PixelFormat::RGB_FLT;
    let transform = lcms2::Transform::new(
        &source_profile,
        pixfmt,
        &target_profile,
        pixfmt,
        lcms2::Intent::RelativeColorimetric,
    ).expect("failed to create transform");
    transform.transform_in_place(fb.buf_grouped_mut::<3>());
    let fb = fb.buf();
    for y in 0..height {
        for x in 0..width {
            for c in 0..3 {
                grids[c].set(x, y, fb[c + (x + y * width) * 3]);
            }
        }
    }

    let fb = jxl_image::FrameBuffer::from_grids(&grids, headers.size.width as usize, headers.size.height as usize).unwrap();
    let channels = fb.channels();

    let interleaved_buffer = fb.buf();
    assert_eq!(expected.len(), interleaved_buffer.len());

    let mut sum_se = vec![0.0f32; channels];
    let mut peak_error = 0.0f32;
    for y in 0..height {
        for x in 0..width {
            for c in 0..channels {
                let reference = expected[c + (x + y * width) * channels];
                let output = interleaved_buffer[c + (x + y * width) * channels];
                let sum_se = &mut sum_se[c];

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
    ($(#[$attr:meta])* $name:ident ($npy_hash:literal, $icc_hash:literal, $channels:literal, $peak_error:expr, $max_rmse:expr $(,)? )) => {
        #[test]
        $(#[$attr])*
        fn $name() {
            let buf = download_object_with_cache($npy_hash, "npy");
            let target_icc = download_object_with_cache($icc_hash, "icc");

            let expected = read_numpy(std::io::Cursor::new(buf), $channels);

            let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            path.push("tests/conformance/testcases");
            path.push(stringify!($name));
            path.push("input.jxl");
            let file = std::fs::File::open(path).expect("Failed to open file");
            let bitstream = jxl_bitstream::Bitstream::new_detect(file);

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
        3,
        0.000976562,
        0.000976562,
    )
}

conformance_test! {
    delta_palette(
        "952b9e16aa0ae23df38c6b358cb4835b5f9479838f6855b96845ea54b0528c1f",
        "80a1d9ea2892c89ab10a05fcbd1d752069557768fac3159ecd91c33be0d74a19",
        3,
        0.000976562,
        0.000976562,
    )
}

conformance_test! {
    #[ignore = "slight ICC profile mismatch"]
    lz77_flower(
        "953d3ada476e3218653834c9addc9c16bb6f9f03b18be1be8a85c07a596ea32d",
        "793cb9df4e4ce93ce8fe827fde34e7fb925b7079fcb68fba1e56fc4b35508ccb",
        3,
        0.000976562,
        0.000976562,
    )
}

conformance_test! {
    patches_lossless(
        "806201a2c99d27a54c400134b3db7bfc57476f9bc0775e59eea802d28aba75de",
        "3a10bcd8e4c39d12053ebf66d18075c7ded4fd6cf78d26d9c47bdc0cde215115",
        4,
        0.000976562,
        0.000976562,
    )
}

conformance_test! {
    #[ignore = "high error on some pixels"]
    bike(
        "815c89d1fe0bf67b6a1c8139d0af86b6e3f11d55c5a0ed9396256fb05744469e",
        "809e189d1bf1fadb66f130ed0463d0de374b46497d299997e7c84619cbd35ed3",
        3,
        0.06,
        0.02,
    )
}
