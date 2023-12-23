use jxl_oxide::JxlImage;

mod util;

fn read_numpy(mut r: impl std::io::Read, frames: usize, channels: usize) -> Vec<Vec<f32>> {
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

    let mut fb = Vec::new();
    let mut buf = vec![0u8; channels * 4];
    while r.read_exact(&mut buf).is_ok() {
        let mut val = [0u8; 4];
        for c in buf.chunks_exact(4) {
            val.copy_from_slice(c);
            let x = u32::from_le_bytes(val);
            fb.push(f32::from_bits(x));
        }
    }

    let chunk_size = fb.len() / frames;
    fb.chunks_exact(chunk_size).map(|b| b.to_vec()).collect()
}

fn download_object_with_cache(hash: &str, ext: &str) -> Vec<u8> {
    let url = format!(
        "https://storage.googleapis.com/storage/v1/b/jxl-conformance/o/objects%2F{hash}?alt=media"
    );
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
    mut image: JxlImage,
    target_icc: Option<Vec<u8>>,
    expected: Vec<Vec<f32>>,
    expected_peak_error: f32,
    expected_max_rmse: f32,
) {
    let debug = std::env::var("JXL_OXIDE_DEBUG").is_ok();

    image.set_render_spot_colour(false);

    image.set_cms(util::Lcms2);
    if let Some(target_icc) = target_icc {
        image.request_icc(target_icc);
    }

    let num_keyframes = image.num_loaded_keyframes();
    for idx in 0..num_keyframes {
        let render = image.render_frame(idx).expect("failed to render frame");
        let keyframe_idx = render.keyframe_index();
        let expected = &expected[keyframe_idx];

        eprintln!("Testing keyframe #{keyframe_idx}");

        let idx_black = render.extra_channels().iter().position(|x| x.is_black());
        let num_color_channels = render.color_channels().len();

        let fb = render.image_all_channels();
        let interleaved_buffer = fb.buf();
        assert_eq!(expected.len(), interleaved_buffer.len());

        let width = fb.width();
        let height = fb.height();
        let channels = fb.channels();

        let mut sum_se = vec![0.0f32; channels];
        let mut peak_error = 0.0f32;
        for y in 0..height {
            for x in 0..width {
                for c in 0..channels {
                    let reference = expected[c + (x + y * width) * channels];
                    let mut output = interleaved_buffer[c + (x + y * width) * channels];
                    if let Some(idx_black) = idx_black {
                        let idx_black = idx_black + num_color_channels;
                        if c < num_color_channels || c == idx_black {
                            output = 1.0 - output;
                        }
                    }
                    let sum_se = &mut sum_se[c];

                    let abs_error = (output - reference).abs();
                    if debug && abs_error >= expected_peak_error {
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

        assert!(peak_error <= expected_peak_error);
        assert!(max_rmse <= expected_max_rmse);
    }

    assert_eq!(expected.len(), num_keyframes);
    assert!(image.is_loading_done());
}

macro_rules! conformance_test {
    ($($(#[$attr:meta])* $name:ident ($npy_hash:literal, $icc_hash:literal, $frames:literal, $channels:literal, $peak_error:expr, $max_rmse:expr $(,)? )),* $(,)?) => {
        $(
            #[test]
            $(#[$attr])*
            fn $name() {
                let perform_ct = $icc_hash != "skip";

                let buf = download_object_with_cache($npy_hash, "npy");
                let target_icc = perform_ct.then(|| download_object_with_cache($icc_hash, "icc"));

                let expected = read_numpy(std::io::Cursor::new(buf), $frames, $channels);

                let path = util::conformance_path(stringify!($name));
                let image = JxlImage::builder().open(path).expect("Failed to open file");

                run_test(image, target_icc, expected, $peak_error, $max_rmse);
            }
        )*
    };
}

conformance_test! {
    bicycles(
        "6f71d8ca122872e7d850b672e7fb46b818c2dfddacd00b3934fe70aa8e0b327e",
        "80a1d9ea2892c89ab10a05fcbd1d752069557768fac3159ecd91c33be0d74a19",
        1,
        3,
        0.000976562,
        0.000976562,
    ),
    delta_palette(
        "952b9e16aa0ae23df38c6b358cb4835b5f9479838f6855b96845ea54b0528c1f",
        "80a1d9ea2892c89ab10a05fcbd1d752069557768fac3159ecd91c33be0d74a19",
        1,
        3,
        0.000976562,
        0.000976562,
    ),
    lz77_flower(
        "953d3ada476e3218653834c9addc9c16bb6f9f03b18be1be8a85c07a596ea32d",
        "793cb9df4e4ce93ce8fe827fde34e7fb925b7079fcb68fba1e56fc4b35508ccb",
        1,
        3,
        0.000976562,
        0.000976562,
    ),
    patches_lossless(
        "806201a2c99d27a54c400134b3db7bfc57476f9bc0775e59eea802d28aba75de",
        "3a10bcd8e4c39d12053ebf66d18075c7ded4fd6cf78d26d9c47bdc0cde215115",
        1,
        4,
        0.000976562,
        0.000976562,
    ),
    bike(
        "815c89d1fe0bf67b6a1c8139d0af86b6e3f11d55c5a0ed9396256fb05744469e",
        "809e189d1bf1fadb66f130ed0463d0de374b46497d299997e7c84619cbd35ed3",
        1,
        3,
        0.007,
        0.0001,
    ),
    sunset_logo(
        "bf1c1d5626ced3746df867cf3e3a25f3d17512c2e837b5e3a04743660e42ad81",
        "80a1d9ea2892c89ab10a05fcbd1d752069557768fac3159ecd91c33be0d74a19",
        1,
        4,
        0.000244141,
        0.000244141,
    ),
    blendmodes(
        "6ef265631818f313a70cb23788d1185408ce07243db8d5940553e7ea7467b786",
        "80a1d9ea2892c89ab10a05fcbd1d752069557768fac3159ecd91c33be0d74a19",
        1,
        4,
        0.004,
        0.0001,
    ),
    progressive(
        "5a9d25412e2393ee11632942b4b683cda3f838dd72ab2550cfffc8f34d69c852",
        "956c9b6ecfef8ef1420e8e93e30a89d3c1d4f7ce5c2f3e2612f95c05a7097064",
        1,
        3,
        0.02,
        0.0001,
    ),
    animation_icos4d(
        "77a060cfa0d4df183255424e13e4f41a90b3edcea1248e3f22a3b7fcafe89e49",
        "80a1d9ea2892c89ab10a05fcbd1d752069557768fac3159ecd91c33be0d74a19",
        48,
        4,
        0.005,
        0.0001,
    ),
    animation_spline(
        "a571c5cbba58affeeb43c44c13f81e2b1962727eb9d4e017e4f25d95c7388f10",
        "80a1d9ea2892c89ab10a05fcbd1d752069557768fac3159ecd91c33be0d74a19",
        60,
        3,
        0.004,
        0.0001,
    ),
    animation_newtons_cradle(
        "4309286cd22fa4008db3dcceee6a72a806c9291bd7e035cf555f3b470d0693d8",
        "80a1d9ea2892c89ab10a05fcbd1d752069557768fac3159ecd91c33be0d74a19",
        36,
        4,
        0.000976562,
        0.000976562,
    ),
    alpha_triangles(
        "1d8471e3b7f0768f408b5e5bf5fee0de49ad6886d846712b1aaa702379722e2b",
        "80a1d9ea2892c89ab10a05fcbd1d752069557768fac3159ecd91c33be0d74a19",
        1,
        4,
        0.001953125,
        0.001953125,
    ),
    lossless_pfm(
        "1eac3ced5c60ef8a3a602f54c6a9d28162dfee51cd85b8dd7e52f6e3212bbb52",
        "5e44e64c9f97515f43cfe7f4f725c6ec8983533f44c06c4c533a1b9a4c2ce6a6",
        1,
        3,
        0.0,
        0.0,
    ),
    noise(
        "b7bb25b911ab5f4b9a6a6da9c220c9ea738de685f9df25eb860e6bbe1302237d",
        "80a1d9ea2892c89ab10a05fcbd1d752069557768fac3159ecd91c33be0d74a19",
        1,
        3,
        0.004,
        0.0001,
    ),
    cafe(
        "4aaea4e1bda3771e62643fcdf2003ffe6048ee2870c93f67d34d6cc16cb7da4b",
        "bef95ce5cdb139325f2a299b943158e00e39a7ca3cf597ab3dfa3098e42fc707",
        1,
        3,
        0.004,
        1e-5,
    ),
    upsampling(
        "9b83952c4bba9dc93fd5c5c49e27eab29301e848bf70dceccfec96b48d3ab975",
        "80a1d9ea2892c89ab10a05fcbd1d752069557768fac3159ecd91c33be0d74a19",
        1,
        4,
        0.004,
        0.0001,
    ),
    spot(
        "82de72e756db992792b8e3eb5eac5194ef83e9ab4dc03e846492fbedde7b58da",
        "ce0caee9506116ea94d7367d646f7fd6d0b7e82feb8d1f3de4edb3ba57bae07e",
        1,
        6,
        3.815e-6,
        3.815e-6,
    ),
    grayscale(
        "59162e158e042dc44710eb4d334cea78135399613461079582d963fe79251b68",
        "57363d9ec00043fe6e3f40b1c3e0cc4676773011fd0818710fb26545002ac27d",
        1,
        1,
        0.004,
        0.0001,
    ),
    grayscale_jpeg(
        "c0b86989e287649b944f1734ce182d1c1ac0caebf12cec7d487d9793f51f5b8f",
        "78001f4bf342ecf417b8dac5e3c7cf8da3ee25701951bc2a7e0868bc6dc81cac",
        1,
        1,
        0.004,
        1e-5,
    ),
    grayscale_public_university(
        "851abd36b93948cfaeabeb65c8bb8727ebca4bb1d2697bce73461e05ccf24c1e",
        "48d006762d583f6e354a3223c0a5aeaff7f45a324e229d237d69630bcc170241",
        1,
        1,
        0.000976562,
        0.000976562,
    ),
    alpha_nonpremultiplied(
        "cad070b944d8aff6b7865c211e44bc469b08addf5b4a19d11fdc4ef2f7494d1b",
        "80a1d9ea2892c89ab10a05fcbd1d752069557768fac3159ecd91c33be0d74a19",
        1,
        4,
        6.1035e-5,
        6.1035e-5,
    ),
    #[ignore]
    alpha_premultiplied(
        "073c1d942ba408f94f6c0aed2fef3d442574899901c616afa060dbd8044bbdb9",
        "80a1d9ea2892c89ab10a05fcbd1d752069557768fac3159ecd91c33be0d74a19",
        1,
        4,
        3.815e-06,
        3.815e-06,
    ),
    bench_oriented_brg(
        "eac8c30907e41e53a73a0c002bc39e998e0ceb021bd523f5bff4580b455579e6",
        "6603ae12a4ac1ac742cacd887e9b35552a12c354ff25a00cae069ad4b932e6cc",
        1,
        3,
        0.004,
        1e-5,
    ),
    opsin_inverse(
        "a3142a144c112160b8e5a12eb17723fa5fe0cfeb00577e4fb59a5f6cea126a9b",
        "80a1d9ea2892c89ab10a05fcbd1d752069557768fac3159ecd91c33be0d74a19",
        1,
        3,
        0.004,
        0.0001,
    ),
    cmyk_layers(
        "a01913d4e4b1a89bd96e5de82a5dfb9925c7827ee6380ad60c0b1c4becb53880",
        "4855b8fabb96bdc6495d45d089bb8c8efb1ae18389e0dc9e75a5f701a9c0b662",
        1,
        5,
        0.000976562,
        0.000976562,
    ),
}
