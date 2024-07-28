use std::io::prelude::*;

#[derive(Debug)]
struct FixtureHeader {
    width: u32,
    height: u32,
    channels: u32,
}

impl FixtureHeader {
    fn from_bytes(buf: [u8; 12]) -> Self {
        let mut tmp = [0u8; 4];
        tmp.copy_from_slice(&buf[0..4]);
        let width = u32::from_le_bytes(tmp);
        tmp.copy_from_slice(&buf[4..8]);
        let height = u32::from_le_bytes(tmp);
        tmp.copy_from_slice(&buf[8..12]);
        let channels = u32::from_le_bytes(tmp);
        Self {
            width,
            height,
            channels,
        }
    }
}

fn decode<R: Read>(data: &[u8], mut expected: R) {
    let mut header = [0u8; 12];
    expected.read_exact(&mut header).unwrap();
    let fixture_header = FixtureHeader::from_bytes(header);

    let image = jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(data))
        .unwrap();
    let image_header = &image.image_header().metadata;
    let bit_depth = image_header.bit_depth.bits_per_sample();

    for idx in 0..image.num_loaded_keyframes() {
        let frame = image.render_frame(idx).unwrap();
        let mut marker = 0u8;
        expected
            .read_exact(std::slice::from_mut(&mut marker))
            .unwrap();
        if marker != 0 {
            panic!();
        }

        let color_channels = frame.color_channels();
        let extra_channels = frame.extra_channels();
        assert_eq!(
            fixture_header.channels as usize,
            color_channels.len() + extra_channels.len()
        );

        // Peak error threshold of Level 10 tests, from 18181-3
        let frame_header = image.frame_header(idx).unwrap();
        let color_peak_error_threshold = match frame_header.encoding {
            jxl_frame::header::Encoding::VarDct => (0.004 * 65535.0) as u16,
            jxl_frame::header::Encoding::Modular => 1u16 << 14u32.saturating_sub(bit_depth),
        };

        let channels_it = color_channels
            .iter()
            .map(|cc| (cc, color_peak_error_threshold))
            .chain(
                extra_channels
                    .iter()
                    .map(|ec| (ec.grid(), 1u16 << 14u32.saturating_sub(bit_depth))),
            );

        for (grid, peak_error_threshold) in channels_it {
            assert_eq!(fixture_header.width as usize, grid.width());
            assert_eq!(fixture_header.height as usize, grid.height());
            let buf = grid.buf();
            let mut expected_buf = vec![0u8; grid.width() * grid.height() * 2];
            expected.read_exact(&mut expected_buf).unwrap();

            for (&actual_float, expected) in buf.iter().zip(expected_buf.chunks_exact(2)) {
                let expected = u16::from_le_bytes([expected[0], expected[1]]);
                let actual = (actual_float * 65535.0 + 0.5) as u16;
                let diff = actual.abs_diff(expected);
                assert!(diff < peak_error_threshold);
            }
        }
    }
}

fn download_url_with_cache(url: &str, name: &str) -> Vec<u8> {
    let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/cache");
    path.push(name);

    if let Ok(buf) = std::fs::read(&path) {
        buf
    } else {
        let bytes = reqwest::blocking::get(url)
            .and_then(|resp| resp.error_for_status())
            .and_then(|resp| resp.bytes())
            .expect("Cannot download the given URL");
        std::fs::write(path, &bytes).ok();
        bytes.to_vec()
    }
}

macro_rules! test {
    ($($(#[$attr:meta])* $name:ident),* $(,)?) => {
        $(
            #[test]
            $(#[$attr])*
            fn $name() {
                let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
                path.push("tests/decode");
                path.push(stringify!($name));

                let mut input_jxl = path.clone();
                input_jxl.push("input.jxl");
                let input = std::fs::read(input_jxl).expect("Failed to read file");

                let mut output_buf = path.clone();
                output_buf.push("output.buf.zst");

                match std::fs::File::open(output_buf) {
                    Ok(expected) => {
                        let expected = zstd::Decoder::new(expected).expect("Failed to open Zstandard stream");
                        decode(&input, expected);
                    },
                    Err(_) => {
                        let mut output_url = path.clone();
                        output_url.push("output.buf.zst.url");
                        let output_url = std::fs::read_to_string(output_url).expect("Failed to get fixture URL");

                        let filename = concat!(stringify!($name), ".buf.zst");
                        let resp = download_url_with_cache(output_url.trim(), filename);
                        let resp = std::io::Cursor::new(resp);
                        let expected = zstd::Decoder::new(resp).expect("Failed to open Zstandard stream");
                        decode(&input, expected);
                    },
                }
            }
        )*
    };
}

test! {
    minecraft_vardct_e7,
    genshin_ycbcr_420,
    starrail_jpegli_xyb,
    issue_24,
    issue_26,
    issue_32,
    squeeze_edge,
    issue_311,
}
