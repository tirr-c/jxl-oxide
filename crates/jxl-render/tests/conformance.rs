use jxl_bitstream::{Bundle, header::{Headers, TransferFunction}};
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

fn download_object_with_cache(hash: &str) -> Vec<u8> {
    let url = format!("https://storage.googleapis.com/storage/v1/b/jxl-conformance/o/objects%2F{hash}?alt=media");
    let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/cache");
    path.push(hash);
    path.set_extension("npy");

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

#[test]
fn bicycles() {
    let buf = download_object_with_cache("6f71d8ca122872e7d850b672e7fb46b818c2dfddacd00b3934fe70aa8e0b327e");
    let expected = read_numpy(std::io::Cursor::new(buf));

    let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/reference/testcases");
    path.push("bicycles/input.jxl");
    let file = std::fs::File::open(path).expect("Failed to open file");
    let mut bitstream = jxl_bitstream::Bitstream::new(file);
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

    if headers.metadata.xyb_encoded {
        fb.yxb_to_srgb_linear(&headers.metadata);
    }
    if headers.metadata.xyb_encoded || headers.metadata.colour_encoding.tf == TransferFunction::Linear {
        fb.srgb_linear_to_standard();
    }

    let expected_len = expected.len();
    let actual_len = fb.width() as usize * fb.height() as usize * fb.channels() as usize;
    assert_eq!(expected_len, actual_len);
}
