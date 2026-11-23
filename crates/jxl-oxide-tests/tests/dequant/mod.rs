//! Dequantization matrices encoded in the Identity and DCT2 modes store their
//! weights divided by 64. `library.jxl` is a synthetic image whose frame uses
//! Identity and DCT2x2 blocks with the library matrices; each `custom_*.jxl`
//! is the same frame with one matrix written in its custom mode, carrying the
//! library weights. All three decode to the same pixels.
use jxl_oxide::JxlImage;

fn decode(data: &[u8]) -> Vec<f32> {
    let image = JxlImage::builder()
        .read(std::io::Cursor::new(data))
        .expect("failed to read image");
    let render = image.render_frame(0).expect("failed to render frame");
    render.image_all_channels().buf().to_vec()
}

macro_rules! test_same_as_library {
    ($($name:ident),* $(,)?) => {
        $(
            #[test]
            fn $name() {
                let expected = decode(include_bytes!("library.jxl"));
                let actual = decode(include_bytes!(concat!(stringify!($name), ".jxl")));
                assert!(expected == actual, "decoded pixels differ from library.jxl");
            }
        )*
    };
}

test_same_as_library!(custom_identity, custom_dct2);
