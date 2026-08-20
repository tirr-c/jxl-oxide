use std::fs::File;

use image::DynamicImage;
use jxl_oxide::integration::JxlDecoder;
use jxl_oxide_tests as util;

#[test]
fn decode_u8() {
    let path = util::conformance_path("lz77_flower");
    let file = File::open(path).unwrap();
    let decoder = JxlDecoder::new(file).unwrap();

    let image = DynamicImage::from_decoder(decoder).unwrap();
    assert_eq!(image.color(), image::ColorType::Rgb8);
    assert_eq!(image.width(), 834);
    assert_eq!(image.height(), 244);
}

#[test]
fn decode_u16() {
    let path = util::conformance_path("sunset_logo");
    let file = File::open(path).unwrap();
    let decoder = JxlDecoder::new(file).unwrap();

    let image = DynamicImage::from_decoder(decoder).unwrap();
    assert_eq!(image.color(), image::ColorType::Rgba16);
    assert_eq!(image.width(), 924);
    assert_eq!(image.height(), 1386);
}

#[test]
fn decode_f32() {
    let path = util::conformance_path("lossless_pfm");
    let file = File::open(path).unwrap();
    let decoder = JxlDecoder::new(file).unwrap();

    let image = DynamicImage::from_decoder(decoder).unwrap();
    assert_eq!(image.color(), image::ColorType::Rgb32F);
    assert_eq!(image.width(), 500);
    assert_eq!(image.height(), 500);
}

#[test]
fn decode_gray_xyb() {
    let path = util::conformance_path("grayscale");
    let file = File::open(path).unwrap();
    let decoder = JxlDecoder::new(file).unwrap();

    let image = DynamicImage::from_decoder(decoder).unwrap();
    assert_eq!(image.color(), image::ColorType::L8);
    assert_eq!(image.width(), 200);
    assert_eq!(image.height(), 200);
}

#[test]
fn decode_gray_modular() {
    let path = util::conformance_path("grayscale_public_university");
    let file = File::open(path).unwrap();
    let decoder = JxlDecoder::new(file).unwrap();

    let image = DynamicImage::from_decoder(decoder).unwrap();
    assert_eq!(image.color(), image::ColorType::L8);
    assert_eq!(image.width(), 2880);
    assert_eq!(image.height(), 1620);
}

#[test]
fn decode_cmyk() {
    let path = util::conformance_path("cmyk_layers");
    let file = File::open(path).unwrap();
    let decoder = JxlDecoder::new(file).unwrap();

    let image = DynamicImage::from_decoder(decoder).unwrap();
    assert_eq!(image.color(), image::ColorType::Rgba8);
    assert_eq!(image.width(), 512);
    assert_eq!(image.height(), 512);
}

#[test]
fn icc_profile() {
    let path = util::conformance_path("grayscale");
    let file = File::open(path).unwrap();
    let mut decoder = JxlDecoder::new(file).unwrap();
    let icc = image::ImageDecoder::icc_profile(&mut decoder)
        .unwrap()
        .unwrap();
    assert_eq!(&icc, include_bytes!("./grayscale.icc"));
}

#[test]
fn with_reference_frame() {
    let path = util::conformance_path("patches");
    let file = File::open(path).unwrap();
    let decoder = JxlDecoder::new(file).unwrap();

    let image = DynamicImage::from_decoder(decoder).unwrap();
    assert_eq!(image.width(), 1600);
    assert_eq!(image.height(), 1096);
}

/// `set_lf_only` through the `image`-crate integration, which is a different path from the
/// `JxlImage` tests in `lf_only` and has its own way to go wrong.
///
/// **This is a regression test for a panic, not just a feature test.** `dimensions()` answers
/// from the first frame's header (modular frames ignore the request, and the reduction factor
/// includes the frame's own `upsampling`). If that header is not loaded yet, `dimensions()`
/// falls back to the full size while the render still comes back reduced, and the two
/// disagreeing lands in `stream_to_buf`'s length assertion. A 313 KB VarDCT file did exactly
/// that: the small conformance files happen to be fully buffered during init and hid it. So
/// `set_lf_only(true)` now loads up to the first keyframe before returning.
#[test]
fn decode_lf_only() {
    for (name, full_w, full_h) in [("bike", 2048u32, 2560u32), ("sunset_logo", 924, 1386)] {
        let path = util::conformance_path(name);
        let file = File::open(&path).unwrap();
        let mut decoder = JxlDecoder::new(file).unwrap();
        decoder.set_lf_only(true).unwrap();

        // Whatever `dimensions()` promises, the decode must produce exactly that, or the
        // `image` crate has allocated the wrong buffer.
        let (dw, dh) = {
            use image::ImageDecoder;
            decoder.dimensions()
        };
        let image = DynamicImage::from_decoder(decoder).unwrap();
        assert_eq!(
            (image.width(), image.height()),
            (dw, dh),
            "{name}: dimensions() disagreed with the decoded image",
        );

        // `bike` is VarDCT and must actually shrink; `sunset_logo` is modular and must not.
        if name == "bike" {
            assert!(
                image.width() < full_w && image.height() < full_h,
                "{name}: a VarDCT frame must be reduced under lf_only",
            );
        } else {
            assert_eq!((image.width(), image.height()), (full_w, full_h));
        }
    }
}
