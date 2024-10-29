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
