use image::{DynamicImage, ImageDecoder};
use jxl_oxide::integration::JxlDecoder;

fn main() {
    let mut args = std::env::args_os().skip(1);
    let path = args
        .next()
        .expect("expected input filename as a command line argument");
    let output_path = args
        .next()
        .expect("expected output filename as a command line argument");
    assert!(args.next().is_none(), "extra command line argument found");

    let file = std::fs::File::open(path).expect("cannot open file");
    let mut decoder = JxlDecoder::new(file).expect("cannot decode image");

    #[allow(unused)]
    let icc = decoder.icc_profile().unwrap();
    let image = DynamicImage::from_decoder(decoder).expect("cannot decode image");

    let output_file = std::fs::File::create(output_path).expect("cannot open output file");
    let encoder = image::codecs::png::PngEncoder::new(output_file);
    // FIXME: PNG encoder of `image` doesn't support setting ICC profile for some reason
    // use image::ImageEncoder;
    // let mut encoder = encoder;
    // if let Some(icc) = icc {
    //     encoder.set_icc_profile(icc).unwrap();
    // }
    image
        .write_with_encoder(encoder)
        .expect("cannot encode image");
}
