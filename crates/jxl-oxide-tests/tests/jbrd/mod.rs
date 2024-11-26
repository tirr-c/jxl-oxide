use jxl_oxide::JxlImage;

#[allow(unused)]
use jxl_oxide_tests as util;

#[allow(unused)]
fn reconstruct(image: JxlImage) -> ring::digest::Digest {
    let jbrd = image.jbrd().expect("no reconstruction data available");
    let mut jpeg = Vec::new();

    let aux_boxes = image.aux_boxes();
    let jbrd_header = jbrd.header();
    let expected_icc_len = jbrd_header.expected_icc_len();
    let expected_exif_len = jbrd_header.expected_exif_len();
    let expected_xmp_len = jbrd_header.expected_xmp_len();

    let icc = if expected_icc_len > 0 {
        image.original_icc().unwrap_or(&[])
    } else {
        &[]
    };

    let exif = if expected_exif_len > 0 {
        let b = aux_boxes.first_exif().expect("failed to parse Exif box");
        b.map(|x| x.payload()).unwrap_or(&[])
    } else {
        &[]
    };

    let xmp = if expected_xmp_len > 0 {
        aux_boxes.first_xml().unwrap_or(&[])
    } else {
        &[]
    };

    let frame = image.frame_by_keyframe(0).unwrap();
    let mut reconstructor = jbrd
        .reconstruct(frame, icc, exif, xmp)
        .expect("failed to verify reconstruction data");

    reconstructor
        .write(&mut jpeg)
        .expect("failed to reconstruct JPEG data");

    ring::digest::digest(&ring::digest::SHA256, &jpeg)
}

#[allow(unused)]
macro_rules! test {
    ($($(#[$attr:meta])* $name:ident ($path:expr, $hash:literal $(,)?)),* $(,)?) => {
        $(
            #[test]
            $(#[$attr])*
            fn $name() {
                let input_jxl = std::path::PathBuf::from($path);
                let input = JxlImage::builder().open(input_jxl).expect("Failed to read file");
                let digest = reconstruct(input);

                let expected = ring::test::from_hex($hash).unwrap();
                assert_eq!(digest.as_ref(), &*expected);
            }
        )*
    };
}

#[cfg(feature = "conformance")]
test! {
    cafe(
        util::conformance_path("cafe"),
        "b6f6e4f820ac69234184434e5b77156401fb782bb46b96e26255ff51be1ec290",
    ),
    grayscale_jpeg(
        util::conformance_path("grayscale_jpeg"),
        "a170600cc02b2b029dc79c5ee72dbf9107e8de5a64f1d8b1732c759e09a3d41d",
    ),
}
