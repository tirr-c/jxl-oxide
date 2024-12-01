#[allow(unused)]
use jxl_oxide::JxlImage;

#[allow(unused)]
use jxl_oxide_tests as util;

#[allow(unused)]
macro_rules! test {
    ($($(#[$attr:meta])* $name:ident ($path:expr, $hash:literal $(,)?)),* $(,)?) => {
        $(
            #[test]
            $(#[$attr])*
            fn $name() {
                let input_jxl = std::path::PathBuf::from($path);
                let input = JxlImage::builder().open(input_jxl).expect("Failed to read file");

                let mut jpeg = Vec::new();
                input.reconstruct_jpeg(&mut jpeg).expect("failed to reconstruct JPEG data");
                let digest = ring::digest::digest(&ring::digest::SHA256, &jpeg);

                let expected = ring::test::from_hex($hash).unwrap();
                assert_eq!(digest.as_ref(), &*expected);
            }
        )*
    };
}

#[cfg(feature = "conformance")]
test! {
    bench_oriented_brg(
        util::conformance_path("bench_oriented_brg"),
        "cad665c67d74e3e5cf775ef618c73b0e70dfece33db7dbe0130bc889f2214e1b",
    ),
    cafe(
        util::conformance_path("cafe"),
        "b6f6e4f820ac69234184434e5b77156401fb782bb46b96e26255ff51be1ec290",
    ),
    grayscale_jpeg(
        util::conformance_path("grayscale_jpeg"),
        "a170600cc02b2b029dc79c5ee72dbf9107e8de5a64f1d8b1732c759e09a3d41d",
    ),
}

#[cfg(feature = "decode")]
test! {
    genshin_ycbcr_420(
        util::decode_testcases_dir().join("genshin_ycbcr_420/input.jxl"),
        "24e05ce200df019710eb56cad43e4349bd6edee05865b1eb0eee1581406535b5",
    ),
    starrail_jpegli_xyb(
        util::decode_testcases_dir().join("starrail_jpegli_xyb/input.jxl"),
        "21ddc2688c89ec279a2e9415bc2e82c4d4b56362316d1332131b6fb97b0d1ea0",
    ),
}
