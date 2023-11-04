use jxl_oxide::JxlImage;

fn fuzz_decode(data: &[u8]) {
    if let Ok(mut image) = JxlImage::from_reader(std::io::Cursor::new(data)) {
        let header = image.image_header();
        let max_size = u32::max(header.size.width, header.size.height);
        for keyframe_idx in 0..image.num_loaded_keyframes() {
            let _ = image.render_frame(keyframe_idx);
        }
    }
}

// Macro to simplify writing these tests. To use it write the name of the file in the fuzz_findings directory without the .fuzz extension.
macro_rules! test_by_include {
    ($($name:ident),* $(,)?) => {
        $(
            #[ignore]
            #[test]
            fn $name() {
                let data = include_bytes!(concat!("fuzz_findings/", stringify!($name), ".fuzz"));
                fuzz_decode(data);
            }
        )*
    }
}

test_by_include!(
    large_output_size,
    multiply_integer_overflow,
    out_of_bounds_access,
);
