use jxl_oxide::JxlImage;

fn fuzz_decode(data: &[u8]) {
    if let Ok(mut image) = JxlImage::from_reader(std::io::Cursor::new(data)) {
        let _ = image.image_header();
        for keyframe_idx in 0..image.num_loaded_keyframes() {
            let _ = image.render_frame(keyframe_idx);
        }
    }
}

// Macro to simplify writing these tests. To use it write the name of the file in the fuzz_findings directory without the .fuzz extension.
macro_rules! test_by_include {
    ($($(#[$attr:meta])* $name:ident),* $(,)?) => {
        $(
            #[test]
            $(#[$attr])*
            fn $name() {
                let data = include_bytes!(concat!("fuzz_findings/", stringify!($name), ".fuzz"));
                fuzz_decode(data);
            }
        )*
    }
}

test_by_include!(
    #[ignore] large_output_size,
    multiply_integer_overflow,
    #[ignore] out_of_bounds_access,
);
