use jxl_oxide::JxlImage;

fn fuzz_decode(data: &[u8]) {
    if let Ok(image) = JxlImage::from_reader(std::io::Cursor::new(data)) {
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
    large_output_size,
    multiply_integer_overflow,
    dequant_matrix_encoding_mode,
    num_groups_overflow,
    extensions_overflow,
    hybrid_integer_bits,
    icc_output_size_alloc_failed,
    spline_starting_point_overflow,
    noise_on_invisible_frame,
    sharp_lut_oob,
    modular_zero_width,
    icc_tag_size,
    hf_coeff_non_zeros,
    modular_wrong_palette,
    permutation_lehmer_oob,
    permutation_overflow,
    patch_coord_overflow,
);
