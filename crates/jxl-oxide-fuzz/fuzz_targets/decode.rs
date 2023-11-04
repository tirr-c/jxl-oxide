use jxl_oxide::JxlImage;
fn fuzz_decode(data: &[u8]) {
    if let Ok(mut image) = JxlImage::from_reader(std::io::Cursor::new(data)) {
        let header = image.image_header();
        let max_size = u32::max(header.size.width, header.size.height);

        // Skip huge images
        if max_size > 65536 {
            return;
        }
        for keyframe_idx in 0..image.num_loaded_keyframes() {
            let _ = image.render_frame(keyframe_idx);
        }
    }
}
fn main() {
    // Honggfuzz does not support windows yet
    #[cfg(not(target_os = "windows"))]
    {
        use honggfuzz::fuzz;
        loop {
            fuzz!(|data: &[u8]| {
                fuzz_decode(data);
            });
        }
    }
}
