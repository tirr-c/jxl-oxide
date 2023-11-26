use jxl_oxide::{AllocTracker, JxlImage, JxlThreadPool};

fn fuzz_decode(data: &[u8]) {
    let image = JxlImage::builder()
        .pool(JxlThreadPool::none())
        .alloc_tracker(AllocTracker::with_limit(128 * 1024 * 1024)) // 128 MiB
        .read(std::io::Cursor::new(data));
    if let Ok(image) = image {
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
