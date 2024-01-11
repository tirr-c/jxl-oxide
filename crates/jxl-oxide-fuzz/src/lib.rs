use jxl_oxide::{AllocTracker, JxlImage, JxlThreadPool};

pub fn fuzz_decode(data: &[u8], dimension_limit: u32, alloc_limit: usize) {
    let image = JxlImage::builder()
        .pool(JxlThreadPool::none())
        .alloc_tracker(AllocTracker::with_limit(alloc_limit))
        .read(std::io::Cursor::new(data));
    if let Ok(image) = image {
        let header = image.image_header();
        let max_size = u32::max(header.size.width, header.size.height);

        // Skip huge images
        if max_size > dimension_limit {
            return;
        }
        for keyframe_idx in 0..image.num_loaded_keyframes() {
            let _ = image.render_frame(keyframe_idx);
        }
    }
}
