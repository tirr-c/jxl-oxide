use honggfuzz::fuzz;
use jxl_oxide::JxlImage;

fn decode(data: &[u8]) {
    if let Ok(mut image) = JxlImage::from_reader(std::io::Cursor::new(data)) {
        let header = image.image_header();
        let max_size = u32::max(header.size.width, header.size.height);
        drop(header);

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
    loop {
        fuzz!(|data: &[u8]| {
            decode(data);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_multiply_with_overflow_height() {
        //let data = include_bytes!("../hfuzz_workspace/fuzz_decode/SIGABRT.PC.7ffff7d1383c.STACK.191f89ecd1.CODE.-6.ADDR.0.INSTR.mov____%eax,%ebx.fuzz");
        //let _ = JxlImage::from_reader(std::io::Cursor::new(data));
    }
    #[test]
    fn test_crash_1() {
        let data = include_bytes!("../hfuzz_workspace/fuzz_decode/SIGABRT.PC.7ffff7d1383c.STACK.c9ec5b426.CODE.-6.ADDR.0.INSTR.mov____%eax,%ebx.fuzz");
        let _ = JxlImage::from_reader(std::io::Cursor::new(data));
    }

    #[test]
    fn test_crash_2() {
        let data = include_bytes!("../hfuzz_workspace/fuzz_decode/SIGABRT.PC.7ffff7d1383c.STACK.1b3d7db539.CODE.-6.ADDR.0.INSTR.mov____%eax,%ebx.fuzz");
        decode(data);
    }

    #[test]
    fn test_crash_3() {
        let data = include_bytes!("../hfuzz_workspace/fuzz_decode/SIGABRT.PC.7ffff7d1383c.STACK.1be105aa8d.CODE.-6.ADDR.0.INSTR.mov____%eax,%ebx.fuzz");
        decode(data);
    }
    #[test]
    fn test_crash_out_of_bounds() {
        let data = include_bytes!("../hfuzz_workspace/fuzz_decode/SIGABRT.PC.7ffff7d1383c.STACK.1be13466eb.CODE.-6.ADDR.0.INSTR.mov____%eax,%ebx.fuzz");
        decode(data);
    }
}
