use honggfuzz::fuzz;
use jxl_oxide::JxlImage;

fn main() {
    loop {
        fuzz!(|data: &[u8]| {
            let _ = JxlImage::from_reader(std::io::Cursor::new(data));
        });
    }
}

#[cfg(test)]
mod tests{
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
}