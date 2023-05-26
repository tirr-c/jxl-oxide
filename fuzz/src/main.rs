use afl::fuzz;
use jxl_oxide::JxlImage;
use std::{error::Error, io::BufReader};

fn main() {
    fuzz!(|data| {
        let _ = jxl_decode(data);
    })
}

fn jxl_decode(data: &[u8]) -> Result<(), Box<dyn Error + Send + Sync>> {
    let reader = BufReader::new(data);
    let mut image = JxlImage::from_reader(reader)?;
    let header = image.image_header();
    let max_size = u32::max(header.size.width, header.size.height);

    // Skip huge images
    if max_size > 65536 {
        return Ok(());
    }

    let mut keyframes = Vec::new();
    let mut renderer = image.renderer();
    loop {
        let result = renderer.render_next_frame()?;
        match result {
            jxl_oxide::RenderResult::Done(frame) => keyframes.push(frame),
            jxl_oxide::RenderResult::NeedMoreData => {
                // unexpected end of file
                return Ok(());
            }
            jxl_oxide::RenderResult::NoMoreFrames => break,
        }
    }

    for keyframe in keyframes {
        let fb = keyframe.image();
        fb.buf().into_iter().for_each(drop);
    }

    Ok(())
}
