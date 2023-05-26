use jxl_oxide::JxlImage;
use std::{error::Error, env::args, path::PathBuf};

fn main() {
    let path = args().skip(1).next().unwrap();
    match jxl_decode(path) {
        Ok(_) => (),
        Err(e) => eprintln!("{}", e),
    };
}

fn jxl_decode(path: String) -> Result<(), Box<dyn Error + Send + Sync>> {
    let mut image = JxlImage::open(PathBuf::from(path))?;
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
