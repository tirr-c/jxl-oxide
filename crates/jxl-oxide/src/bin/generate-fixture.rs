use std::io::prelude::*;

fn main() {
    let stdin = std::io::stdin().lock();
    let mut stdout = std::io::stdout().lock();
    let mut image = jxl_oxide::JxlImage::from_reader(stdin).unwrap();

    let mut header = [0u8; 12];
    let image_header = image.image_header();
    header[0..4].copy_from_slice(&image_header.size.width.to_le_bytes());
    header[4..8].copy_from_slice(&image_header.size.height.to_le_bytes());
    let color_channels = if image_header.metadata.grayscale() { 1u32 } else { 3 };
    let channels = color_channels + image_header.metadata.ec_info.len() as u32;
    header[8..12].copy_from_slice(&channels.to_le_bytes());
    stdout.write_all(&header).unwrap();

    for idx in 0..image.num_loaded_keyframes() {
        let frame = image.render_frame(idx).unwrap();
        stdout.write_all(&[0]).unwrap();

        let color_channels = frame.color_channels();
        let extra_channels = frame.extra_channels();
        for grid in color_channels.iter().chain(extra_channels.iter().map(|ec| ec.grid())) {
            let buf = grid.buf();
            let mut output_buf = vec![0u8; grid.width() * grid.height() * 2];
            for (&sample_float, output) in buf.iter().zip(output_buf.chunks_exact_mut(2)) {
                let sample = (sample_float * 65535.0 + 0.5) as u16;
                output.copy_from_slice(&sample.to_le_bytes());
            }
            stdout.write_all(&output_buf).unwrap();
        }
    }
    stdout.write_all(&[0xff]).unwrap();
}
