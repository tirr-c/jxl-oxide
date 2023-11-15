use std::io::Cursor;

use rand::prelude::*;
use jxl_oxide::{JxlImage, CropInfo, Render};

mod util;

fn run_test(buf: &[u8], name: &str) {
    let is_ci = std::env::var_os("CI").map(|v| !v.is_empty()).unwrap_or(false);
    let mut rng = rand::rngs::SmallRng::from_entropy();

    let mut image = JxlImage::from_reader(Cursor::new(buf)).expect("Failed to open file");
    let width = image.width();
    let height = image.height();
    let width_dist = rand::distributions::Uniform::new_inclusive(128, (width / 2).max(128));
    let height_dist = rand::distributions::Uniform::new_inclusive(128, (height / 2).max(128));

    let mut tester_image = JxlImage::from_reader(Cursor::new(buf)).expect("Failed to open file");

    let num_frames = image.num_loaded_keyframes();
    for _ in 0..8 {
        let crop_width = width_dist.sample(&mut rng);
        let crop_height = height_dist.sample(&mut rng);
        let crop_left = rng.gen_range(0..=(width - crop_width));
        let crop_top = rng.gen_range(0..=(height - crop_height));
        let crop = CropInfo {
            left: crop_left,
            top: crop_top,
            width: crop_width,
            height: crop_height,
        };
        eprintln!("  Crop region: {:?}", crop);

        for idx in 0..num_frames {
            eprintln!("Testing frame #{idx}");
            let full_render = image.render_frame(idx).expect("Failed to render full image");
            let cropped_render = tester_image
                .render_frame_cropped(idx, Some(crop))
                .expect("Failed to render cropped image");

            for (expected, actual) in full_render.image_planar().into_iter().zip(cropped_render.image_planar()) {
                let expected = expected.buf();
                let actual = actual.buf();

                let it = expected
                    .chunks_exact(width as usize)
                    .skip(crop_top as usize)
                    .zip(actual.chunks_exact(crop_width as usize));
                for (y, (expected_row, actual_row)) in it.enumerate() {
                    let expected_row = &expected_row[crop_left as usize..][..crop_width as usize];
                    for (x, (expected, actual)) in expected_row.iter().zip(actual_row).enumerate() {
                        if (expected - actual).abs() > 1e-6 {
                            if is_ci {
                                eprintln!("Test failed at x={x}, y={y}");
                                let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
                                path.push("tests/.artifact");
                                std::fs::create_dir_all(&path).unwrap();

                                let mut full = path.clone();
                                full.push(format!("{name}-full.npy"));
                                write_npy(&full_render, full);

                                let mut cropped = path.clone();
                                cropped.push(format!("{name}-cropped.npy"));
                                write_npy(&cropped_render, cropped);
                            }
                            panic!();
                        }
                    }
                }
            }
        }
    }
}

macro_rules! testcase {
    {$($(#[$attr:meta])* $name:ident),* $(,)?} => {
        $(
            #[test]
            $(#[$attr])*
            fn $name() {
                let path = util::conformance_path(stringify!($name));
                let buf = std::fs::read(path).expect("Failed to open file");
                run_test(&buf, stringify!($name));
            }
        )*
    };
}

testcase! {
    bicycles,
    bike,
    alpha_triangles,
    lz77_flower,
    sunset_logo,
    blendmodes,
    progressive,
    animation_icos4d,
    animation_spline,
    animation_newtons_cradle,
    lossless_pfm,
    noise,
    cafe,
    upsampling,
    delta_palette,
    patches_lossless,
    grayscale,
    grayscale_jpeg,
    grayscale_public_university,
    spot,
}

fn write_npy(render: &Render, path: impl AsRef<std::path::Path>) {
    use std::io::prelude::*;

    let file = std::fs::File::create(path).unwrap();
    let mut file = std::io::BufWriter::new(file);

    file.write_all(b"\x93NUMPY\x01\x00").unwrap();

    let color_channels = render.color_channels();
    let extra_channels = render.extra_channels();

    let num_channels = color_channels.len() + extra_channels.len();
    let first_channel = &color_channels[0];
    let width = first_channel.width();
    let height = first_channel.height();
    let header_string = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': (1, {height}, {width}, {num_channels}), }}\n");
    eprintln!("width={width}, height={height}, num_channels={num_channels}");
    let header_len = header_string.len() as u16;
    file.write_all(&header_len.to_le_bytes()).unwrap();
    file.write_all(header_string.as_bytes()).unwrap();
    file.flush().unwrap();

    for y in 0..height {
        for x in 0..width {
            for cc in color_channels {
                let f = *cc.get(x, y).unwrap();
                let b = f.to_bits().to_le_bytes();
                file.write_all(&b).unwrap();
            }
            for ec in extra_channels {
                let f = *ec.grid().get(x, y).unwrap();
                let b = f.to_bits().to_le_bytes();
                file.write_all(&b).unwrap();
            }
        }
    }

    file.flush().unwrap();
}
