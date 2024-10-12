use std::io::Cursor;

use jxl_oxide::{CropInfo, JxlImage, Render};
use rand::prelude::*;

mod util;

fn run_test(buf: &[u8], name: &str) {
    let mut rng = rand::rngs::SmallRng::from_entropy();

    let image = JxlImage::builder()
        .read(Cursor::new(buf))
        .expect("Failed to open file");

    let width = image.width();
    let height = image.height();
    let width_dist = rand::distributions::Uniform::new_inclusive(128, (width / 2).max(128));
    let height_dist = rand::distributions::Uniform::new_inclusive(128, (height / 2).max(128));

    let mut tester_image = JxlImage::builder()
        .read(Cursor::new(buf))
        .expect("Failed to open file");

    for _ in 0..4 {
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
        tester_image.set_image_region(crop);
        test_crop_region(&image, &tester_image, crop, name);
    }
}

fn test_crop_region(image: &JxlImage, tester_image: &JxlImage, crop: CropInfo, name: &str) {
    eprintln!("Testing crop region {crop:?}");

    let CropInfo {
        width: crop_width,
        left: crop_left,
        top: crop_top,
        ..
    } = crop;
    let width = image.width();

    let num_frames = image.num_loaded_keyframes();
    for idx in 0..num_frames {
        eprintln!("Testing frame #{idx}");
        let full_render = image
            .render_frame(idx)
            .expect("Failed to render full image");
        let cropped_render = tester_image
            .render_frame_cropped(idx)
            .expect("Failed to render cropped image");

        for (c, (expected, actual)) in full_render
            .image_planar()
            .into_iter()
            .zip(cropped_render.image_planar())
            .enumerate()
        {
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
                        eprintln!("Test failed at c={c}, x={x}, y={y} (expected={expected}, actual={actual})");

                        let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
                        path.push("tests/.artifact");
                        std::fs::create_dir_all(&path).unwrap();

                        let mut full = path.clone();
                        full.push(format!("{name}-full.npy"));
                        write_npy(&full_render, &full);
                        eprintln!("Full frame data written at {}", full.to_string_lossy());

                        let mut cropped = path.clone();
                        cropped.push(format!("{name}-cropped.npy"));
                        write_npy(&cropped_render, &cropped);
                        eprintln!(
                            "Cropped frame data written at {}",
                            cropped.to_string_lossy()
                        );

                        panic!("Test failed at c={c}, x={x}, y={y} (expected={expected}, actual={actual})");
                    }
                }
            }
        }

        eprintln!("Frame #{idx} OK");
    }

    eprintln!("Crop region {crop:?} OK");
    eprintln!();
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

macro_rules! testcase_with_crop {
    {$($(#[$attr:meta])* $name:ident: $testimage:ident [$($region:expr),* $(,)?]),* $(,)?} => {
        $(
            #[test]
            $(#[$attr])*
            fn $name() {
                let path = util::conformance_path(stringify!($testimage));
                let buf = std::fs::read(path).expect("Failed to open file");

                let mut image = JxlImage::builder().read(Cursor::new(&buf)).expect("Failed to open file");
                let mut tester_image = JxlImage::builder().read(Cursor::new(&buf)).expect("Failed to open file");

                let regions = [
                    $($region,)*
                ];

                for region in regions {
                    tester_image.set_image_region(region);
                    test_crop_region(&mut image, &mut tester_image, region, stringify!($name));
                }
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

testcase_with_crop! {
    crop_sunset_logo_0: sunset_logo[CropInfo { width: 179, height: 258, left: 527, top: 298 }],
    crop_progressive_0: progressive[CropInfo { width: 315, height: 571, left: 1711, top: 800 }],
    crop_progressive_1: progressive[CropInfo { width: 1159, height: 359, left: 776, top: 1745 }],
    crop_noise_0: noise[CropInfo { width: 195, height: 162, left: 169, top: 194 }],
    crop_blendmodes_0: blendmodes[CropInfo { width: 242, height: 163, left: 81, top: 302 }],
    crop_alpha_triangles_triple: alpha_triangles[
        CropInfo { width: 460, height: 325, left: 468, top: 356 },
        CropInfo { width: 361, height: 147, left: 524, top: 475 },
    ],
    crop_progressive_triple: progressive[
        CropInfo { width: 941, height: 659, left: 1893, top: 35 },
        CropInfo { width: 1847, height: 1220, left: 850, top: 929 },
        CropInfo { width: 1421, height: 814, left: 1568, top: 1460 },
    ],
    crop_bike_0: bike[CropInfo { width: 936, height: 137, left: 877, top: 2353 }],
    #[ignore = "fails only on CI with emulated aarch64"]
    crop_upsampling_0: upsampling[CropInfo { width: 368, height: 128, left: 90, top: 460 }],
}

fn write_npy(render: &Render, path: impl AsRef<std::path::Path>) {
    use std::io::prelude::*;

    let file = std::fs::File::create(path).unwrap();
    let mut file = std::io::BufWriter::new(file);

    file.write_all(b"\x93NUMPY\x01\x00").unwrap();

    let image = render.image_all_channels();
    let num_channels = image.channels();
    let width = image.width();
    let height = image.height();
    let header_string = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': (1, {height}, {width}, {num_channels}), }}\n");
    eprintln!("width={width}, height={height}, num_channels={num_channels}");
    let header_len = header_string.len() as u16;
    file.write_all(&header_len.to_le_bytes()).unwrap();
    file.write_all(header_string.as_bytes()).unwrap();
    file.flush().unwrap();

    for &f in image.buf() {
        let b = f.to_bits().to_le_bytes();
        file.write_all(&b).unwrap();
    }

    file.flush().unwrap();
}
