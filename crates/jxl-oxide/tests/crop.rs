use std::io::Cursor;

use rand::prelude::*;
use jxl_oxide::{JxlImage, CropInfo};

mod util;

fn run_test(buf: &[u8]) {
    let mut rng = rand::rngs::SmallRng::from_entropy();

    let mut image = JxlImage::from_reader(Cursor::new(buf)).expect("Failed to open file");
    let width = image.width();
    let height = image.height();
    let width_dist = rand::distributions::Uniform::new_inclusive(128, (width / 2).max(128));
    let height_dist = rand::distributions::Uniform::new_inclusive(128, (height / 2).max(128));

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

    let mut tester_image = JxlImage::from_reader(Cursor::new(buf)).expect("Failed to open file");
    loop {
        let idx = match image.load_next_frame().expect("Failed to load frame") {
            jxl_oxide::LoadResult::Done(idx) => idx,
            jxl_oxide::LoadResult::NeedMoreData => panic!("Failed to load frame"),
            jxl_oxide::LoadResult::NoMoreFrames => break,
        };
        tester_image.load_next_frame().unwrap();

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
            for (expected_row, actual_row) in it {
                let expected_row = &expected_row[crop_left as usize..][..crop_width as usize];
                assert_eq!(expected_row, actual_row);
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
                run_test(&buf);
            }
        )*
    };
}

testcase! {
    bicycles,
    bike,
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
}
