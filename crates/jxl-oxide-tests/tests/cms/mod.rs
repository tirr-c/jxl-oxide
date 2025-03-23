use std::path::PathBuf;

use jxl_oxide::color::TransferFunction;
use jxl_oxide::{EnumColourEncoding, JxlImage, RenderingIntent};
use ssimulacra2::LinearRgb;

#[test]
fn cmyk_lcms2_moxcms() {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/cms/cmyk_layers.jxl");

    let mut image = JxlImage::builder().open(&path).unwrap();

    let width = image.width() as usize;
    let height = image.height() as usize;
    image.request_color_encoding(EnumColourEncoding {
        tf: TransferFunction::Linear,
        ..EnumColourEncoding::srgb(RenderingIntent::Relative)
    });

    image.set_cms(jxl_oxide::Lcms2);
    let lcms2_frame = {
        let render = image.render_frame(0).unwrap();
        let [r, g, b] = render.color_channels() else {
            unreachable!()
        };
        let [r, g, b] = [r, g, b].map(|x| x.as_float().unwrap().buf());

        let mut frame = Vec::with_capacity(width * height);
        for ((&r, &g), &b) in r.iter().zip(g).zip(b) {
            frame.push([r, g, b]);
        }

        frame
    };

    image.set_cms(jxl_oxide::Moxcms);
    let moxcms_frame = {
        let render = image.render_frame(0).unwrap();
        let [r, g, b] = render.color_channels() else {
            unreachable!()
        };
        let [r, g, b] = [r, g, b].map(|x| x.as_float().unwrap().buf());

        let mut frame = Vec::with_capacity(width * height);
        for ((&r, &g), &b) in r.iter().zip(g).zip(b) {
            frame.push([r, g, b]);
        }

        frame
    };

    let score = ssimulacra2::compute_frame_ssimulacra2(
        LinearRgb::new(lcms2_frame, width, height).unwrap(),
        LinearRgb::new(moxcms_frame, width, height).unwrap(),
    )
    .unwrap();

    assert!(
        score >= 90.0,
        "SSIMULACRA2 score too low (score < 90.0), score = {score}"
    );
}
