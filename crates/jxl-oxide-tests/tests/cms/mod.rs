use std::path::PathBuf;

use jxl_oxide::{EnumColourEncoding, JxlImage, RenderingIntent};

#[test]
fn cmyk_lcms2_moxcms() {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/cms/cmyk_layers.jxl");

    let mut image_lcms2 = JxlImage::builder().open(&path).unwrap();
    image_lcms2.set_cms(jxl_oxide::Lcms2);
    image_lcms2.request_color_encoding(EnumColourEncoding::srgb(RenderingIntent::Relative));

    let render = image_lcms2.render_frame(0).unwrap();
    let lcms2_channels = render.color_channels();

    let mut image_moxcms = JxlImage::builder().open(&path).unwrap();
    image_moxcms.set_cms(jxl_oxide::Moxcms);
    image_moxcms.request_color_encoding(EnumColourEncoding::srgb(RenderingIntent::Relative));

    let render = image_moxcms.render_frame(0).unwrap();
    let moxcms_channels = render.color_channels();

    let mut max_abs_error = 0f32;
    for (ch_idx, (lcms2, moxcms)) in std::iter::zip(lcms2_channels, moxcms_channels).enumerate() {
        let lcms2 = lcms2.as_float().unwrap();
        let moxcms = moxcms.as_float().unwrap();

        for y in 0..lcms2.height() {
            let lcms2 = lcms2.get_row(y).unwrap();
            let moxcms = moxcms.get_row(y).unwrap();
            for (x, (&lcms2, &moxcms)) in std::iter::zip(lcms2, moxcms).enumerate() {
                let abs_error = (lcms2 - moxcms).abs();
                max_abs_error = max_abs_error.max(abs_error);
                if abs_error >= 0.06 {
                    eprintln!("c={ch_idx}, x={x}, y={y}, abs_error={abs_error} (lcms2={lcms2}, moxcms={moxcms})");
                }
            }
        }
    }

    assert!(
        max_abs_error < 0.06,
        "max_abs_error < 0.06 failed (max_abs_error = {max_abs_error})"
    );
}
