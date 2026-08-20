//! Tests for 1:8 LF-only rendering (`JxlImage::set_lf_only`).
//!
//! Three things have to hold, and only the first is obvious:
//!
//! 1. The image really comes back at 1:8 of the CODED frame, which is 8 x the frame's own
//!    `upsampling` smaller than the final image. Getting that factor wrong does not fail
//!    loudly; it puts the picture in a corner of a buffer several times too large.
//! 2. The picture is the same picture. An LF image that decoded to noise, or to the wrong
//!    colour space, would still be the right SIZE, so size alone proves very little. Each
//!    case is compared against the full render box-downsampled to the same grid.
//! 3. Modular frames, which have no LF image, still render at 1:1 rather than failing.

use std::io::Cursor;

use jxl_oxide::JxlImage;
use jxl_oxide_tests as util;

/// Mean absolute difference per channel sample, in the renders' own float units.
///
/// The LF image is the DC of each 8x8 block, so against a box average of the full render it
/// should agree closely; what it cannot reproduce is detail finer than a block, which a box
/// average of the full render has already removed.
fn mean_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    if a.is_empty() {
        return 0.0;
    }
    let sum: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs() as f64)
        .sum::<f64>();
    (sum / a.len() as f64) as f32
}

fn run_test(buf: &[u8], name: &str) {
    let full = JxlImage::builder()
        .read(Cursor::new(buf))
        .expect("Failed to open file");
    let mut lf = JxlImage::builder()
        .read(Cursor::new(buf))
        .expect("Failed to open file");
    lf.set_lf_only(true);
    assert!(lf.lf_only());

    let (rw, rh) = lf.render_size(0).expect("keyframe 0 must exist");
    let full_render = full.render_frame(0).expect("full render failed");
    let lf_render = lf.render_frame(0).expect("lf-only render failed");

    let full_fb = full_render.image_all_channels();
    let lf_fb = lf_render.image_all_channels();

    eprintln!(
        "{name}: {}x{} -> {}x{} (reported {rw}x{rh})",
        full_fb.width(),
        full_fb.height(),
        lf_fb.width(),
        lf_fb.height(),
    );

    // `render_size` must agree with what actually came out, or callers that size a buffer
    // from it (as the CLI does) write into the wrong shape.
    assert_eq!(
        (lf_fb.width() as u32, lf_fb.height() as u32),
        (rw, rh),
        "{name}: render_size disagrees with the rendered image",
    );
    assert_eq!(
        lf_fb.channels(),
        full_fb.channels(),
        "{name}: LF render must expose the same channels as a full render",
    );

    let is_vardct = full
        .frame_header(0)
        .map(|h| h.encoding == jxl_oxide::frame::Encoding::VarDct)
        .unwrap_or(false);
    if !is_vardct {
        // Modular: the request is ignored and the frame renders at 1:1. That is the documented
        // behaviour, and it is why callers must read the size rather than assume it.
        assert_eq!(
            (lf_fb.width(), lf_fb.height()),
            (full_fb.width(), full_fb.height()),
            "{name}: a modular frame must ignore the LF request and render at 1:1",
        );
        return;
    }

    assert!(
        lf_fb.width() < full_fb.width() && lf_fb.height() < full_fb.height(),
        "{name}: a VarDCT frame must actually be reduced",
    );

    // Box-downsample the full render onto the LF grid and compare. This is the real check:
    // the same size can be reached with completely wrong pixels.
    let (fw, fh, ch) = (full_fb.width(), full_fb.height(), full_fb.channels());
    let (lw, lh) = (lf_fb.width(), lf_fb.height());
    let full_buf = full_fb.buf();
    let mut reference = vec![0f32; lw * lh * ch];
    for y in 0..lh {
        let y0 = y * fh / lh;
        let y1 = (((y + 1) * fh).div_ceil(lh)).min(fh).max(y0 + 1);
        for x in 0..lw {
            let x0 = x * fw / lw;
            let x1 = (((x + 1) * fw).div_ceil(lw)).min(fw).max(x0 + 1);
            let n = ((y1 - y0) * (x1 - x0)) as f32;
            for c in 0..ch {
                let mut acc = 0f32;
                for yy in y0..y1 {
                    for xx in x0..x1 {
                        acc += full_buf[(yy * fw + xx) * ch + c];
                    }
                }
                reference[(y * lw + x) * ch + c] = acc / n;
            }
        }
    }

    let diff = mean_abs_diff(lf_fb.buf(), &reference);
    eprintln!("{name}: mean abs diff vs box-downsampled full render = {diff:.4}");

    // Generous on purpose. This is an approximation with the restoration filters and
    // upsampling skipped, so it is not expected to match exactly; what the bound catches is
    // the failure that matters, an LF path that decoded the wrong thing entirely. Renders are
    // in [0, 1]-ish float, so 0.05 is roughly 13 levels out of 255 on average.
    assert!(
        diff < 0.05,
        "{name}: LF render differs from the full render by {diff:.4} on average, which is far \
         more than an 8x downsample should",
    );
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
    sunset_logo,
    alpha_triangles,
    progressive,
    // Modular, so it must fall back to a 1:1 render rather than fail.
    lz77_flower,
}
