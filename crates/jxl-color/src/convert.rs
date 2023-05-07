use jxl_grid::SimpleGrid;

use crate::{
    ciexyz::*,
    consts::*,
    tf,
    ColourEncoding,
    Primaries,
    TransferFunction,
    WhitePoint,
};

/// Converts given framebuffer to the target color encoding.
///
/// Assumes that input framebuffer is in linear sRGB.
pub fn from_linear_srgb(fb: &mut [SimpleGrid<f32>], encoding: &ColourEncoding, intensity_target: f32) {
    let target_wp = match &encoding.white_point {
        WhitePoint::D65 => illuminant::D65,
        WhitePoint::Custom(xy) => [xy.x as f32 / 1e6, xy.y as f32 / 1e6],
        WhitePoint::E => illuminant::E,
        WhitePoint::Dci => illuminant::DCI,
    };
    let target_primaries = match &encoding.primaries {
        Primaries::Srgb => primaries::SRGB,
        Primaries::Custom { red, green, blue } => [
            [red.x as f32 / 1e6, red.y as f32 / 1e6],
            [green.x as f32 / 1e6, green.y as f32 / 1e6],
            [blue.x as f32 / 1e6, blue.y as f32 / 1e6],
        ],
        Primaries::Bt2100 => primaries::BT2100,
        Primaries::P3 => primaries::P3,
    };

    let merged = (target_primaries != primaries::SRGB || target_wp != illuminant::D65).then(|| {
        let srgb_xyz = primaries_to_xyz_mat(primaries::SRGB, illuminant::D65);
        let xyz_target = xyz_to_primaries_mat(target_primaries, target_wp);

        let mut merged = srgb_xyz;
        if target_wp != illuminant::D65 {
            let adapt = adapt_mat(illuminant::D65, target_wp);
            merged = matmul3(&adapt, &merged);
        }
        matmul3(&xyz_target, &merged)
    });

    let [r, g, b, ..] = fb else { panic!() };
    let r = r.buf_mut();
    let g = g.buf_mut();
    let b = b.buf_mut();
    if let Some(merged) = &merged {
        for ((r, g), b) in r.iter_mut().zip(g.iter_mut()).zip(b.iter_mut()) {
            let [or, og, ob] = matmul3vec(merged, &[*r, *g, *b]);
            *r = or;
            *g = og;
            *b = ob;
        }
    }

    match encoding.tf {
        TransferFunction::Gamma(gamma) => {
            let gamma = gamma as f32 / 1e7;
            tf::linear_to_gamma(r, gamma);
            tf::linear_to_gamma(g, gamma);
            tf::linear_to_gamma(b, gamma);
        },
        TransferFunction::Bt709 => {
            tf::linear_to_bt709(r);
            tf::linear_to_bt709(g);
            tf::linear_to_bt709(b);
        },
        TransferFunction::Unknown => {}
        TransferFunction::Linear => {},
        TransferFunction::Srgb => {
            tf::linear_to_srgb(r);
            tf::linear_to_srgb(g);
            tf::linear_to_srgb(b);
        },
        TransferFunction::Pq => {
            tf::linear_to_pq(r, intensity_target);
            tf::linear_to_pq(g, intensity_target);
            tf::linear_to_pq(b, intensity_target);
        },
        TransferFunction::Dci => {
            let gamma = 1.0 / 2.6;
            tf::linear_to_gamma(r, gamma);
            tf::linear_to_gamma(g, gamma);
            tf::linear_to_gamma(b, gamma);
        },
        TransferFunction::Hlg => {
            let luminances = {
                let xyz = primaries_to_xyz_mat(target_primaries, target_wp);
                [xyz[3], xyz[4], xyz[5]]
            };
            tf::hlg_inverse_oo([r, g, b], luminances, intensity_target);
            tf::linear_to_hlg(r);
            tf::linear_to_hlg(g);
            tf::linear_to_hlg(b);
        },
    }
}
