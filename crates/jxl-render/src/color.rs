use jxl_bitstream::{header::ImageMetadata, Bitstream};

use crate::FrameBuffer;

pub fn read_icc<R: std::io::Read>(bitstream: &mut Bitstream<R>) -> crate::Result<Vec<u8>> {
    fn get_icc_ctx(idx: usize, b1: u8, b2: u8) -> u32 {
        if idx <= 128 {
            return 0;
        }

        let p1 = match b1 {
            | b'a'..=b'z'
            | b'A'..=b'Z' => 0,
            | b'0'..=b'9'
            | b'.'
            | b',' => 1,
            | 0..=1 => 2 + b1 as u32,
            | 2..=15 => 4,
            | 241..=254 => 5,
            | 255 => 6,
            | _ => 7,
        };
        let p2 = match b2 {
            | b'a'..=b'z'
            | b'A'..=b'Z' => 0,
            | b'0'..=b'9'
            | b'.'
            | b',' => 1,
            | 0..=15 => 2,
            | 241..=255 => 3,
            | _ => 4,
        };

        1 + p1 + 8 * p2
    }

    let enc_size = jxl_bitstream::read_bits!(bitstream, U64)?;
    let mut decoder = jxl_coding::Decoder::parse(bitstream, 41)?;

    let mut encoded_icc = vec![0u8; enc_size as usize];
    let mut b1 = 0u8;
    let mut b2 = 0u8;
    decoder.begin(bitstream).unwrap();
    for (idx, b) in encoded_icc.iter_mut().enumerate() {
        let sym = decoder.read_varint(bitstream, get_icc_ctx(idx, b1, b2))?;
        if sym >= 256 {
            panic!("Decoded symbol out of range");
        }
        *b = sym as u8;

        b2 = b1;
        b1 = *b;
    }

    Ok(encoded_icc)
}

pub fn perform_inverse_xyb(fb_yxb: &mut FrameBuffer, metadata: &ImageMetadata) {
    let itscale = 255.0 / metadata.tone_mapping.intensity_target as f64;
    let oim = &metadata.opsin_inverse_matrix;

    let ob = oim.opsin_bias.map(|v| v as f64);
    let cbrt_ob = ob.map(|v| v.cbrt());

    let inv_mat = oim.inv_mat.map(|r| r.map(|v| v as f64));

    let width = fb_yxb.width() as usize;
    let height = fb_yxb.height() as usize;
    let stride = fb_yxb.stride() as usize;

    let mut channels = fb_yxb.channel_buffers_mut().into_iter();
    let y = channels.next().unwrap();
    let x = channels.next().unwrap();
    let b = channels.next().unwrap();
    for r in 0..height {
        for c in 0..width {
            let idx = r * stride + c;
            let y = &mut y[idx];
            let x = &mut x[idx];
            let b = &mut b[idx];
            let g_l = (*y + *x) as f64;
            let g_m = (*y - *x) as f64;
            let g_s = *b as f64;

            let mix_l = ((g_l - cbrt_ob[0]).powi(3) + ob[0]) * itscale;
            let mix_m = ((g_m - cbrt_ob[1]).powi(3) + ob[1]) * itscale;
            let mix_s = ((g_s - cbrt_ob[2]).powi(3) + ob[2]) * itscale;

            *y = (inv_mat[0][0] * mix_l + inv_mat[0][1] * mix_m + inv_mat[0][2] * mix_s) as f32;
            *x = (inv_mat[1][0] * mix_l + inv_mat[1][1] * mix_m + inv_mat[1][2] * mix_s) as f32;
            *b = (inv_mat[2][0] * mix_l + inv_mat[2][1] * mix_m + inv_mat[2][2] * mix_s) as f32;
        }
    }
}

pub fn perform_inverse_ycbcr(fb_ycbcr: &mut FrameBuffer) {
    let width = fb_ycbcr.width() as usize;
    let height = fb_ycbcr.height() as usize;
    let stride = fb_ycbcr.stride() as usize;

    let mut channels = fb_ycbcr.channel_buffers_mut().into_iter();
    let y = channels.next().unwrap();
    let cb = channels.next().unwrap();
    let cr = channels.next().unwrap();
    for r in 0..height {
        for c in 0..width {
            let idx = r * stride + c;
            let r = &mut y[idx];
            let g = &mut cb[idx];
            let b = &mut cr[idx];
            let y = *r + 0.5;
            let cb = *g;
            let cr = *b;

            *r = y + 1.402 * cr;
            *g = y - 0.344016 * cb - 0.714136 * cr;
            *b = y + 1.772 * cb;
        }
    }
}
