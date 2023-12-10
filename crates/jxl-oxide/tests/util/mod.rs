#![allow(dead_code)]

use lcms2::{Profile, Transform};

pub struct Lcms2;
impl jxl_oxide::ColorManagementSystem for Lcms2 {
    fn transform_impl(
        &self,
        from: &[u8],
        to: &[u8],
        intent: jxl_oxide::color::RenderingIntent,
        channels: &mut [&mut [f32]],
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync + 'static>> {
        use lcms2::ColorSpaceSignatureExt;

        let from_profile = Profile::new_icc(from)?;
        let from_channels = from_profile.color_space().channels() as usize;
        let to_profile = Profile::new_icc(to)?;
        let to_channels = to_profile.color_space().channels() as usize;
        let max_channels = from_channels.max(to_channels);
        assert!(channels.len() >= max_channels);

        #[allow(clippy::unusual_byte_groupings)]
        let format_base = 0b010_00000_000000_000_0000_100;
        let from_pixel_format = lcms2::PixelFormat(format_base | ((from_channels as u32) << 3));
        let to_pixel_format = lcms2::PixelFormat(format_base | ((to_channels as u32) << 3));
        let transform = Transform::new(
            &from_profile,
            from_pixel_format,
            &to_profile,
            to_pixel_format,
            match intent {
                jxl_oxide::color::RenderingIntent::Perceptual => lcms2::Intent::Perceptual,
                jxl_oxide::color::RenderingIntent::Relative => lcms2::Intent::RelativeColorimetric,
                jxl_oxide::color::RenderingIntent::Saturation => lcms2::Intent::Saturation,
                jxl_oxide::color::RenderingIntent::Absolute => lcms2::Intent::AbsoluteColorimetric,
            },
        )?;

        let mut buf_in = vec![0f32; 1024 * from_channels];
        let mut buf_out = vec![0f32; 1024 * to_channels];
        let len = channels.iter().map(|x| x.len()).min().unwrap();
        for idx in (0..len).step_by(1024) {
            let chunk_len = (len - idx).min(1024);
            for k in 0..chunk_len {
                for (channel_idx, ch) in channels[..from_channels].iter().enumerate() {
                    buf_in[k * from_channels + channel_idx] = ch[idx + k];
                }
            }
            unsafe {
                let buf_in_ptr = buf_in.as_ptr();
                let buf_out_ptr = buf_out.as_mut_ptr();
                let transform_buf_in = std::slice::from_raw_parts(buf_in_ptr as *const u8, chunk_len * from_channels * std::mem::size_of::<f32>());
                let transform_buf_out = std::slice::from_raw_parts_mut(buf_out_ptr as *mut u8, chunk_len * to_channels * std::mem::size_of::<f32>());
                transform.transform_pixels(transform_buf_in, transform_buf_out);
            }
            for k in 0..chunk_len {
                for (channel_idx, ch) in channels[..to_channels].iter_mut().enumerate() {
                    ch[idx + k] = buf_out[k * to_channels + channel_idx];
                }
            }
        }

        Ok(to_channels)
    }
}

pub fn conformance_path(name: &str) -> std::path::PathBuf {
    let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/conformance/testcases");
    path.push(name);
    path.push("input.jxl");
    path
}
