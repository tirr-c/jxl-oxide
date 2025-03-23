use moxcms::{DataColorSpace, Layout};

use crate::RenderingIntent;

/// `moxcms` crate integration for ICC profile handling.
pub struct Moxcms;

struct PreparedTransform {
    transform: Box<moxcms::TransformF32BitExecutor>,
    from_channels: usize,
    to_channels: usize,
}

fn color_space_to_layout(color_space: DataColorSpace) -> Result<(Layout, usize), moxcms::CmsError> {
    match color_space {
        DataColorSpace::Xyz
        | DataColorSpace::Lab
        | DataColorSpace::Luv
        | DataColorSpace::YCbr
        | DataColorSpace::Yxy
        | DataColorSpace::Rgb
        | DataColorSpace::Hsv
        | DataColorSpace::Hls
        | DataColorSpace::Cmy => Ok((Layout::Rgb, 3)),
        DataColorSpace::Gray => Ok((Layout::Gray, 1)),
        DataColorSpace::Cmyk => Ok((Layout::Rgba, 4)),
        _ => Err(moxcms::CmsError::UnsupportedProfileConnection),
    }
}

impl crate::ColorManagementSystem for Moxcms {
    fn prepare_transform(
        &self,
        from_icc: &[u8],
        to_icc: &[u8],
        intent: RenderingIntent,
    ) -> Result<
        Box<dyn jxl_color::PreparedTransform>,
        Box<dyn std::error::Error + Send + Sync + 'static>,
    > {
        let profile_from = moxcms::ColorProfile::new_from_slice(from_icc)?;
        let profile_to = moxcms::ColorProfile::new_from_slice(to_icc)?;

        let (layout_from, from_channels) = color_space_to_layout(profile_from.color_space)?;
        let (layout_to, to_channels) = color_space_to_layout(profile_to.color_space)?;

        let rendering_intent = match intent {
            RenderingIntent::Perceptual => moxcms::RenderingIntent::Perceptual,
            RenderingIntent::Relative => moxcms::RenderingIntent::RelativeColorimetric,
            RenderingIntent::Saturation => moxcms::RenderingIntent::Saturation,
            RenderingIntent::Absolute => moxcms::RenderingIntent::AbsoluteColorimetric,
        };
        let options = moxcms::TransformOptions {
            rendering_intent,
            ..Default::default()
        };

        let transform =
            profile_from.create_transform_f32(layout_from, &profile_to, layout_to, options)?;

        let prepared = PreparedTransform {
            transform,
            from_channels,
            to_channels,
        };

        Ok(Box::new(prepared) as Box<_>)
    }
}

impl crate::PreparedTransform for PreparedTransform {
    fn num_input_channels(&self) -> usize {
        self.from_channels
    }

    fn num_output_channels(&self) -> usize {
        self.to_channels
    }

    fn transform(
        &self,
        channels: &mut [&mut [f32]],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync + 'static>> {
        let Self {
            ref transform,
            from_channels,
            to_channels,
        } = *self;

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

            transform.transform(
                &buf_in[..chunk_len * from_channels],
                &mut buf_out[..chunk_len * to_channels],
            )?;

            for k in 0..chunk_len {
                for (channel_idx, ch) in channels[..to_channels].iter_mut().enumerate() {
                    ch[idx + k] = buf_out[k * to_channels + channel_idx];
                }
            }
        }

        Ok(())
    }
}
