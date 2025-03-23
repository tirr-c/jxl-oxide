use lcms2::{CacheFlag, Profile, ThreadContext, Transform};

use crate::RenderingIntent;

/// Little CMS 2 integration for ICC profile handling.
pub struct Lcms2;

struct PreparedTransform<C: lcms2::CacheFlag> {
    transform: Transform<u8, u8, ThreadContext, C>,
    from_channels: usize,
    to_channels: usize,
}

thread_local! {
    static LCMS2_CTX: lcms2::ThreadContext = lcms2::ThreadContext::new();
}

impl crate::ColorManagementSystem for Lcms2 {
    fn prepare_transform(
        &self,
        from_icc: &[u8],
        to_icc: &[u8],
        intent: RenderingIntent,
    ) -> Result<Box<dyn crate::PreparedTransform>, Box<dyn std::error::Error + Send + Sync + 'static>>
    {
        use lcms2::ColorSpaceSignatureExt;

        let prepared = LCMS2_CTX.with(|ctx| -> lcms2::LCMSResult<_> {
            let from_profile = Profile::new_icc_context(ctx, from_icc)?;
            let from_channels = from_profile.color_space().channels() as usize;
            let to_profile = Profile::new_icc_context(ctx, to_icc)?;
            let to_channels = to_profile.color_space().channels() as usize;

            #[allow(clippy::unusual_byte_groupings)]
            let format_base = 0b010_00000_000000_000_0000_100;
            let from_pixel_format = lcms2::PixelFormat(format_base | ((from_channels as u32) << 3));
            let to_pixel_format = lcms2::PixelFormat(format_base | ((to_channels as u32) << 3));
            let transform = Transform::new_flags_context(
                ctx,
                &from_profile,
                from_pixel_format,
                &to_profile,
                to_pixel_format,
                match intent {
                    RenderingIntent::Perceptual => lcms2::Intent::Perceptual,
                    RenderingIntent::Relative => lcms2::Intent::RelativeColorimetric,
                    RenderingIntent::Saturation => lcms2::Intent::Saturation,
                    RenderingIntent::Absolute => lcms2::Intent::AbsoluteColorimetric,
                },
                lcms2::Flags::NO_CACHE,
            )?;

            Ok(PreparedTransform {
                transform,
                from_channels,
                to_channels,
            })
        })?;

        Ok(Box::new(prepared) as Box<_>)
    }
}

impl<C> crate::PreparedTransform for PreparedTransform<C>
where
    C: CacheFlag,
    PreparedTransform<C>: Send + Sync,
{
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

            unsafe {
                let buf_in_ptr = buf_in.as_ptr();
                let buf_out_ptr = buf_out.as_mut_ptr();
                let transform_buf_in = std::slice::from_raw_parts(
                    buf_in_ptr as *const u8,
                    chunk_len * from_channels * std::mem::size_of::<f32>(),
                );
                let transform_buf_out = std::slice::from_raw_parts_mut(
                    buf_out_ptr as *mut u8,
                    chunk_len * to_channels * std::mem::size_of::<f32>(),
                );
                transform.transform_pixels(transform_buf_in, transform_buf_out);
            }

            for k in 0..chunk_len {
                for (channel_idx, ch) in channels[..to_channels].iter_mut().enumerate() {
                    ch[idx + k] = buf_out[k * to_channels + channel_idx];
                }
            }
        }

        Ok(())
    }
}
