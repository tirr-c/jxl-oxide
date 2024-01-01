use crate::RenderingIntent;

/// Color management system that handles ICCv4 profiles.
///
/// Implementors can implement `transform_impl` to integrate into external color management system.
pub trait ColorManagementSystem {
    fn transform_impl(
        &self,
        from: &[u8],
        to: &[u8],
        intent: RenderingIntent,
        channels: &mut [&mut [f32]],
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync + 'static>>;

    /// Performs color transformation between two ICC profiles.
    ///
    /// # Errors
    /// This function will return an error if the internal CMS implementation returned an error.
    fn transform(
        &self,
        from: &[u8],
        to: &[u8],
        intent: RenderingIntent,
        channels: &mut [&mut [f32]],
    ) -> Result<usize, crate::Error> {
        self.transform_impl(from, to, intent, channels)
            .map_err(crate::Error::CmsFailure)
    }
}

/// "Null" color management system that fails on every operation.
#[derive(Debug, Copy, Clone)]
pub struct NullCms;
impl ColorManagementSystem for NullCms {
    fn transform_impl(
        &self,
        _: &[u8],
        _: &[u8],
        _: RenderingIntent,
        _: &mut [&mut [f32]],
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync + 'static>> {
        Err(Box::new(crate::Error::CmsNotAvailable))
    }

    fn transform(
        &self,
        _: &[u8],
        _: &[u8],
        _: RenderingIntent,
        _: &mut [&mut [f32]],
    ) -> Result<usize, crate::Error> {
        Err(crate::Error::CmsNotAvailable)
    }
}
