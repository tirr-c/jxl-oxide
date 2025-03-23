use crate::RenderingIntent;

/// Color management system that handles ICCv4 profiles.
pub trait ColorManagementSystem {
    /// Prepares color transformation between two ICC profiles.
    ///
    /// # Errors
    /// This function will return an error if the internal CMS implementation returned an error.
    fn prepare_transform(
        &self,
        from_icc: &[u8],
        to_icc: &[u8],
        intent: RenderingIntent,
    ) -> Result<Box<dyn PreparedTransform>, Box<dyn std::error::Error + Send + Sync + 'static>>;

    /// Returns whether the CMS supports linear transfer function.
    ///
    /// This method will return `false` if it doesn't support (or it lacks precision to handle)
    /// linear transfer function.
    fn supports_linear_tf(&self) -> bool {
        true
    }
}

/// Prepared transformation created by [`ColorManagementSystem`].
///
/// Prepared transformations may be cached by jxl-oxide internally.
pub trait PreparedTransform: Send + Sync {
    /// The number of expected input channels.
    fn num_input_channels(&self) -> usize;

    /// The number of expected output channels.
    fn num_output_channels(&self) -> usize;

    /// Performs prepared color transformation.
    ///
    /// # Errors
    /// This function will return an error if the internal CMS implementation returned an error.
    fn transform(
        &self,
        channels: &mut [&mut [f32]],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync + 'static>>;
}

/// "Null" color management system that fails on every operation.
#[derive(Debug, Copy, Clone)]
pub struct NullCms;
impl ColorManagementSystem for NullCms {
    fn prepare_transform(
        &self,
        _: &[u8],
        _: &[u8],
        _: RenderingIntent,
    ) -> Result<Box<dyn PreparedTransform>, Box<dyn std::error::Error + Send + Sync + 'static>>
    {
        Err(Box::new(crate::Error::CmsNotAvailable))
    }
}
