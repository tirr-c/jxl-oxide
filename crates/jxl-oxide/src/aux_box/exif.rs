use crate::Result;

/// Raw Exif metadata.
pub struct RawExif<'image> {
    tiff_header_offset: u32,
    payload: &'image [u8],
}

impl std::fmt::Debug for RawExif<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RawExif")
            .field("tiff_header_offset", &self.tiff_header_offset)
            .finish_non_exhaustive()
    }
}

impl<'image> RawExif<'image> {
    pub(crate) fn new(box_data: &'image [u8]) -> Result<Self> {
        if box_data.len() < 4 {
            tracing::error!(len = box_data.len(), "Exif box is too short");
            return Err(jxl_bitstream::Error::ValidationFailed("Exif box is too short").into());
        }

        let (tiff, payload) = box_data.split_at(4);
        let tiff_header_offset = u32::from_be_bytes([tiff[0], tiff[1], tiff[2], tiff[3]]);
        if tiff_header_offset as usize >= payload.len() {
            tracing::error!(
                payload_len = payload.len(),
                tiff_header_offset,
                "tiff_header_offset of Exif box is too large"
            );
            return Err(jxl_bitstream::Error::ValidationFailed(
                "tiff_header_offset of Exif box is too large",
            )
            .into());
        }

        Ok(Self {
            tiff_header_offset,
            payload,
        })
    }
}

impl<'image> RawExif<'image> {
    /// Returns the offset of TIFF header within the payload.
    pub fn tiff_header_offset(&self) -> u32 {
        self.tiff_header_offset
    }

    /// Returns the payload.
    pub fn payload(&self) -> &'image [u8] {
        self.payload
    }
}
