use crate::Error;

/// Box header used in JPEG XL containers.
#[derive(Debug, Clone)]
pub struct ContainerBoxHeader {
    pub(super) ty: ContainerBoxType,
    box_size: Option<u64>,
    is_last: bool,
}

/// Result of parsing container box header.
pub enum HeaderParseResult {
    /// Box header is read successfully.
    Done {
        header: ContainerBoxHeader,
        header_size: usize,
    },
    /// Parser needs more data to read a header.
    NeedMoreData,
}

impl ContainerBoxHeader {
    pub(super) fn parse(buf: &[u8]) -> Result<HeaderParseResult, Error> {
        let (tbox, box_size, header_size) = match *buf {
            #[rustfmt::skip]
            [0, 0, 0, 1, t0, t1, t2, t3, s0, s1, s2, s3, s4, s5, s6, s7, ..] => {
                let xlbox = u64::from_be_bytes([s0, s1, s2, s3, s4, s5, s6, s7]);
                let tbox = ContainerBoxType([t0, t1, t2, t3]);
                let xlbox = xlbox.checked_sub(16).ok_or(Error::InvalidBox)?;
                (tbox, Some(xlbox), 16)
            }
            [s0, s1, s2, s3, t0, t1, t2, t3, ..] => {
                let sbox = u32::from_be_bytes([s0, s1, s2, s3]);
                let tbox = ContainerBoxType([t0, t1, t2, t3]);
                let sbox = if sbox == 0 {
                    None
                } else if let Some(sbox) = sbox.checked_sub(8) {
                    Some(sbox as u64)
                } else {
                    return Err(Error::InvalidBox);
                };
                (tbox, sbox, 8)
            }
            _ => return Ok(HeaderParseResult::NeedMoreData),
        };
        let is_last = box_size.is_none();

        let header = Self {
            ty: tbox,
            box_size,
            is_last,
        };
        Ok(HeaderParseResult::Done {
            header,
            header_size,
        })
    }
}

impl ContainerBoxHeader {
    #[inline]
    pub fn box_type(&self) -> ContainerBoxType {
        self.ty
    }

    #[inline]
    pub fn box_size(&self) -> Option<u64> {
        self.box_size
    }

    #[inline]
    pub fn is_last(&self) -> bool {
        self.is_last
    }
}

/// Type of JPEG XL container box.
///
/// Known types are defined as associated consts.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct ContainerBoxType(pub [u8; 4]);

impl ContainerBoxType {
    /// JPEG XL signature box.
    pub const JXL: Self = Self(*b"JXL ");

    /// JPEG XL file type box.
    pub const FILE_TYPE: Self = Self(*b"ftyp");

    /// JPEG XL level box.
    pub const JXL_LEVEL: Self = Self(*b"jxll");

    /// JUMBF box.
    pub const JUMBF: Self = Self(*b"jumb");

    /// Exif box.
    pub const EXIF: Self = Self(*b"Exif");

    /// XML box, mainly for storing XMP metadata.
    pub const XML: Self = Self(*b"xml ");

    /// Brotli-compressed box.
    pub const BROTLI_COMPRESSED: Self = Self(*b"brob");

    /// Frame index box.
    pub const FRAME_INDEX: Self = Self(*b"jxli");

    /// JPEG XL codestream box.
    pub const CODESTREAM: Self = Self(*b"jxlc");

    /// JPEG XL partial codestream box.
    pub const PARTIAL_CODESTREAM: Self = Self(*b"jxlp");

    /// JPEG bistream reconstruction data box.
    pub const JPEG_RECONSTRUCTION: Self = Self(*b"jbrd");

    /// HDR gain map box.
    pub const HDR_GAIN_MAP: Self = Self(*b"jhgm");
}
