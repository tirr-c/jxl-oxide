use std::io::prelude::*;

use crate::Error;

#[derive(Debug, Clone)]
pub struct ContainerBoxHeader {
    ty: ContainerBoxType,
    size: Option<u64>,
    is_last: bool,
}

impl ContainerBoxHeader {
    pub fn parse<R: Read>(mut reader: R) -> std::io::Result<Self> {
        let mut sbox = [0u8; 4];
        reader.read_exact(&mut sbox)?;
        let sbox = u32::from_be_bytes(sbox);

        let mut tbox = [0u8; 4];
        reader.read_exact(&mut tbox)?;
        let tbox = ContainerBoxType(tbox);

        let size = if sbox == 1 {
            let mut xlbox = [0u8; 8];
            reader.read_exact(&mut xlbox)?;
            let xlbox = u64::from_be_bytes(xlbox).checked_sub(16).ok_or(
                std::io::Error::new(std::io::ErrorKind::InvalidData, Error::InvalidBoxSize)
            )?;
            Some(xlbox)
        } else if sbox == 0 {
            None
        } else {
            let sbox = sbox.checked_sub(8).ok_or(
                std::io::Error::new(std::io::ErrorKind::InvalidData, Error::InvalidBoxSize)
            )?;
            Some(sbox as u64)
        };
        let is_last = size.is_none();

        Ok(Self {
            ty: tbox,
            size,
            is_last,
        })
    }
}

impl ContainerBoxHeader {
    #[inline]
    pub fn box_type(&self) -> ContainerBoxType {
        self.ty
    }

    #[inline]
    pub fn size(&self) -> Option<u64> {
        self.size
    }

    #[inline]
    pub fn is_last(&self) -> bool {
        self.is_last
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct ContainerBoxType(pub [u8; 4]);

impl ContainerBoxType {
    pub const JXL: Self = Self(*b"JXL ");
    pub const FILE_TYPE: Self = Self(*b"ftyp");
    pub const JXL_LEVEL: Self = Self(*b"jxll");
    pub const JUMBF: Self = Self(*b"jumb");
    pub const EXIF: Self = Self(*b"Exif");
    pub const XML: Self = Self(*b"xml ");
    pub const BROTLI_COMPRESSED: Self = Self(*b"brob");
    pub const FRAME_INDEX: Self = Self(*b"jxli");
    pub const CODESTREAM: Self = Self(*b"jxlc");
    pub const PARTIAL_CODESTREAM: Self = Self(*b"jxlp");
    pub const JPEG_RECONSTRUCTION: Self = Self(*b"jbrd");
}
