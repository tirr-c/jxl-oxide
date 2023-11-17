/// Varblock transform types.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
#[repr(u8)]
pub enum TransformType {
    Dct8 = 0,
    Hornuss,
    Dct2,
    Dct4,
    Dct16,
    Dct32,
    Dct16x8,
    Dct8x16,
    Dct32x8,
    Dct8x32,
    Dct32x16,
    Dct16x32,
    Dct4x8,
    Dct8x4,
    Afv0,
    Afv1,
    Afv2,
    Afv3,
    Dct64,
    Dct64x32,
    Dct32x64,
    Dct128,
    Dct128x64,
    Dct64x128,
    Dct256,
    Dct256x128,
    Dct128x256,
}

impl TryFrom<u8> for TransformType {
    type Error = jxl_bitstream::Error;

    fn try_from(value: u8) -> jxl_bitstream::Result<Self> {
        if value <= TransformType::Dct128x256 as u8 {
            // SAFETY: TransformType is repr(u8) and all value <= Dct128x256 is valid
            Ok(unsafe { std::mem::transmute::<u8, Self>(value) })
        } else {
            Err(jxl_bitstream::Error::InvalidEnum {
                name: "TransformType",
                value: value as u32,
            })
        }
    }
}

impl TransformType {
    /// Returns the size of the transform type, in 8x8 blocks.
    pub fn dct_select_size(self) -> (u32, u32) {
        use TransformType::*;

        match self {
            Dct8 | Hornuss | Dct2 | Dct4 | Dct4x8 | Dct8x4 | Afv0 | Afv1 | Afv2 | Afv3 => (1, 1),
            Dct16 => (2, 2),
            Dct32 => (4, 4),
            Dct16x8 => (1, 2),
            Dct8x16 => (2, 1),
            Dct32x8 => (1, 4),
            Dct8x32 => (4, 1),
            Dct32x16 => (2, 4),
            Dct16x32 => (4, 2),
            Dct64 => (8, 8),
            Dct64x32 => (4, 8),
            Dct32x64 => (8, 4),
            Dct128 => (16, 16),
            Dct128x64 => (8, 16),
            Dct64x128 => (16, 8),
            Dct256 => (32, 32),
            Dct256x128 => (16, 32),
            Dct128x256 => (32, 16),
        }
    }

    pub(crate) fn dequant_matrix_param_index(self) -> u32 {
        use TransformType::*;

        match self {
            Dct8 => 0,
            Hornuss => 1,
            Dct2 => 2,
            Dct4 => 3,
            Dct16 => 4,
            Dct32 => 5,
            Dct16x8 | Dct8x16 => 6,
            Dct32x8 | Dct8x32 => 7,
            Dct32x16 | Dct16x32 => 8,
            Dct4x8 | Dct8x4 => 9,
            Afv0 | Afv1 | Afv2 | Afv3 => 10,
            Dct64 => 11,
            Dct64x32 | Dct32x64 => 12,
            Dct128 => 13,
            Dct128x64 | Dct64x128 => 14,
            Dct256 => 15,
            Dct256x128 | Dct128x256 => 16,
        }
    }

    pub(crate) fn dequant_matrix_size(self) -> (u32, u32) {
        use TransformType::*;

        match self {
            Dct8 | Hornuss | Dct2 | Dct4 | Dct4x8 | Dct8x4 | Afv0 | Afv1 | Afv2 | Afv3 => (8, 8),
            Dct16 => (16, 16),
            Dct32 => (32, 32),
            Dct16x8 | Dct8x16 => (16, 8),
            Dct32x8 | Dct8x32 => (32, 8),
            Dct32x16 | Dct16x32 => (32, 16),
            Dct64 => (64, 64),
            Dct64x32 | Dct32x64 => (64, 32),
            Dct128 => (128, 128),
            Dct128x64 | Dct64x128 => (128, 64),
            Dct256 => (256, 256),
            Dct256x128 | Dct128x256 => (256, 128),
        }
    }

    pub(crate) fn order_id(self) -> u32 {
        use TransformType::*;

        match self {
            Dct8 => 0,
            Hornuss | Dct2 | Dct4 | Dct4x8 | Dct8x4 | Afv0 | Afv1 | Afv2 | Afv3 => 1,
            Dct16 => 2,
            Dct32 => 3,
            Dct16x8 | Dct8x16 => 4,
            Dct32x8 | Dct8x32 => 5,
            Dct32x16 | Dct16x32 => 6,
            Dct64 => 7,
            Dct64x32 | Dct32x64 => 8,
            Dct128 => 9,
            Dct128x64 | Dct64x128 => 10,
            Dct256 => 11,
            Dct256x128 | Dct128x256 => 12,
        }
    }

    /// Returns whether DCT coefficients should be transposed.
    #[inline]
    pub fn need_transpose(&self) -> bool {
        use TransformType::*;

        if matches!(
            self,
            Hornuss | Dct2 | Dct4 | Dct4x8 | Dct8x4 | Afv0 | Afv1 | Afv2 | Afv3
        ) {
            false
        } else {
            let (w, h) = self.dct_select_size();
            h >= w
        }
    }
}
