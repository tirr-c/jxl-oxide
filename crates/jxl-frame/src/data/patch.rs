use jxl_bitstream::{Bitstream, Bundle, unpack_signed};
use jxl_image::ImageHeader;

use crate::{Result, FrameHeader};

#[derive(Debug)]
pub struct Patches {
    pub patches: Vec<PatchRef>,
}

#[derive(Debug)]
pub struct PatchRef {
    pub ref_idx: u32,
    pub x0: u32,
    pub y0: u32,
    pub width: u32,
    pub height: u32,
    pub patch_targets: Vec<PatchTarget>,
}

#[derive(Debug)]
pub struct PatchTarget {
    pub x: i32,
    pub y: i32,
    pub blending: Vec<BlendingModeInformation>,
}

#[derive(Debug)]
pub struct BlendingModeInformation {
    pub mode: PatchBlendMode,
    pub alpha_channel: u32,
    pub clamp: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PatchBlendMode {
    None = 0,
    Replace,
    Add,
    Mul,
    BlendAbove,
    BlendBelow,
    MulAddAbove,
    MulAddBelow,
}

impl TryFrom<u32> for PatchBlendMode {
    type Error = jxl_bitstream::Error;

    fn try_from(value: u32) -> std::result::Result<Self, Self::Error> {
        use PatchBlendMode::*;

        Ok(match value {
            0 => PatchBlendMode::None,
            1 => Replace,
            2 => Add,
            3 => Mul,
            4 => BlendAbove,
            5 => BlendBelow,
            6 => MulAddAbove,
            7 => MulAddBelow,
            _ => return Err(jxl_bitstream::Error::InvalidEnum { name: "PatchBlendMode", value }),
        })
    }
}

impl PatchBlendMode {
    #[inline]
    pub fn use_alpha(self) -> bool {
        matches!(self, Self::BlendAbove | Self::BlendBelow | Self::MulAddAbove | Self::MulAddBelow)
    }
}

impl Bundle<(&ImageHeader, &FrameHeader)> for Patches {
    type Error = crate::Error;

    fn parse(
        bitstream: &mut Bitstream,
        (image_header, frame_header): (&ImageHeader, &FrameHeader),
    ) -> Result<Self> {
        let num_extra = image_header.metadata.ec_info.len();
        let alpha_channel_indices = image_header.metadata.ec_info.iter()
            .enumerate()
            .filter_map(|(idx, info)| info.is_alpha().then_some(idx as u32))
            .collect::<Vec<_>>();

        let mut decoder = jxl_coding::Decoder::parse(bitstream, 10)?;
        decoder.begin(bitstream)?;

        let num_patches = decoder.read_varint(bitstream, 0)?;
        let max_num_patches = (1 << 24).min((frame_header.width as u64 * frame_header.height as u64 / 16) as u32);
        if num_patches > max_num_patches {
            tracing::error!(num_patches, max_num_patches, "Too many patches");
            return Err(jxl_bitstream::Error::ProfileConformance(
                "too many patches"
            ).into());
        }

        let patches = std::iter::repeat_with(|| -> Result<_> {
            let ref_idx = decoder.read_varint(bitstream, 1)?;
            let x0 = decoder.read_varint(bitstream, 3)?;
            let y0 = decoder.read_varint(bitstream, 3)?;
            let width = decoder.read_varint(bitstream, 2)? + 1;
            let height = decoder.read_varint(bitstream, 2)? + 1;
            let count = decoder.read_varint(bitstream, 7)? + 1;

            let mut prev_xy = None;
            let patch_targets = std::iter::repeat_with(|| -> Result<_> {
                let (x, y) = if let Some((px, py)) = prev_xy {
                    let dx = decoder.read_varint(bitstream, 6)?;
                    let dy = decoder.read_varint(bitstream, 6)?;
                    let dx = unpack_signed(dx);
                    let dy = unpack_signed(dy);
                    (dx + px, dy + py)
                } else {
                    (decoder.read_varint(bitstream, 4)? as i32, decoder.read_varint(bitstream, 4)? as i32)
                };
                prev_xy = Some((x, y));

                let blending = std::iter::repeat_with(|| -> Result<_> {
                    let raw_mode = decoder.read_varint(bitstream, 5)?;
                    let mode = PatchBlendMode::try_from(raw_mode)?;
                    let alpha_channel = if raw_mode >= 4 && alpha_channel_indices.len() >= 2 {
                        decoder.read_varint(bitstream, 8)?
                    } else {
                        alpha_channel_indices.first().copied().unwrap_or_default()
                    };
                    let clamp = if raw_mode >= 3 {
                        decoder.read_varint(bitstream, 9)? != 0
                    } else {
                        false
                    };

                    Ok(BlendingModeInformation { mode, alpha_channel, clamp })
                }).take(num_extra + 1).collect::<Result<Vec<_>>>()?;

                Ok(PatchTarget { x, y, blending })
            }).take(count as usize).collect::<Result<Vec<_>>>()?;

            Ok(PatchRef {
                ref_idx,
                x0,
                y0,
                width,
                height,
                patch_targets,
            })
        }).take(num_patches as usize).collect::<Result<Vec<_>>>()?;

        decoder.finalize()?;
        Ok(Self { patches })
    }
}
