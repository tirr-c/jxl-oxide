#![allow(unused_variables, unused_mut, dead_code)]

use std::io::Read;

use jxl_bitstream::{Bitstream, Bundle};
use jxl_image::{Headers, ExtraChannelType};

use crate::Result;

#[derive(Debug)]
pub struct Patches {
    patches: Vec<PatchRef>,
}

#[derive(Debug)]
struct PatchRef {
    ref_idx: u32,
    x0: u32,
    y0: u32,
    width: u32,
    height: u32,
    patch_targets: Vec<PatchTarget>,
}

#[derive(Debug)]
struct PatchTarget {
    x: u32,
    y: u32,
    blending: Vec<BlendingModeInformation>,
}

#[derive(Debug)]
pub struct BlendingModeInformation {
    pub mode: PatchBlendMode,
    pub alpha_channel: u32,
    pub clamp: bool,
}

#[derive(Debug, PartialEq, Eq)]
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

impl Bundle<&Headers> for Patches {
    type Error = crate::Error;

    fn parse<R: Read>(bitstream: &mut Bitstream<R>, image_header: &Headers) -> Result<Self> {
        let num_extra = image_header.metadata.num_extra as usize;
        let alpha_channel_indices = image_header.metadata.ec_info.iter()
            .enumerate()
            .filter_map(|(idx, info)| (info.ty == ExtraChannelType::Alpha).then_some(idx as u32))
            .collect::<Vec<_>>();

        let mut decoder = jxl_coding::Decoder::parse(bitstream, 10)?;
        decoder.begin(bitstream)?;

        let num_patches = decoder.read_varint(bitstream, 0)?;
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
                    (dx + px, dy + py)
                } else {
                    (decoder.read_varint(bitstream, 4)?, decoder.read_varint(bitstream, 4)?)
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
