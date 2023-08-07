//! This crate provides types related to JPEG XL frames.
//!
//! A JPEG XL image contains one or more frames. A frame represents single unit of image that can
//! be displayed or referenced by other frames.
//!
//! A frame consists of a few components:
//! - [Frame header][FrameHeader].
//! - [Table of contents (TOC)][data::Toc].
//! - [Actual frame data][FrameData], in the following order, potentially permuted as specified in
//!   the TOC:
//!   - one [`LfGlobal`][data::LfGlobal],
//!   - [`num_lf_groups`] [`LfGroup`][data::LfGroup]'s, in raster order,
//!   - one [`HfGlobal`][data::HfGlobal], potentially empty for Modular frames, and
//!   - [`num_passes`] times [`num_groups`] [`PassGroup`][data::PassGroup]'s, in raster order.
//!
//! [`num_lf_groups`]: FrameHeader::num_lf_groups
//! [`num_groups`]: FrameHeader::num_groups
//! [`num_passes`]: header::Passes::num_passes
use std::{collections::BTreeMap, io::Cursor};
use std::io::Read;
use std::sync::Arc;

use jxl_bitstream::{read_bits, Bitstream, Bundle};
use jxl_image::ImageHeader;

mod error;
pub mod filter;
pub mod data;
pub mod header;

pub use error::{Error, Result};
pub use header::FrameHeader;

use crate::data::*;

/// JPEG XL frame.
///
/// A frame represents a single unit of image that can be displayed or referenced by other frames.
#[derive(Debug)]
pub struct Frame {
    image_header: Arc<ImageHeader>,
    header: FrameHeader,
    toc: Toc,
    data: Vec<Vec<u8>>,
    pass_shifts: BTreeMap<u32, (i32, i32)>,
}

impl Bundle<Arc<ImageHeader>> for Frame {
    type Error = crate::Error;

    fn parse<R: Read>(bitstream: &mut Bitstream<R>, image_header: Arc<ImageHeader>) -> Result<Self> {
        bitstream.zero_pad_to_byte()?;
        let header = read_bits!(bitstream, Bundle(FrameHeader), &image_header)?;

        for blending_info in std::iter::once(&header.blending_info).chain(&header.ec_blending_info) {
            if blending_info.mode.use_alpha()
                && blending_info.alpha_channel as usize >= image_header.metadata.ec_info.len()
            {
                return Err(jxl_bitstream::Error::ValidationFailed(
                    "blending_info.alpha_channel out of range",
                ).into());
            }
        }
        if header.flags.use_lf_frame() && header.lf_level >= 4 {
            return Err(jxl_bitstream::Error::ValidationFailed("lf_level out of range").into());
        }

        for ec_info in &image_header.metadata.ec_info {
            if ec_info.dim_shift > 7 + header.group_size_shift {
                return Err(jxl_bitstream::Error::ValidationFailed(
                    "dim_shift too large"
                ).into());
            }
        }

        if header.upsampling > 1 {
            for (ec_upsampling, ec_info) in header
                .ec_upsampling
                .iter()
                .zip(image_header.metadata.ec_info.iter())
            {
                if (ec_upsampling << ec_info.dim_shift) < header.upsampling {
                    return Err(jxl_bitstream::Error::ValidationFailed(
                        "EC upsampling < color upsampling, which is invalid"
                    ).into());
                }
            }
        }

        if header.width == 0 || header.height == 0 {
            return Err(jxl_bitstream::Error::ValidationFailed(
                "Invalid crop dimensions for frame: zero width or height"
            ).into());
        }

        let toc = read_bits!(bitstream, Bundle(Toc), &header)?;

        let passes = &header.passes;
        let mut pass_shifts = BTreeMap::new();
        let mut maxshift = 3i32;
        for (&downsample, &last_pass) in passes.downsample.iter().zip(&passes.last_pass) {
            let minshift = downsample.trailing_zeros() as i32;
            pass_shifts.insert(last_pass, (minshift, maxshift));
            maxshift = minshift;
        }
        pass_shifts.insert(header.passes.num_passes - 1, (0i32, maxshift));

        Ok(Self {
            image_header,
            header,
            toc,
            data: Vec::new(),
            pass_shifts,
        })
    }
}

impl Frame {
    pub fn image_header(&self) -> &ImageHeader {
        &self.image_header
    }

    pub fn clone_image_header(&self) -> Arc<ImageHeader> {
        Arc::clone(&self.image_header)
    }

    /// Returns the frame header.
    pub fn header(&self) -> &FrameHeader {
        &self.header
    }

    /// Returns the TOC.
    ///
    /// See the documentation of [`Toc`] for details.
    pub fn toc(&self) -> &Toc {
        &self.toc
    }

    pub fn pass_shifts(&self, pass_idx: u32) -> Option<(i32, i32)> {
        self.pass_shifts.get(&pass_idx).copied()
    }

    pub fn data(&self, group: TocGroupKind) -> Option<&[u8]> {
        let idx = self.toc.group_index_bitstream_order(group);
        self.data.get(idx).map(|b| &**b)
    }
}

impl Frame {
    pub fn read_all<R: Read>(&mut self, bitstream: &mut Bitstream<R>) -> Result<()> {
        assert!(self.data.is_empty());

        for group in self.toc.iter_bitstream_order() {
            tracing::trace!(?group);
            bitstream.zero_pad_to_byte()?;

            let mut data = vec![0u8; group.size as usize];
            bitstream.read_bytes_aligned(&mut data)?;

            self.data.push(data);
        }

        Ok(())
    }
}

impl Frame {
    fn try_parse_all(&self) -> Option<Result<(LfGlobal, LfGroup, Option<HfGlobal>, Bitstream<Cursor<&[u8]>>)>> {
        if !self.toc.is_single_entry() {
            panic!();
        }

        let group = self.data.get(0)?;
        let mut bitstream = Bitstream::new(Cursor::new(&**group));
        let result = (|| -> Result<_> {
            let lf_global = LfGlobal::parse(&mut bitstream, (&self.image_header, &self.header))?;
            let lf_group = LfGroup::parse(&mut bitstream, LfGroupParams::new(&self.header, &lf_global, 0))?;
            let hf_global = (self.header.encoding == header::Encoding::VarDct).then(|| {
                HfGlobal::parse(&mut bitstream, HfGlobalParams::new(&self.image_header.metadata, &self.header, &lf_global))
            }).transpose()?;
            Ok((lf_global, lf_group, hf_global))
        })();

        match result {
            Ok((lf_global, lf_group, hf_global)) => Some(Ok((lf_global, lf_group, hf_global, bitstream))),
            Err(e) => Some(Err(e)),
        }
    }

    pub fn try_parse_lf_global(&self) -> Option<Result<LfGlobal>> {
        Some(if self.toc.is_single_entry() {
            let group = self.data.get(0)?;
            let mut bitstream = Bitstream::new(Cursor::new(group));
            LfGlobal::parse(&mut bitstream, (&self.image_header, &self.header))
        } else {
            let idx = self.toc.group_index_bitstream_order(TocGroupKind::LfGlobal);
            let group = self.data.get(idx)?;
            let mut bitstream = Bitstream::new(Cursor::new(group));
            LfGlobal::parse(&mut bitstream, (&self.image_header, &self.header))
        })
    }

    pub fn try_parse_lf_group(&self, cached_lf_global: Option<&LfGlobal>, lf_group_idx: u32) -> Option<Result<LfGroup>> {
        if self.toc.is_single_entry() {
            if lf_group_idx != 0 {
                return None;
            }
            Some(self.try_parse_all()?.map(|(_, x, _, _)| x))
        } else {
            let idx = self.toc.group_index_bitstream_order(TocGroupKind::LfGroup(lf_group_idx));
            let group = self.data.get(idx)?;
            let mut bitstream = Bitstream::new(Cursor::new(group));
            let lf_global = if cached_lf_global.is_none() {
                match self.try_parse_lf_global()? {
                    Ok(lf_global) => Some(lf_global),
                    Err(e) => return Some(Err(e)),
                }
            } else {
                None
            };
            let lf_global = cached_lf_global.or(lf_global.as_ref()).unwrap();
            Some(LfGroup::parse(&mut bitstream, LfGroupParams::new(&self.header, lf_global, lf_group_idx)))
        }
    }

    pub fn try_parse_hf_global(&self, cached_lf_global: Option<&LfGlobal>) -> Option<Result<HfGlobal>> {
        if self.header.encoding == header::Encoding::Modular {
            return None;
        }

        if self.toc.is_single_entry() {
            Some(self.try_parse_all()?.map(|(_, _, x, _)| x.unwrap()))
        } else {
            let idx = self.toc.group_index_bitstream_order(TocGroupKind::HfGlobal);
            let group = self.data.get(idx)?;
            let mut bitstream = Bitstream::new(Cursor::new(group));
            let lf_global = if cached_lf_global.is_none() {
                match self.try_parse_lf_global()? {
                    Ok(lf_global) => Some(lf_global),
                    Err(e) => return Some(Err(e)),
                }
            } else {
                None
            };
            let lf_global = cached_lf_global.or(lf_global.as_ref()).unwrap();
            let params = HfGlobalParams::new(&self.image_header.metadata, &self.header, lf_global);
            Some(HfGlobal::parse(&mut bitstream, params))
        }
    }

    pub fn pass_group_bitstream(&self, pass_idx: u32, group_idx: u32) -> Option<Result<Bitstream<Cursor<&[u8]>>>> {
        if self.toc.is_single_entry() {
            Some(self.try_parse_all()?.map(|(_, _, _, x)| x))
        } else {
            let idx = self.toc.group_index_bitstream_order(TocGroupKind::GroupPass { pass_idx, group_idx });
            let group = self.data.get(idx)?;
            Some(Ok(Bitstream::new(Cursor::new(&**group))))
        }
    }
}

impl Frame {
    /// Adjusts the cropping region of the image to the actual decoding region of the frame.
    ///
    /// The cropping region of the *image* needs to be adjusted to be used in a *frame*, for a few
    /// reasons:
    /// - A frame may be blended to the canvas with offset, which makes the image and the frame
    ///   have different coordinates.
    /// - Some filters reference other samples, which requires padding to the region.
    ///
    /// This method takes care of those and adjusts the given region appropriately.
    pub fn adjust_region(&self, (left, top, width, height): &mut (u32, u32, u32, u32)) {
        if self.header.have_crop {
            *left = left.saturating_add_signed(-self.header.x0);
            *top = top.saturating_add_signed(-self.header.y0);
        };

        let mut padding = 0u32;
        if self.header.restoration_filter.gab.enabled() {
            tracing::debug!("Gabor-like filter requires padding of 1 pixel");
            padding = 1;
        }
        if self.header.restoration_filter.epf.enabled() {
            tracing::debug!("Edge-preserving filter requires padding of 3 pixels");
            padding = 3;
        }
        if padding > 0 {
            let delta_w = (*left).min(padding);
            let delta_h = (*top).min(padding);
            *left -= delta_w;
            *top -= delta_h;
            *width += delta_w + padding;
            *height += delta_h + padding;
        }
    }
}
