//! This crate provides types related to JPEG XL frames.
//!
//! A JPEG XL image contains one or more frames. A frame represents single unit of image that can
//! be displayed or referenced by other frames.
//!
//! A frame consists of a few components:
//! - [Frame header][FrameHeader].
//! - [Table of contents (TOC)][data::Toc].
//! - Actual frame data, in the following order, potentially permuted as specified in the TOC:
//!   - one [`LfGlobal`][data::LfGlobal],
//!   - [`num_lf_groups`] [`LfGroup`][data::LfGroup]'s, in raster order,
//!   - one [`HfGlobal`][data::HfGlobal], potentially empty for Modular frames, and
//!   - [`num_passes`] times [`num_groups`] [pass groups][data::decode_pass_group], in raster
//!     order.
//!
//! [`num_lf_groups`]: FrameHeader::num_lf_groups
//! [`num_groups`]: FrameHeader::num_groups
//! [`num_passes`]: header::Passes::num_passes
use std::collections::BTreeMap;
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
    data: Vec<GroupData>,
    reading_data_index: usize,
    pass_shifts: BTreeMap<u32, (i32, i32)>,
}

#[derive(Debug)]
struct GroupData {
    toc_group: TocGroup,
    bytes: Vec<u8>,
}

impl From<TocGroup> for GroupData {
    fn from(value: TocGroup) -> Self {
        let cap = value.size as usize;
        Self {
            toc_group: value,
            bytes: Vec::with_capacity(cap),
        }
    }
}

impl Bundle<Arc<ImageHeader>> for Frame {
    type Error = crate::Error;

    fn parse(bitstream: &mut Bitstream, image_header: Arc<ImageHeader>) -> Result<Self> {
        bitstream.zero_pad_to_byte()?;
        let base_offset = bitstream.num_read_bits() / 8;
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

        let mut toc = read_bits!(bitstream, Bundle(Toc), &header)?;
        toc.adjust_offsets(base_offset);
        let data = toc.iter_bitstream_order().map(GroupData::from).collect();

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
            data,
            reading_data_index: 0,
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
        self.data.get(idx).map(|b| &*b.bytes)
    }
}

impl Frame {
    pub fn feed_bytes<'buf>(&mut self, mut buf: &'buf [u8]) -> &'buf [u8] {
        while let Some(group_data) = self.data.get_mut(self.reading_data_index) {
            let bytes_left = group_data.toc_group.size as usize - group_data.bytes.len();
            if buf.len() < bytes_left {
                group_data.bytes.extend_from_slice(buf);
                return &[];
            }
            let (l, r) = buf.split_at(bytes_left);
            group_data.bytes.extend_from_slice(l);
            buf = r;
            self.reading_data_index += 1;
        }
        buf
    }

    #[inline]
    pub fn is_loading_done(&self) -> bool {
        self.reading_data_index >= self.data.len()
    }
}

struct AllParseResult<'buf> {
    #[allow(unused)]
    lf_global: LfGlobal,
    lf_group: LfGroup,
    hf_global: Option<HfGlobal>,
    pass_group_bitstream: Bitstream<'buf>,
}

impl Frame {
    fn try_parse_all(&self) -> Option<Result<AllParseResult>> {
        if !self.toc.is_single_entry() {
            panic!();
        }

        let group = self.data.get(0)?;
        let mut bitstream = Bitstream::new(&group.bytes);
        let result = (|| -> Result<_> {
            let lf_global = LfGlobal::parse(&mut bitstream, LfGlobalParams::new(&self.image_header, &self.header, false))?;
            let lf_group = LfGroup::parse(&mut bitstream, LfGroupParams::new(&self.header, &lf_global, 0, false))?;
            let hf_global = (self.header.encoding == header::Encoding::VarDct).then(|| {
                HfGlobal::parse(&mut bitstream, HfGlobalParams::new(&self.image_header.metadata, &self.header, &lf_global))
            }).transpose()?;
            Ok((lf_global, lf_group, hf_global))
        })();

        match result {
            Ok((lf_global, lf_group, hf_global)) => Some(Ok(AllParseResult {
                lf_global,
                lf_group,
                hf_global,
                pass_group_bitstream: bitstream,
            })),
            Err(e) => Some(Err(e)),
        }
    }

    pub fn try_parse_lf_global(&self) -> Option<Result<LfGlobal>> {
        Some(if self.toc.is_single_entry() {
            let group = self.data.get(0)?;
            let mut bitstream = Bitstream::new(&group.bytes);
            LfGlobal::parse(&mut bitstream, LfGlobalParams::new(&self.image_header, &self.header, false))
        } else {
            let idx = self.toc.group_index_bitstream_order(TocGroupKind::LfGlobal);
            let group = self.data.get(idx)?;
            let allow_partial = group.bytes.len() < group.toc_group.size as usize;

            let mut bitstream = Bitstream::new(&group.bytes);
            LfGlobal::parse(&mut bitstream, LfGlobalParams::new(&self.image_header, &self.header, allow_partial))
        })
    }

    pub fn try_parse_lf_group(&self, cached_lf_global: Option<&LfGlobal>, lf_group_idx: u32) -> Option<Result<LfGroup>> {
        if self.toc.is_single_entry() {
            if lf_group_idx != 0 {
                return None;
            }
            Some(self.try_parse_all()?.map(|x| x.lf_group))
        } else {
            let idx = self.toc.group_index_bitstream_order(TocGroupKind::LfGroup(lf_group_idx));
            let group = self.data.get(idx)?;
            let allow_partial = group.bytes.len() < group.toc_group.size as usize;

            let mut bitstream = Bitstream::new(&group.bytes);
            let lf_global = if cached_lf_global.is_none() {
                match self.try_parse_lf_global()? {
                    Ok(lf_global) => Some(lf_global),
                    Err(e) => return Some(Err(e)),
                }
            } else {
                None
            };
            let lf_global = cached_lf_global.or(lf_global.as_ref()).unwrap();
            let result = LfGroup::parse(&mut bitstream, LfGroupParams::new(&self.header, lf_global, lf_group_idx, allow_partial));
            if allow_partial && result.is_err() {
                return None;
            }
            Some(result)
        }
    }

    pub fn try_parse_hf_global(&self, cached_lf_global: Option<&LfGlobal>) -> Option<Result<HfGlobal>> {
        if self.header.encoding == header::Encoding::Modular {
            return None;
        }

        if self.toc.is_single_entry() {
            Some(self.try_parse_all()?.map(|x| x.hf_global.unwrap()))
        } else {
            let idx = self.toc.group_index_bitstream_order(TocGroupKind::HfGlobal);
            let group = self.data.get(idx)?;
            if group.bytes.len() < group.toc_group.size as usize {
                return None;
            }

            let mut bitstream = Bitstream::new(&group.bytes);
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

    pub fn pass_group_bitstream(&self, pass_idx: u32, group_idx: u32) -> Option<Result<PassGroupBitstream>> {
        Some(if self.toc.is_single_entry() {
            self.try_parse_all()?.map(|group| PassGroupBitstream {
                bitstream: group.pass_group_bitstream,
                partial: self.data[0].bytes.len() < self.data[0].toc_group.size as usize,
            })
        } else {
            let idx = self.toc.group_index_bitstream_order(TocGroupKind::GroupPass { pass_idx, group_idx });
            let group = self.data.get(idx)?;
            let partial = group.bytes.len() < group.toc_group.size as usize;

            Ok(PassGroupBitstream {
                bitstream: Bitstream::new(&group.bytes),
                partial,
            })
        })
    }
}

#[derive(Debug)]
pub struct PassGroupBitstream<'buf> {
    pub bitstream: Bitstream<'buf>,
    pub partial: bool,
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
