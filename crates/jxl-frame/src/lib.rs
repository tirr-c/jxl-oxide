//! This crate provides types related to JPEG XL frames.
//!
//! A JPEG XL image contains one or more frames. A frame represents single unit of image that can
//! be displayed or referenced by other frames.
//!
//! A frame consists of a few components:
//! - [Frame header][FrameHeader].
//! - [Table of contents (TOC)][data::Toc].
//! - Actual frame data, in the following order, potentially permuted as specified in the TOC:
//!   - one [`LfGlobal`],
//!   - [`num_lf_groups`] [`LfGroup`]'s, in raster order,
//!   - one [`HfGlobal`], potentially empty for Modular frames, and
//!   - [`num_passes`] times [`num_groups`] [pass groups][data::decode_pass_group], in raster
//!     order.
//!
//! [`num_lf_groups`]: FrameHeader::num_lf_groups
//! [`num_groups`]: FrameHeader::num_groups
//! [`num_passes`]: header::Passes::num_passes
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use jxl_bitstream::{read_bits, Bitstream, Bundle, Lz77Mode};
use jxl_grid::AllocTracker;
use jxl_image::ImageHeader;

pub mod data;
mod error;
pub mod filter;
pub mod header;

pub use error::{Error, Result};
pub use header::FrameHeader;
use jxl_modular::Sample;
use jxl_modular::{image::TransformedModularSubimage, MaConfig};
use jxl_threadpool::JxlThreadPool;

use crate::data::*;

/// JPEG XL frame.
///
/// A frame represents a single unit of image that can be displayed or referenced by other frames.
#[derive(Debug)]
pub struct Frame {
    pool: JxlThreadPool,
    tracker: Option<AllocTracker>,
    image_header: Arc<ImageHeader>,
    header: FrameHeader,
    toc: Toc,
    data: Vec<GroupData>,
    all_group_offsets: AllGroupOffsets,
    reading_data_index: usize,
    pass_shifts: BTreeMap<u32, (i32, i32)>,
    lz77_mode: Lz77Mode,
}

#[derive(Debug, Default)]
struct AllGroupOffsets {
    lf_group: AtomicUsize,
    hf_global: AtomicUsize,
    pass_group: AtomicUsize,
    has_error: AtomicUsize,
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

#[derive(Debug, Clone)]
pub struct FrameContext<'a> {
    pub image_header: Arc<ImageHeader>,
    pub tracker: Option<&'a AllocTracker>,
    pub pool: JxlThreadPool,
}

impl Bundle<FrameContext<'_>> for Frame {
    type Error = crate::Error;

    fn parse(bitstream: &mut Bitstream, ctx: FrameContext) -> Result<Self> {
        let FrameContext {
            image_header,
            tracker,
            pool,
        } = ctx;
        let tracker = tracker.cloned();

        bitstream.zero_pad_to_byte()?;
        let base_offset = bitstream.num_read_bits() / 8;
        let header = read_bits!(bitstream, Bundle(FrameHeader), &image_header)?;

        let width = header.width as u64;
        let height = header.height as u64;
        if width > (1 << 30) {
            tracing::error!(width, "Frame width too large; limit is 2^30");
            return Err(jxl_bitstream::Error::ProfileConformance("frame width too large").into());
        }
        if height > (1 << 30) {
            tracing::error!(width, "Frame height too large; limit is 2^30");
            return Err(jxl_bitstream::Error::ProfileConformance("frame height too large").into());
        }
        if (width * height) > (1 << 40) {
            tracing::error!(
                area = width * height,
                "Frame area (width * height) too large; limit is 2^40"
            );
            return Err(jxl_bitstream::Error::ProfileConformance("frame area too large").into());
        }

        for blending_info in std::iter::once(&header.blending_info).chain(&header.ec_blending_info)
        {
            if blending_info.mode.use_alpha() {
                let alpha_idx = blending_info.alpha_channel as usize;
                let Some(alpha_ec_info) = image_header.metadata.ec_info.get(alpha_idx) else {
                    tracing::error!(?blending_info, "blending_info.alpha_channel out of range");
                    return Err(jxl_bitstream::Error::ValidationFailed(
                        "blending_info.alpha_channel out of range",
                    )
                    .into());
                };
                if !alpha_ec_info.is_alpha() {
                    tracing::error!(
                        ?blending_info,
                        ?alpha_ec_info,
                        "blending_info.alpha_channel is not the type of Alpha",
                    );
                    return Err(jxl_bitstream::Error::ValidationFailed(
                        "blending_info.alpha_channel is not the type of Alpha",
                    )
                    .into());
                }
            }
        }

        if header.flags.use_lf_frame() && header.lf_level >= 4 {
            return Err(jxl_bitstream::Error::ValidationFailed("lf_level out of range").into());
        }

        let color_upsampling_shift = header.upsampling.trailing_zeros();
        for (ec_upsampling, ec_info) in header
            .ec_upsampling
            .iter()
            .zip(image_header.metadata.ec_info.iter())
        {
            let ec_upsampling_shift = ec_upsampling.trailing_zeros();
            let dim_shift = ec_info.dim_shift;

            if ec_upsampling_shift + dim_shift < color_upsampling_shift {
                return Err(jxl_bitstream::Error::ValidationFailed(
                    "EC upsampling < color upsampling, which is invalid",
                )
                .into());
            }

            if ec_upsampling_shift + dim_shift > 6 {
                tracing::error!(
                    ec_upsampling,
                    dim_shift = ec_info.dim_shift,
                    "Cumulative EC upsampling factor is too large"
                );
                return Err(jxl_bitstream::Error::ValidationFailed(
                    "cumulative EC upsampling factor is too large",
                )
                .into());
            }

            let actual_dim_shift = ec_upsampling_shift + dim_shift - color_upsampling_shift;

            if actual_dim_shift > 7 + header.group_size_shift {
                return Err(jxl_bitstream::Error::ValidationFailed("dim_shift too large").into());
            }
        }

        if header.width == 0 || header.height == 0 {
            return Err(jxl_bitstream::Error::ValidationFailed(
                "Invalid crop dimensions for frame: zero width or height",
            )
            .into());
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
            pool,
            tracker,
            image_header,
            header,
            toc,
            data,
            all_group_offsets: AllGroupOffsets::default(),
            reading_data_index: 0,
            pass_shifts,
            lz77_mode: bitstream.lz77_mode(),
        })
    }
}

impl Frame {
    #[inline]
    pub fn alloc_tracker(&self) -> Option<&AllocTracker> {
        self.tracker.as_ref()
    }

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

    pub fn pass_shifts(&self) -> &BTreeMap<u32, (i32, i32)> {
        &self.pass_shifts
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
    pub fn current_loading_group(&self) -> Option<TocGroup> {
        self.toc.iter_bitstream_order().nth(self.reading_data_index)
    }

    #[inline]
    pub fn is_loading_done(&self) -> bool {
        self.reading_data_index >= self.data.len()
    }
}

impl Frame {
    pub fn try_parse_lf_global<S: Sample>(&self) -> Option<Result<LfGlobal<S>>> {
        Some(if self.toc.is_single_entry() {
            if self.all_group_offsets.has_error.load(Ordering::Relaxed) != 0 {
                return Some(Err(Error::HadError));
            }

            let group = self.data.first()?;
            let loaded = self.reading_data_index != 0;
            let mut bitstream = Bitstream::new(&group.bytes);
            bitstream.set_lz77_mode(self.lz77_mode);
            let lf_global = LfGlobal::parse(
                &mut bitstream,
                LfGlobalParams::new(
                    &self.image_header,
                    &self.header,
                    self.tracker.as_ref(),
                    false,
                ),
            );
            match lf_global {
                Ok(lf_global) => {
                    tracing::trace!(num_read_bits = bitstream.num_read_bits(), "LfGlobal");
                    self.all_group_offsets
                        .lf_group
                        .store(bitstream.num_read_bits(), Ordering::Relaxed);
                    Ok(lf_global)
                }
                Err(e) if !loaded && e.unexpected_eof() => Err(e),
                Err(e) => {
                    self.all_group_offsets.has_error.store(1, Ordering::Relaxed);
                    Err(e)
                }
            }
        } else {
            let idx = self.toc.group_index_bitstream_order(TocGroupKind::LfGlobal);
            let group = self.data.get(idx)?;
            let allow_partial = group.bytes.len() < group.toc_group.size as usize;

            let mut bitstream = Bitstream::new(&group.bytes);
            bitstream.set_lz77_mode(self.lz77_mode);
            LfGlobal::parse(
                &mut bitstream,
                LfGlobalParams::new(
                    &self.image_header,
                    &self.header,
                    self.tracker.as_ref(),
                    allow_partial,
                ),
            )
        })
    }

    pub fn try_parse_lf_group<S: Sample>(
        &self,
        lf_global_vardct: Option<&LfGlobalVarDct>,
        global_ma_config: Option<&MaConfig>,
        mlf_group: Option<TransformedModularSubimage<S>>,
        lf_group_idx: u32,
    ) -> Option<Result<LfGroup<S>>> {
        if self.toc.is_single_entry() {
            if self.all_group_offsets.has_error.load(Ordering::Relaxed) != 0 {
                return Some(Err(Error::HadError));
            }

            if lf_group_idx != 0 {
                return None;
            }

            let group = self.data.first()?;
            let loaded = self.reading_data_index != 0;
            let mut bitstream = Bitstream::new(&group.bytes);
            bitstream.set_lz77_mode(self.lz77_mode);
            let offset = self.all_group_offsets.lf_group.load(Ordering::Relaxed);
            if offset == 0 {
                let lf_global = self.try_parse_lf_global::<S>().unwrap();
                if let Err(e) = lf_global {
                    return Some(Err(e));
                }
            }
            let offset = self.all_group_offsets.lf_group.load(Ordering::Relaxed);
            bitstream.skip_bits(offset).unwrap();

            let result = LfGroup::parse(
                &mut bitstream,
                LfGroupParams {
                    frame_header: &self.header,
                    quantizer: lf_global_vardct.map(|x| &x.quantizer),
                    global_ma_config,
                    mlf_group,
                    lf_group_idx,
                    allow_partial: !loaded,
                    tracker: self.tracker.as_ref(),
                    pool: &self.pool,
                },
            );

            match result {
                Ok(result) => {
                    tracing::trace!(num_read_bits = bitstream.num_read_bits(), "LfGroup");
                    self.all_group_offsets
                        .hf_global
                        .store(bitstream.num_read_bits(), Ordering::Relaxed);
                    Some(Ok(result))
                }
                Err(e) if !loaded && e.unexpected_eof() => None,
                Err(e) => {
                    self.all_group_offsets.has_error.store(2, Ordering::Relaxed);
                    Some(Err(e))
                }
            }
        } else {
            let idx = self
                .toc
                .group_index_bitstream_order(TocGroupKind::LfGroup(lf_group_idx));
            let group = self.data.get(idx)?;
            let allow_partial = group.bytes.len() < group.toc_group.size as usize;

            let mut bitstream = Bitstream::new(&group.bytes);
            bitstream.set_lz77_mode(self.lz77_mode);
            let result = LfGroup::parse(
                &mut bitstream,
                LfGroupParams {
                    frame_header: &self.header,
                    quantizer: lf_global_vardct.map(|x| &x.quantizer),
                    global_ma_config,
                    mlf_group,
                    lf_group_idx,
                    allow_partial,
                    tracker: self.tracker.as_ref(),
                    pool: &self.pool,
                },
            );
            if allow_partial && result.is_err() {
                return None;
            }
            Some(result)
        }
    }

    pub fn try_parse_hf_global<S: Sample>(
        &self,
        cached_lf_global: Option<&LfGlobal<S>>,
    ) -> Option<Result<HfGlobal>> {
        let is_modular = self.header.encoding == header::Encoding::Modular;

        if self.toc.is_single_entry() {
            if self.all_group_offsets.has_error.load(Ordering::Relaxed) != 0 {
                return Some(Err(Error::HadError));
            }

            let group = self.data.first()?;
            let loaded = self.reading_data_index != 0;
            let mut bitstream = Bitstream::new(&group.bytes);
            bitstream.set_lz77_mode(self.lz77_mode);
            let offset = self.all_group_offsets.hf_global.load(Ordering::Relaxed);
            let lf_global = if cached_lf_global.is_none() && (offset == 0 || !is_modular) {
                match self.try_parse_lf_global()? {
                    Ok(lf_global) => Some(lf_global),
                    Err(e) => return Some(Err(e)),
                }
            } else {
                None
            };
            let lf_global = cached_lf_global.or(lf_global.as_ref());

            if offset == 0 {
                let lf_global = lf_global.unwrap();
                let mut gmodular = match lf_global.gmodular.try_clone() {
                    Ok(gmodular) => gmodular,
                    Err(e) => return Some(Err(e)),
                };
                let groups = gmodular
                    .modular
                    .image_mut()
                    .map(|x| x.prepare_groups(&self.pass_shifts))
                    .transpose();
                let groups = match groups {
                    Ok(groups) => groups,
                    Err(e) => return Some(Err(e.into())),
                };
                let mlf_group = groups.and_then(|mut x| x.lf_groups.pop());
                let lf_group = self
                    .try_parse_lf_group(
                        lf_global.vardct.as_ref(),
                        lf_global.gmodular.ma_config(),
                        mlf_group,
                        0,
                    )
                    .ok_or(
                        jxl_bitstream::Error::Io(std::io::ErrorKind::UnexpectedEof.into()).into(),
                    )
                    .and_then(|x| x);
                if let Err(e) = lf_group {
                    return Some(Err(e));
                }
            }
            let offset = self.all_group_offsets.hf_global.load(Ordering::Relaxed);

            if self.header.encoding == header::Encoding::Modular {
                self.all_group_offsets
                    .pass_group
                    .store(offset, Ordering::Relaxed);
                return None;
            }

            bitstream.skip_bits(offset).unwrap();
            let lf_global = lf_global.unwrap();
            let result = HfGlobal::parse(
                &mut bitstream,
                HfGlobalParams::new(
                    &self.image_header.metadata,
                    &self.header,
                    lf_global,
                    self.tracker.as_ref(),
                    &self.pool,
                ),
            );

            Some(match result {
                Ok(result) => {
                    self.all_group_offsets
                        .pass_group
                        .store(bitstream.num_read_bits(), Ordering::Relaxed);
                    Ok(result)
                }
                Err(e) if !loaded && e.unexpected_eof() => Err(e),
                Err(e) => {
                    self.all_group_offsets.has_error.store(3, Ordering::Relaxed);
                    Err(e)
                }
            })
        } else {
            if self.header.encoding == header::Encoding::Modular {
                return None;
            }

            let idx = self.toc.group_index_bitstream_order(TocGroupKind::HfGlobal);
            let group = self.data.get(idx)?;
            if group.bytes.len() < group.toc_group.size as usize {
                return None;
            }

            let mut bitstream = Bitstream::new(&group.bytes);
            bitstream.set_lz77_mode(self.lz77_mode);
            let lf_global = if cached_lf_global.is_none() {
                match self.try_parse_lf_global()? {
                    Ok(lf_global) => Some(lf_global),
                    Err(e) => return Some(Err(e)),
                }
            } else {
                None
            };
            let lf_global = cached_lf_global.or(lf_global.as_ref()).unwrap();
            let params = HfGlobalParams::new(
                &self.image_header.metadata,
                &self.header,
                lf_global,
                self.tracker.as_ref(),
                &self.pool,
            );
            Some(HfGlobal::parse(&mut bitstream, params))
        }
    }

    pub fn pass_group_bitstream(
        &self,
        pass_idx: u32,
        group_idx: u32,
    ) -> Option<Result<PassGroupBitstream>> {
        Some(if self.toc.is_single_entry() {
            if self.all_group_offsets.has_error.load(Ordering::Relaxed) != 0 {
                return Some(Err(Error::HadError));
            }

            if pass_idx != 0 || group_idx != 0 {
                return None;
            }

            let group = self.data.first()?;
            let loaded = self.reading_data_index != 0;
            let mut bitstream = Bitstream::new(&group.bytes);
            bitstream.set_lz77_mode(self.lz77_mode);
            let mut offset = self.all_group_offsets.pass_group.load(Ordering::Relaxed);
            if offset == 0 {
                let hf_global = self.try_parse_hf_global::<i32>(None)?;
                if let Err(e) = hf_global {
                    return Some(Err(e));
                }
                offset = self.all_group_offsets.pass_group.load(Ordering::Relaxed);
            }
            bitstream.skip_bits(offset).unwrap();

            Ok(PassGroupBitstream {
                bitstream,
                partial: !loaded,
            })
        } else {
            let idx = self
                .toc
                .group_index_bitstream_order(TocGroupKind::GroupPass {
                    pass_idx,
                    group_idx,
                });
            let group = self.data.get(idx)?;
            let partial = group.bytes.len() < group.toc_group.size as usize;

            let mut bitstream = Bitstream::new(&group.bytes);
            bitstream.set_lz77_mode(self.lz77_mode);
            Ok(PassGroupBitstream { bitstream, partial })
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
