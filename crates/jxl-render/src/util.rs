use jxl_color::{
    ColorEncodingWithProfile, ColorTransform, ColourEncoding, ColourSpace, EnumColourEncoding,
};
use jxl_frame::{
    Frame, FrameHeader,
    data::{LfGlobal, LfGroup},
    filter::{EdgePreservingFilter, EpfParams},
    header::FrameType,
};
use jxl_grid::{AlignedGrid, MutableSubgrid};
use jxl_image::ImageHeader;
use jxl_modular::{ChannelShift, Sample, image::TransformedModularSubimage};
use jxl_threadpool::JxlThreadPool;

use crate::{
    ImageWithRegion, IndexedFrame, Region, Result, image::ImageBuffer, vardct::copy_lf_dequant,
};

pub(crate) fn image_region_to_frame(
    frame: &Frame,
    image_region: Region,
    ignore_lf_level: bool,
) -> Region {
    let image_header = frame.image_header();
    let frame_header = frame.header();
    let full_frame_region = Region::with_size(frame_header.width, frame_header.height);

    let frame_region = if frame_header.frame_type == FrameType::ReferenceOnly {
        full_frame_region
    } else {
        let region = apply_orientation_to_image_region(image_header, image_region);
        region
            .translate(-frame_header.x0, -frame_header.y0)
            .intersection(full_frame_region)
    };

    if ignore_lf_level {
        frame_region
    } else {
        frame_region.downsample(frame_header.lf_level * 3)
    }
}

pub(crate) fn apply_orientation_to_image_region(
    image_header: &ImageHeader,
    image_region: Region,
) -> Region {
    image_region.apply_orientation(image_header)
}

pub(crate) fn pad_lf_region(frame_header: &FrameHeader, frame_region: Region) -> Region {
    if frame_header.lf_level != 0 {
        // Lower level frames might be padded, so apply padding to LF frames
        frame_region.pad(4 * frame_header.lf_level + 32)
    } else {
        frame_region
    }
}

pub(crate) fn pad_upsampling(
    image_header: &ImageHeader,
    frame_header: &FrameHeader,
    frame_region: Region,
) -> Region {
    let color_upsample_factor = frame_header.upsampling.trailing_zeros();
    let max_upsample_factor = frame_header
        .ec_upsampling
        .iter()
        .zip(image_header.metadata.ec_info.iter())
        .map(|(upsampling, ec_info)| upsampling.ilog2() + ec_info.dim_shift)
        .max()
        .unwrap_or(color_upsample_factor);

    if max_upsample_factor > 0 {
        // Additional upsampling pass is needed for every 3 levels of upsampling factor.
        frame_region
            .downsample(max_upsample_factor)
            .pad(2 + (max_upsample_factor - 1) / 3)
            .upsample(max_upsample_factor)
    } else {
        frame_region
    }
}

pub(crate) fn pad_color_region(
    image_header: &ImageHeader,
    frame_header: &FrameHeader,
    frame_region: Region,
) -> Region {
    let color_upsample_factor = frame_header.upsampling.ilog2();
    let mut color_padded_region =
        pad_upsampling(image_header, frame_header, frame_region).downsample(color_upsample_factor);

    // TODO: actual region could be smaller.
    if let EdgePreservingFilter::Enabled(EpfParams { iters, .. }) =
        frame_header.restoration_filter.epf
    {
        // EPF references adjacent samples.
        color_padded_region = if iters == 1 {
            color_padded_region.pad(2)
        } else if iters == 2 {
            color_padded_region.pad(5)
        } else {
            color_padded_region.pad(6)
        };
    }
    if frame_header.restoration_filter.gab.enabled() {
        // Gabor-like filter references adjacent samples.
        color_padded_region = color_padded_region.pad(1);
    }
    if frame_header.do_ycbcr {
        // Chroma upsampling references adjacent samples.
        color_padded_region = color_padded_region.pad(1).downsample(2).upsample(2);
    }
    if frame_header.restoration_filter.epf.enabled() {
        // EPF performs filtering in 8x8 blocks.
        color_padded_region = color_padded_region.container_aligned(8);
    }
    color_padded_region
}

pub(crate) fn load_lf_groups<S: Sample>(
    frame: &IndexedFrame,
    lf_global: &LfGlobal<S>,
    lf_groups: &mut std::collections::HashMap<u32, LfGroup<S>>,
    mlf_groups: Vec<TransformedModularSubimage<S>>,
    lf_region: Region,
    pool: &JxlThreadPool,
) -> Result<Option<ImageWithRegion>> {
    #[derive(Default)]
    struct LfGroupJob<'modular, 'xyb, S: Sample> {
        idx: u32,
        lf_group: Option<LfGroup<S>>,
        modular: Option<TransformedModularSubimage<'modular, S>>,
        lf_xyb: Option<[MutableSubgrid<'xyb, f32>; 3]>,
    }

    let frame_header = frame.header();
    let lf_global_vardct = lf_global.vardct.as_ref();
    let global_ma_config = lf_global.gmodular.ma_config();
    let has_lf_frame = frame_header.flags.use_lf_frame();
    let jpeg_upsampling = frame_header.jpeg_upsampling;
    let shifts_cbycr: [_; 3] =
        std::array::from_fn(|idx| ChannelShift::from_jpeg_upsampling(jpeg_upsampling, idx));
    let mut lf_xyb = if lf_global_vardct.is_some() && !frame_header.flags.use_lf_frame() {
        let tracker = frame.alloc_tracker();
        let mut out = ImageWithRegion::new(3, tracker);
        let Region { width, height, .. } = lf_region;
        for shift in shifts_cbycr {
            let (width, height) = shift.shift_size((width, height));
            let buffer = AlignedGrid::with_alloc_tracker(width as usize, height as usize, tracker)?;
            out.append_channel_shifted(ImageBuffer::F32(buffer), lf_region, shift);
        }
        Some(out)
    } else {
        None
    };

    let lf_groups_per_row = frame_header.lf_groups_per_row();
    let group_dim = frame_header.group_dim();
    let num_lf_groups = frame_header.num_lf_groups();

    let lf_group_base_x = lf_region.left / group_dim as i32;
    let lf_group_base_y = lf_region.top / group_dim as i32;
    let num_cols = lf_region.width.div_ceil(group_dim);
    let num_rows = lf_region.height.div_ceil(group_dim);

    let mut lf_xyb_groups = lf_xyb.as_mut().map(|lf_xyb| {
        let lf_xyb_arr = lf_xyb.as_color_floats_mut();
        let mut idx = 0usize;
        lf_xyb_arr.map(|grid| {
            let grid = grid.as_subgrid_mut();
            let shift = shifts_cbycr[idx];
            let group_width = group_dim >> shift.hshift();
            let group_height = group_dim >> shift.vshift();
            let ret = grid.into_groups_with_fixed_count(
                group_width as usize,
                group_height as usize,
                num_cols as usize,
                num_rows as usize,
            );
            idx += 1;
            ret
        })
    });

    let mlf_groups = mlf_groups
        .into_iter()
        .map(Some)
        .chain(std::iter::repeat_with(|| None));
    let mut lf_groups_out = (0..num_lf_groups)
        .map(|idx| LfGroupJob {
            idx,
            ..Default::default()
        })
        .zip(mlf_groups)
        .filter_map(|(job, modular)| {
            let idx = job.idx;
            let lf_group_x = idx % lf_groups_per_row;
            let lf_group_y = idx / lf_groups_per_row;
            let left = lf_group_x * group_dim;
            let top = lf_group_y * group_dim;
            let lf_group_region = Region {
                left: left as i32,
                top: top as i32,
                width: group_dim,
                height: group_dim,
            };
            if lf_region.intersection(lf_group_region).is_empty() {
                return None;
            }

            let xyb_group_idx = lf_group_y.wrapping_add_signed(-lf_group_base_y) * num_cols
                + lf_group_x.wrapping_add_signed(-lf_group_base_x);
            let lf_xyb = lf_xyb_groups.as_mut().map(|lf_xyb_groups| {
                lf_xyb_groups.each_mut().map(|groups| {
                    std::mem::replace(&mut groups[xyb_group_idx as usize], MutableSubgrid::empty())
                })
            });

            Some(LfGroupJob {
                modular,
                lf_xyb,
                ..job
            })
        })
        .collect::<Vec<_>>();

    let result = std::sync::RwLock::new(Result::Ok(()));
    pool.for_each_mut_slice(&mut lf_groups_out, |job| {
        let LfGroupJob {
            idx,
            ref mut lf_group,
            ref mut modular,
            ref mut lf_xyb,
        } = *job;
        let loaded = lf_group.as_ref().map(|g| !g.partial).unwrap_or(false);

        if !loaded {
            let parse_result =
                frame.try_parse_lf_group(lf_global_vardct, global_ma_config, modular.take(), idx);
            match parse_result {
                Some(Ok(g)) => {
                    *lf_group = Some(g);
                }
                Some(Err(e)) => {
                    *result.write().unwrap() = Err(e.into());
                    return;
                }
                None => {
                    return;
                }
            }
        }

        let lf_group = lf_group.as_ref().unwrap();
        if let (false, Some(lf_xyb)) = (has_lf_frame, lf_xyb) {
            let [lf_x, lf_y, lf_b] = lf_xyb;
            let lf_global_vardct = lf_global_vardct.unwrap();

            let lf_group_x = idx % lf_groups_per_row;
            let lf_group_y = idx / lf_groups_per_row;
            let left = lf_group_x * frame_header.group_dim();
            let top = lf_group_y * frame_header.group_dim();
            let lf_group_region = Region {
                left: left as i32,
                top: top as i32,
                width: group_dim,
                height: group_dim,
            };
            if lf_region.intersection(lf_group_region).is_empty() {
                return;
            }

            let quantizer = &lf_global_vardct.quantizer;
            let lf_coeff = lf_group.lf_coeff.as_ref().unwrap();
            let channel_data = lf_coeff.lf_quant.image().unwrap().image_channels();
            copy_lf_dequant(
                lf_x,
                quantizer,
                lf_global.lf_dequant.m_x_lf,
                &channel_data[1],
                lf_coeff.extra_precision,
            );
            copy_lf_dequant(
                lf_y,
                quantizer,
                lf_global.lf_dequant.m_y_lf,
                &channel_data[0],
                lf_coeff.extra_precision,
            );
            copy_lf_dequant(
                lf_b,
                quantizer,
                lf_global.lf_dequant.m_b_lf,
                &channel_data[2],
                lf_coeff.extra_precision,
            );
        }
    });

    for LfGroupJob { idx, lf_group, .. } in lf_groups_out.into_iter() {
        if let Some(group) = lf_group {
            lf_groups.insert(idx, group);
        }
    }
    result.into_inner().unwrap()?;
    Ok(lf_xyb)
}

pub(crate) fn convert_color_for_record(
    image_header: &ImageHeader,
    do_ycbcr: bool,
    fb: &mut ImageWithRegion,
    pool: &JxlThreadPool,
) -> Result<()> {
    // save_before_ct = false

    let metadata = &image_header.metadata;
    if do_ycbcr {
        // xyb_encoded = false
        fb.convert_modular_color(metadata.bit_depth)?;
        let [cb, y, cr] = fb.as_color_floats_mut();
        jxl_color::ycbcr_to_rgb([cb.buf_mut(), y.buf_mut(), cr.buf_mut()]);
        if metadata.colour_encoding.colour_space() == ColourSpace::Grey {
            fb.remove_color_channels(1);
        }

        fb.set_ct_done(true);
    } else if metadata.xyb_encoded {
        // want_icc = false || is_last = true
        // in any case, blending does not occur when want_icc = true
        let ColourEncoding::Enum(encoding) = &metadata.colour_encoding else {
            return Ok(());
        };

        match encoding.colour_space {
            ColourSpace::Xyb => return Ok(()),
            ColourSpace::Unknown => {
                tracing::warn!(
                    colour_encoding = ?metadata.colour_encoding,
                    "Signalled color encoding is unknown",
                );
                return Ok(());
            }
            _ => {}
        }

        tracing::trace_span!("XYB to target colorspace").in_scope(|| -> Result<_> {
            fb.convert_modular_color(metadata.bit_depth)?;
            let [x, y, b] = fb.as_color_floats_mut();
            tracing::trace!(colour_encoding = ?encoding);
            let transform = ColorTransform::new(
                &ColorEncodingWithProfile::new(EnumColourEncoding::xyb(
                    jxl_color::RenderingIntent::Perceptual,
                )),
                &ColorEncodingWithProfile::new(encoding.clone()),
                &metadata.opsin_inverse_matrix,
                &metadata.tone_mapping,
                &jxl_color::NullCms,
            )
            .unwrap();
            let output_channels = transform
                .run_with_threads(&mut [x.buf_mut(), y.buf_mut(), b.buf_mut()], pool)
                .unwrap();
            fb.remove_color_channels(output_channels);
            Ok(())
        })?;

        fb.set_ct_done(true);
    }

    Ok(())
}

pub(crate) fn mirror(mut offset: isize, len: usize) -> usize {
    loop {
        if offset < 0 {
            offset = -(offset + 1);
        } else if (offset as usize) >= len {
            offset = (-(offset + 1)).wrapping_add_unsigned(len * 2);
        } else {
            return offset as usize;
        }
    }
}

#[derive(Debug)]
pub(crate) struct PaddedGrid<S: Clone> {
    grid: AlignedGrid<S>,
    padding: usize,
}

impl<S: Default + Clone> PaddedGrid<S> {
    pub fn with_alloc_tracker(
        width: usize,
        height: usize,
        padding: usize,
        tracker: Option<&jxl_grid::AllocTracker>,
    ) -> Result<Self> {
        Ok(Self {
            grid: AlignedGrid::with_alloc_tracker(
                width + padding * 2,
                height + padding * 2,
                tracker,
            )?,
            padding,
        })
    }
}

impl<S: Clone> PaddedGrid<S> {
    #[inline]
    pub fn buf_padded(&self) -> &[S] {
        self.grid.buf()
    }

    #[inline]
    pub fn buf_padded_mut(&mut self) -> &mut [S] {
        self.grid.buf_mut()
    }

    pub fn mirror_edges_padding(&mut self) {
        let padding = self.padding;
        let stride = self.grid.width();
        let height = self.grid.height() - padding * 2;

        // Mirror horizontally.
        let buf = self.grid.buf_mut();
        for y in padding..height + padding {
            for x in 0..padding {
                buf[y * stride + x] = buf[y * stride + padding * 2 - x - 1].clone();
                buf[(y + 1) * stride - x - 1] = buf[(y + 1) * stride - padding * 2 + x].clone();
            }
        }

        // Mirror vertically.
        let (out_chunk, in_chunk) = buf.split_at_mut(stride * padding);
        let in_chunk = &in_chunk[..stride * padding];
        for (out_row, in_row) in out_chunk
            .chunks_exact_mut(stride)
            .zip(in_chunk.chunks_exact(stride).rev())
        {
            out_row.clone_from_slice(in_row);
        }

        let (in_chunk, out_chunk) = buf.split_at_mut(stride * (height + padding));
        for (out_row, in_row) in out_chunk
            .chunks_exact_mut(stride)
            .zip(in_chunk.chunks_exact(stride).rev())
        {
            out_row.clone_from_slice(in_row);
        }
    }
}
