use jxl_color::{
    ColorEncodingWithProfile, ColorTransform, ColourEncoding, ColourSpace, EnumColourEncoding,
};
use jxl_frame::{
    data::{LfGlobal, LfGroup},
    filter::{EdgePreservingFilter, EpfParams},
    header::FrameType,
    Frame, FrameHeader,
};
use jxl_grid::{CutGrid, SimpleGrid};
use jxl_image::ImageHeader;
use jxl_modular::{image::TransformedModularSubimage, ChannelShift, Sample};
use jxl_threadpool::JxlThreadPool;

use crate::{vardct::copy_lf_dequant, ImageWithRegion, IndexedFrame, Region, Result};

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
    let image_width = image_header.width_with_orientation();
    let image_height = image_header.height_with_orientation();
    let (_, _, mut left, mut top) = image_header.metadata.apply_orientation(
        image_width,
        image_height,
        image_region.left,
        image_region.top,
        true,
    );
    let (_, _, mut right, mut bottom) = image_header.metadata.apply_orientation(
        image_width,
        image_height,
        image_region.left + image_region.width as i32 - 1,
        image_region.top + image_region.height as i32 - 1,
        true,
    );

    if left > right {
        std::mem::swap(&mut left, &mut right);
    }
    if top > bottom {
        std::mem::swap(&mut top, &mut bottom);
    }
    let width = right.abs_diff(left) + 1;
    let height = bottom.abs_diff(top) + 1;
    Region {
        left,
        top,
        width,
        height,
    }
}

pub(crate) fn pad_lf_region(frame_header: &FrameHeader, frame_region: Region) -> Region {
    if frame_header.lf_level != 0 {
        // Lower level frames might be padded, so apply padding to LF frames
        frame_region.pad(4 * frame_header.lf_level + 32)
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
    let max_upsample_factor = frame_header
        .ec_upsampling
        .iter()
        .zip(image_header.metadata.ec_info.iter())
        .map(|(upsampling, ec_info)| upsampling.ilog2() + ec_info.dim_shift)
        .max()
        .unwrap_or(color_upsample_factor);

    let mut color_padded_region = if max_upsample_factor > 0 {
        // Additional upsampling pass is needed for every 3 levels of upsampling factor.
        let padded_region = frame_region
            .downsample(max_upsample_factor)
            .pad(2 + (max_upsample_factor - 1) / 3);
        let upsample_diff = max_upsample_factor - color_upsample_factor;
        padded_region.upsample(upsample_diff)
    } else {
        frame_region
    };

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
        lf_xyb: Option<[CutGrid<'xyb, f32>; 3]>,
    }

    let frame_header = frame.header();
    let lf_global_vardct = lf_global.vardct.as_ref();
    let global_ma_config = lf_global.gmodular.ma_config();
    let mut lf_xyb = lf_global_vardct
        .map(|_| {
            ImageWithRegion::from_region_and_tracker(3, lf_region, false, frame.alloc_tracker())
        })
        .transpose()?;
    let has_lf_frame = frame_header.flags.use_lf_frame();
    let jpeg_upsampling = frame_header.jpeg_upsampling;
    let shifts_cbycr: [_; 3] =
        std::array::from_fn(|idx| ChannelShift::from_jpeg_upsampling(jpeg_upsampling, idx));

    let lf_groups_per_row = frame_header.lf_groups_per_row();
    let group_dim = frame_header.group_dim();
    let num_lf_groups = frame_header.num_lf_groups();

    let lf_group_base_x = lf_region.left / group_dim as i32;
    let lf_group_base_y = lf_region.top / group_dim as i32;
    let num_cols = lf_region.width.div_ceil(group_dim);
    let num_rows = lf_region.height.div_ceil(group_dim);

    let mut lf_xyb_groups = lf_xyb.as_mut().map(|lf_xyb| {
        let lf_xyb_arr = <&mut [_; 3]>::try_from(lf_xyb.buffer_mut()).unwrap();
        let mut idx = 0usize;
        lf_xyb_arr.each_mut().map(|grid| {
            let grid = CutGrid::from_simple_grid(grid);
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
                    std::mem::replace(&mut groups[xyb_group_idx as usize], CutGrid::empty())
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

pub(crate) fn upsample_lf(
    image: &ImageWithRegion,
    frame: &IndexedFrame,
    frame_region: Region,
) -> Result<ImageWithRegion> {
    let factor = frame.header().lf_level * 3;
    let step = 1usize << factor;
    let new_region = image.region().upsample(factor);
    let mut new_image = ImageWithRegion::from_region_and_tracker(
        image.channels(),
        new_region,
        image.ct_done(),
        frame.alloc_tracker(),
    )?;
    for (original, target) in image.buffer().iter().zip(new_image.buffer_mut()) {
        let height = original.height();
        let width = original.width();
        let stride = target.width();

        let original = original.buf();
        let target = target.buf_mut();
        for y in 0..height {
            let original = &original[y * width..];
            let target = &mut target[y * step * stride..];
            for (x, &value) in original[..width].iter().enumerate() {
                target[x * step..][..step].fill(value);
            }
            for row in 1..step {
                target.copy_within(..stride, stride * row);
            }
        }
    }
    new_image.clone_intersection(frame_region)
}

pub(crate) fn convert_color_for_record(
    image_header: &ImageHeader,
    do_ycbcr: bool,
    grid: &mut [SimpleGrid<f32>],
    pool: &JxlThreadPool,
) -> bool {
    // save_before_ct = false

    let metadata = &image_header.metadata;
    if do_ycbcr {
        // xyb_encoded = false
        let [cb, y, cr, ..] = grid else { panic!() };
        jxl_color::ycbcr_to_rgb([cb, y, cr]);
    } else if metadata.xyb_encoded {
        // want_icc = false || is_last = true
        // in any case, blending does not occur when want_icc = true
        let ColourEncoding::Enum(encoding) = &metadata.colour_encoding else {
            return false;
        };

        match encoding.colour_space {
            ColourSpace::Xyb => return false,
            ColourSpace::Unknown => {
                tracing::warn!(
                    colour_encoding = ?metadata.colour_encoding,
                    "Signalled color encoding is unknown",
                );
                return false;
            }
            _ => {}
        }

        let [x, y, b, ..] = grid else { panic!() };
        tracing::trace_span!("XYB to target colorspace").in_scope(|| {
            tracing::trace!(colour_encoding = ?encoding);
            let transform = ColorTransform::new(
                &ColorEncodingWithProfile::new(EnumColourEncoding::xyb(
                    jxl_color::RenderingIntent::Perceptual,
                )),
                &ColorEncodingWithProfile::new(encoding.clone()),
                &metadata.opsin_inverse_matrix,
                &metadata.tone_mapping,
            )
            .unwrap();
            transform
                .run_with_threads(
                    &mut [x.buf_mut(), y.buf_mut(), b.buf_mut()],
                    &jxl_color::NullCms,
                    pool,
                )
                .unwrap();
        });
    }

    // color transform is done
    true
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
