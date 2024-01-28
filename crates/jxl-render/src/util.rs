use jxl_color::{
    ColorEncodingWithProfile, ColorTransform, ColourEncoding, ColourSpace, EnumColourEncoding,
};
use jxl_frame::{
    data::{LfGlobalVarDct, LfGroup},
    header::FrameType,
    Frame,
};
use jxl_grid::SimpleGrid;
use jxl_image::ImageHeader;
use jxl_modular::{image::TransformedModularSubimage, MaConfig, Sample};
use jxl_threadpool::JxlThreadPool;

use crate::{ImageWithRegion, IndexedFrame, Region, Result};

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

pub(crate) fn load_lf_groups<S: Sample>(
    frame: &IndexedFrame,
    lf_global_vardct: Option<&LfGlobalVarDct>,
    lf_groups: &mut std::collections::HashMap<u32, LfGroup<S>>,
    global_ma_config: Option<&MaConfig>,
    mlf_groups: Vec<TransformedModularSubimage<S>>,
    lf_region: Region,
    pool: &JxlThreadPool,
) -> Result<()> {
    #[derive(Default)]
    struct LfGroupJob<'modular, S: Sample> {
        idx: u32,
        lf_group: Option<LfGroup<S>>,
        modular: Option<TransformedModularSubimage<'modular, S>>,
    }

    let frame_header = frame.header();
    let lf_groups_per_row = frame_header.lf_groups_per_row();
    let group_dim = frame_header.group_dim();
    let num_lf_groups = frame_header.num_lf_groups();

    let mlf_groups = mlf_groups
        .into_iter()
        .map(Some)
        .chain(std::iter::repeat_with(|| None));
    let mut lf_groups_out = (0..num_lf_groups)
        .map(|idx| LfGroupJob {
            idx,
            lf_group: None,
            modular: None,
        })
        .zip(mlf_groups)
        .filter_map(|(job, modular)| {
            let loaded = lf_groups.get(&job.idx).map(|g| !g.partial).unwrap_or(false);
            if loaded {
                return None;
            }

            let idx = job.idx;
            let left = (idx % lf_groups_per_row) * group_dim;
            let top = (idx / lf_groups_per_row) * group_dim;
            let lf_group_region = Region {
                left: left as i32,
                top: top as i32,
                width: group_dim,
                height: group_dim,
            };
            if lf_region.intersection(lf_group_region).is_empty() {
                return None;
            }

            Some(LfGroupJob { modular, ..job })
        })
        .collect::<Vec<_>>();

    let result = std::sync::RwLock::new(Result::Ok(()));
    pool.for_each_mut_slice(
        &mut lf_groups_out,
        |LfGroupJob {
             idx,
             lf_group,
             modular,
         }| {
            match frame.try_parse_lf_group(lf_global_vardct, global_ma_config, modular.take(), *idx)
            {
                Some(Ok(g)) => {
                    *lf_group = Some(g);
                }
                Some(Err(e)) => {
                    *result.write().unwrap() = Err(e.into());
                }
                None => {}
            }
        },
    );

    for LfGroupJob { idx, lf_group, .. } in lf_groups_out.into_iter() {
        if let Some(group) = lf_group {
            lf_groups.insert(idx, group);
        }
    }
    result.into_inner().unwrap()
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
