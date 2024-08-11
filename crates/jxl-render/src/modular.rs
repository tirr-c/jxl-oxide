use jxl_frame::{data::GlobalModular, FrameHeader};
use jxl_modular::{image::TransformedModularSubimage, Sample};

use crate::{util, Error, ImageWithRegion, IndexedFrame, Region, RenderCache, Result};

pub(crate) fn render_modular<S: Sample>(
    frame: &IndexedFrame,
    cache: &mut RenderCache<S>,
    region: Region,
    pool: &jxl_threadpool::JxlThreadPool,
) -> Result<ImageWithRegion> {
    let image_header = frame.image_header();
    let frame_header = frame.header();
    let tracker = frame.alloc_tracker();
    let xyb_encoded = image_header.metadata.xyb_encoded;

    let lf_global = if let Some(x) = &cache.lf_global {
        x
    } else {
        let lf_global = frame
            .try_parse_lf_global()
            .ok_or(Error::IncompleteFrame)??;
        cache.lf_global = Some(lf_global);
        cache.lf_global.as_ref().unwrap()
    };
    let mut gmodular = lf_global.gmodular.try_clone()?;
    let modular_region = compute_modular_region(frame_header, &gmodular, region, false);

    let modular_image = gmodular.modular.image_mut().unwrap();
    let groups = modular_image.prepare_groups(frame.pass_shifts())?;
    let lf_group_image = groups.lf_groups;
    let pass_group_image = groups.pass_groups;

    tracing::trace_span!("Decode").in_scope(|| {
        let result = std::sync::RwLock::new(Result::Ok(()));
        pool.scope(|scope| {
            let lf_groups = &mut cache.lf_groups;
            scope.spawn(|_| {
                let r = util::load_lf_groups(
                    frame,
                    lf_global,
                    lf_groups,
                    lf_group_image,
                    modular_region.downsample(3),
                    pool,
                );
                if let Err(e) = r {
                    *result.write().unwrap() = Err(e);
                }
            });

            struct PassGroupJob<'modular, S: Sample> {
                pass_idx: u32,
                group_idx: u32,
                modular: TransformedModularSubimage<'modular, S>,
            }

            let group_dim = frame_header.group_dim();
            let groups_per_row = frame_header.groups_per_row();
            let jobs = pass_group_image
                .into_iter()
                .enumerate()
                .flat_map(|(pass_idx, pass_image)| {
                    let pass_idx = pass_idx as u32;
                    pass_image
                        .into_iter()
                        .enumerate()
                        .filter_map(move |(group_idx, modular)| {
                            let group_idx = group_idx as u32;
                            let group_x = group_idx % groups_per_row;
                            let group_y = group_idx / groups_per_row;
                            let left = group_x * group_dim;
                            let top = group_y * group_dim;

                            let group_region = Region {
                                left: left as i32,
                                top: top as i32,
                                width: group_dim,
                                height: group_dim,
                            };
                            if group_region.intersection(modular_region).is_empty() {
                                return None;
                            }

                            Some(PassGroupJob {
                                pass_idx,
                                group_idx,
                                modular,
                            })
                        })
                })
                .collect::<Vec<_>>();

            pool.for_each_vec(
                jobs,
                |PassGroupJob {
                     pass_idx,
                     group_idx,
                     modular,
                 }| {
                    let bitstream = match frame.pass_group_bitstream(pass_idx, group_idx) {
                        Some(Ok(bitstream)) => bitstream,
                        Some(Err(e)) => {
                            *result.write().unwrap() = Err(e.into());
                            return;
                        }
                        None => return,
                    };

                    let allow_partial = bitstream.partial;
                    let mut bitstream = bitstream.bitstream;
                    let global_ma_config = gmodular.ma_config.as_ref();
                    let result = &result;
                    let r = jxl_frame::data::decode_pass_group_modular(
                        &mut bitstream,
                        frame_header,
                        global_ma_config,
                        pass_idx,
                        group_idx,
                        modular,
                        allow_partial,
                        tracker,
                        pool,
                    );
                    if !allow_partial && r.is_err() {
                        *result.write().unwrap() = r.map_err(From::from);
                    }
                },
            );
        });
        result.into_inner().unwrap()
    })?;

    tracing::trace_span!("Inverse Modular transform").in_scope(|| {
        modular_image.prepare_subimage().unwrap().finish(pool);
    });

    let mut fb = ImageWithRegion::new(tracker);
    fb.extend_from_gmodular(gmodular);

    if xyb_encoded {
        tracing::trace_span!("Dequant XYB")
            .in_scope(|| fb.convert_modular_xyb(&lf_global.lf_dequant))?;
    }

    Ok(fb)
}

#[inline]
pub fn compute_modular_region<S: Sample>(
    frame_header: &FrameHeader,
    gmodular: &GlobalModular<S>,
    region: Region,
    is_lf: bool,
) -> Region {
    if gmodular.modular.has_palette() || gmodular.modular.has_squeeze() {
        let mut width = frame_header.color_sample_width();
        let mut height = frame_header.color_sample_height();
        if is_lf {
            width = (width + 7) / 8;
            height = (height + 7) / 8;
        }
        let width = width.max(region.width.checked_add_signed(region.left).unwrap());
        let height = height.max(region.height.checked_add_signed(region.top).unwrap());
        Region::with_size(width, height)
    } else {
        region
    }
}
