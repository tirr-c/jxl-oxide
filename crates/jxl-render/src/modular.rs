use jxl_frame::{data::GlobalModular, FrameHeader};
use jxl_grid::SimpleGrid;
use jxl_image::BitDepth;
use jxl_modular::{image::TransformedModularSubimage, ChannelShift};

use crate::{region::ImageWithRegion, Error, IndexedFrame, Region, RenderCache, Result};

pub(crate) fn render_modular(
    frame: &IndexedFrame,
    cache: &mut RenderCache,
    region: Region,
    pool: &jxl_threadpool::JxlThreadPool,
) -> Result<(ImageWithRegion, GlobalModular)> {
    let image_header = frame.image_header();
    let frame_header = frame.header();
    let metadata = &image_header.metadata;
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

    let jpeg_upsampling = frame_header.jpeg_upsampling;
    let shifts_cbycr =
        [0, 1, 2].map(|idx| ChannelShift::from_jpeg_upsampling(jpeg_upsampling, idx));
    let channels = metadata.encoded_color_channels();

    let bit_depth = metadata.bit_depth;
    let mut fb_xyb = ImageWithRegion::from_region(channels, region);

    let modular_image = gmodular.modular.image_mut().unwrap();
    let groups = modular_image.prepare_groups(frame.pass_shifts())?;
    let lf_group_image = groups.lf_groups;
    let pass_group_image = groups.pass_groups;

    tracing::trace_span!("Decode").in_scope(|| {
        let result = std::sync::RwLock::new(Result::Ok(()));
        pool.scope(|scope| {
            let lf_groups = &mut cache.lf_groups;
            scope.spawn(|_| {
                let r = crate::load_lf_groups(
                    frame,
                    lf_global.vardct.as_ref(),
                    lf_groups,
                    gmodular.ma_config.as_ref(),
                    lf_group_image,
                    modular_region.downsample(3),
                    pool,
                );
                if r.is_err() {
                    *result.write().unwrap() = r;
                }
            });

            struct PassGroupJob<'modular> {
                pass_idx: u32,
                group_idx: u32,
                modular: TransformedModularSubimage<'modular>,
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

    tracing::trace_span!("Convert to float samples", xyb_encoded).in_scope(|| -> Result<_> {
        let channel_data = modular_image.image_channels();
        for ((g, shift), buffer) in channel_data
            .iter()
            .zip(shifts_cbycr)
            .zip(fb_xyb.buffer_mut())
        {
            let region = region.downsample_separate(shift.hshift() as u32, shift.vshift() as u32);
            copy_modular_groups(g, buffer, region, bit_depth, xyb_encoded);
        }

        if channels == 1 {
            fb_xyb.add_channel();
            fb_xyb.add_channel();
            let fb_xyb = fb_xyb.buffer_mut();
            fb_xyb[1] = fb_xyb[0].try_clone()?;
            fb_xyb[2] = fb_xyb[0].try_clone()?;
        }
        if xyb_encoded {
            let fb_xyb = fb_xyb.buffer_mut();
            // Make Y'X'B' to X'Y'B'
            fb_xyb.swap(0, 1);
            let [x, y, b] = fb_xyb else { panic!() };
            let x = x.buf_mut();
            let y = y.buf_mut();
            let b = b.buf_mut();
            for ((x, y), b) in x.iter_mut().zip(y).zip(b) {
                *b += *y;
                *x *= lf_global.lf_dequant.m_x_lf_unscaled();
                *y *= lf_global.lf_dequant.m_y_lf_unscaled();
                *b *= lf_global.lf_dequant.m_b_lf_unscaled();
            }
        }

        Ok(())
    })?;

    Ok((fb_xyb, gmodular))
}

#[inline]
pub fn compute_modular_region(
    frame_header: &FrameHeader,
    gmodular: &GlobalModular,
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

pub fn copy_modular_groups(
    g: &SimpleGrid<i32>,
    buffer: &mut SimpleGrid<f32>,
    region: Region,
    bit_depth: BitDepth,
    xyb_encoded: bool,
) {
    let Region {
        left,
        top,
        width: rwidth,
        height: rheight,
    } = region;
    assert_eq!(rwidth as usize, buffer.width());
    assert_eq!(rheight as usize, buffer.height());

    let width = g.width();
    let height = g.height();

    let g_stride = g.width();
    let buffer_stride = buffer.width();
    let g = g.buf();
    let buffer = buffer.buf_mut();

    let bottom = top.checked_add_unsigned(rheight).unwrap();
    let right = left.checked_add_unsigned(rwidth).unwrap();

    if bottom <= 0 || top > height as i32 || right <= 0 || left > width as i32 {
        buffer.fill(0f32);
        return;
    }

    if top < 0 {
        buffer[..(-top) as usize * buffer_stride].fill(0f32);
    }

    if bottom as usize > height {
        let remaining = bottom as usize - height;
        let from_y = rheight as usize - remaining;
        buffer[from_y * buffer_stride..].fill(0f32);
    }

    let mut col_begin = 0usize;
    let mut col_end = buffer_stride;
    if left < 0 {
        col_begin = (-left) as usize;
    }
    if right as usize > width {
        let remaining = right as usize - width;
        col_end = rwidth as usize - remaining;
    }

    for y in (top.max(0) as usize)..(bottom as usize).min(height) {
        let buffer_y = y.checked_add_signed((-top) as isize).unwrap();
        let buffer_row = &mut buffer[buffer_y * buffer_stride..][..buffer_stride];

        buffer_row[..col_begin].fill(0f32);
        buffer_row[col_end..].fill(0f32);

        let g_row = &g[y * g_stride..][left.max(0) as usize..(right as usize).min(width)];
        let buffer_row = &mut buffer_row[col_begin..col_end];
        if xyb_encoded {
            for (&s, v) in g_row.iter().zip(buffer_row) {
                *v = s as f32;
            }
        } else {
            for (&s, v) in g_row.iter().zip(buffer_row) {
                *v = bit_depth.parse_integer_sample(s);
            }
        }
    }
}
