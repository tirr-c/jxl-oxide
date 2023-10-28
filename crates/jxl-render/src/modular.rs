use jxl_frame::{data::{GlobalModular, PassGroupParams}, FrameHeader};
use jxl_grid::{SimpleGrid, Grid};
use jxl_image::BitDepth;
use jxl_modular::ChannelShift;

use crate::{
    Region,
    RenderCache,
    IndexedFrame,
    region::ImageWithRegion,
    Error,
    Result,
};

pub fn render_modular(
    frame: &IndexedFrame,
    cache: &mut RenderCache,
    region: Region,
) -> Result<(ImageWithRegion, GlobalModular)> {
    let image_header = frame.image_header();
    let frame_header = frame.header();
    let metadata = &image_header.metadata;
    let xyb_encoded = image_header.metadata.xyb_encoded;

    let lf_global = if let Some(x) = &cache.lf_global {
        x
    } else {
        let lf_global = frame.try_parse_lf_global().ok_or(Error::IncompleteFrame)??;
        cache.lf_global = Some(lf_global);
        cache.lf_global.as_ref().unwrap()
    };
    let mut gmodular = lf_global.gmodular.clone();
    let modular_region = compute_modular_region(frame_header, &gmodular, region);

    let jpeg_upsampling = frame_header.jpeg_upsampling;
    let shifts_cbycr = [0, 1, 2].map(|idx| {
        ChannelShift::from_jpeg_upsampling(jpeg_upsampling, idx)
    });
    let channels = metadata.encoded_color_channels();

    let bit_depth = metadata.bit_depth;
    let mut fb_xyb = ImageWithRegion::from_region(channels, region);

    let lf_groups = &mut cache.lf_groups;
    tracing::trace_span!("Load LF groups").in_scope(|| {
        crate::load_lf_groups(frame, lf_global, lf_groups, modular_region.downsample(3), &mut gmodular)
    })?;

    let group_dim = frame_header.group_dim();
    let groups_per_row = frame_header.groups_per_row();
    tracing::trace_span!("Decode pass groups").in_scope(|| -> Result<_> {
        for pass_idx in 0..frame_header.passes.num_passes {
            for group_idx in 0..frame_header.num_groups() {
                let lf_group_idx = frame_header.lf_group_idx_from_group_idx(group_idx);
                let Some(lf_group) = lf_groups.get(&lf_group_idx) else { continue; };
                let Some(bitstream) = frame.pass_group_bitstream(pass_idx, group_idx).transpose()? else { continue; };
                let allow_partial = bitstream.partial;
                let mut bitstream = bitstream.bitstream;

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
                    continue;
                }

                let shift = frame.pass_shifts(pass_idx);
                let result = jxl_frame::data::decode_pass_group(
                    &mut bitstream,
                    PassGroupParams {
                        frame_header,
                        lf_group,
                        pass_idx,
                        group_idx,
                        shift,
                        gmodular: &mut gmodular,
                        vardct: None,
                        allow_partial,
                    },
                );
                if !allow_partial {
                    result?;
                }
            }
        }
        Ok(())
    })?;

    tracing::trace_span!("Inverse Modular transform").in_scope(|| {
        gmodular.modular.inverse_transform();
    });

    tracing::trace_span!("Convert to float samples", xyb_encoded).in_scope(|| {
        let channel_data = gmodular.modular.image().channel_data();
        for ((g, shift), buffer) in channel_data.iter().zip(shifts_cbycr).zip(fb_xyb.buffer_mut()) {
            let region = region.downsample_separate(shift.hshift() as u32, shift.vshift() as u32);
            copy_modular_groups(g, buffer, region, bit_depth, xyb_encoded);
        }

        if channels == 1 {
            fb_xyb.add_channel();
            fb_xyb.add_channel();
            let fb_xyb = fb_xyb.buffer_mut();
            fb_xyb[1] = fb_xyb[0].clone();
            fb_xyb[2] = fb_xyb[0].clone();
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
    });

    Ok((fb_xyb, gmodular))
}

#[inline]
pub fn compute_modular_region(
    frame_header: &FrameHeader,
    gmodular: &GlobalModular,
    region: Region,
) -> Region {
    if gmodular.modular.has_palette() || gmodular.modular.has_squeeze() {
        Region::with_size(frame_header.color_sample_width(), frame_header.color_sample_height())
    } else {
        region
    }
}

pub fn copy_modular_groups(
    g: &Grid<i32>,
    buffer: &mut SimpleGrid<f32>,
    region: Region,
    bit_depth: BitDepth,
    xyb_encoded: bool,
) {
    let stride = buffer.width();
    let (gw, gh) = g.group_dim();
    let group_stride = g.groups_per_row();
    let buffer = buffer.buf_mut();
    for (group_idx, g) in g.groups() {
        let base_x = (group_idx % group_stride) * gw;
        let base_y = (group_idx / group_stride) * gh;
        let group_region = Region {
            left: base_x as i32,
            top: base_y as i32,
            width: gw as u32,
            height: gh as u32,
        };
        let region_intersection = region.intersection(group_region);
        if region_intersection.is_empty() {
            continue;
        }

        let group_x = region.left.abs_diff(region_intersection.left) as usize;
        let group_y = region.top.abs_diff(region_intersection.top) as usize;

        let begin_x = region_intersection.left.abs_diff(group_region.left) as usize;
        let begin_y = region_intersection.top.abs_diff(group_region.top) as usize;
        let end_x = begin_x + region_intersection.width as usize;
        let end_y = begin_y + region_intersection.height as usize;
        for (idx, &s) in g.buf().iter().enumerate() {
            let x = idx % g.width();
            let y = idx / g.width();
            if y >= end_y {
                break;
            }
            if y < begin_y || !(begin_x..end_x).contains(&x) {
                continue;
            }

            buffer[(group_y + y - begin_y) * stride + (group_x + x - begin_x)] = if xyb_encoded {
                s as f32
            } else {
                bit_depth.parse_integer_sample(s)
            };
        }
    }
}
