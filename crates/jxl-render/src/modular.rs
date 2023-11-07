use jxl_frame::{data::{GlobalModular, PassGroupParams}, FrameHeader};
use jxl_grid::SimpleGrid;
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

    let modular_image = gmodular.modular.image_mut().unwrap();
    let groups = modular_image.prepare_groups(frame.pass_shifts())?;
    let lf_group_image = groups.lf_groups;
    let pass_group_image = groups.pass_groups;

    let lf_groups = &mut cache.lf_groups;
    tracing::trace_span!("Load LF groups").in_scope(|| {
        crate::load_lf_groups(
            frame,
            lf_global.vardct.as_ref(),
            lf_groups,
            gmodular.ma_config.as_ref(),
            lf_group_image,
            modular_region.downsample(3),
        )
    })?;

    let group_dim = frame_header.group_dim();
    let groups_per_row = frame_header.groups_per_row();
    tracing::trace_span!("Decode pass groups").in_scope(|| -> Result<_> {
        for (pass_idx, pass_image) in pass_group_image.into_iter().enumerate() {
            let pass_idx = pass_idx as u32;
            let mut group_it = pass_image.into_iter();
            for group_idx in 0..frame_header.num_groups() {
                let modular = group_it.next();

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

                let result = jxl_frame::data::decode_pass_group(
                    &mut bitstream,
                    PassGroupParams {
                        frame_header,
                        lf_group,
                        pass_idx,
                        group_idx,
                        global_ma_config: gmodular.ma_config.as_ref(),
                        modular,
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
        modular_image.prepare_subimage().unwrap().finish();
    });

    tracing::trace_span!("Convert to float samples", xyb_encoded).in_scope(|| {
        let channel_data = modular_image.image_channels();
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
    g: &SimpleGrid<i32>,
    buffer: &mut SimpleGrid<f32>,
    region: Region,
    bit_depth: BitDepth,
    xyb_encoded: bool,
) {
    let width = g.width();
    let height = g.height();

    let g_stride = g.width();
    let buffer_stride = buffer.width();
    let g = g.buf();
    let buffer = buffer.buf_mut();
    for y in region.top..region.top.wrapping_add_unsigned(region.height) {
        let buffer_y = y.abs_diff(region.top) as usize;
        for x in region.left..region.left.wrapping_add_unsigned(region.width) {
            let buffer_x = x.abs_diff(region.left) as usize;

            let s = if y < 0 || x < 0 {
                0i32
            } else {
                let x = x as usize;
                let y = y as usize;
                if y >= height || x >= width {
                    0i32
                } else {
                    g[y * g_stride + x]
                }
            };

            buffer[buffer_y * buffer_stride + buffer_x] = if xyb_encoded {
                s as f32
            } else {
                bit_depth.parse_integer_sample(s)
            };
        }
    }
}
