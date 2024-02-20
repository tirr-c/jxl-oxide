use std::collections::HashMap;

use jxl_frame::{
    data::{GlobalModular, HfGlobal, LfGlobal, LfGroup, PassGroupParams, PassGroupParamsVardct},
    FrameHeader,
};
use jxl_grid::{CutGrid, SharedSubgrid, SimpleGrid};
use jxl_image::ImageHeader;
use jxl_modular::{image::TransformedModularSubimage, ChannelShift, Sample};
use jxl_vardct::{
    BlockInfo, LfChannelCorrelation, LfChannelDequantization, Quantizer, TransformType,
};

use crate::{
    dct, modular, util, Error, ImageWithRegion, IndexedFrame, Reference, Region, RenderCache,
    Result,
};

mod transform;
pub use transform::transform;

#[cfg(target_arch = "x86_64")]
mod x86_64;
#[cfg(target_arch = "x86_64")]
use x86_64 as impls;

#[cfg(target_arch = "aarch64")]
mod aarch64;
#[cfg(target_arch = "aarch64")]
use aarch64 as impls;

mod generic;
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
use generic as impls;

pub(crate) fn render_vardct<S: Sample>(
    frame: &IndexedFrame,
    lf_frame: Option<&Reference<S>>,
    cache: &mut RenderCache<S>,
    region: Region,
    pool: &jxl_threadpool::JxlThreadPool,
) -> Result<(ImageWithRegion, GlobalModular<S>)> {
    let span = tracing::span!(tracing::Level::TRACE, "Render VarDCT");
    let _guard = span.enter();

    let image_header = frame.image_header();
    let frame_header = frame.header();
    let tracker = frame.alloc_tracker();

    let jpeg_upsampling = frame_header.jpeg_upsampling;
    let shifts_cbycr: [_; 3] =
        std::array::from_fn(|idx| ChannelShift::from_jpeg_upsampling(jpeg_upsampling, idx));
    let subsampled = jpeg_upsampling.into_iter().any(|x| x != 0);

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
    let lf_global_vardct = lf_global.vardct.as_ref().unwrap();

    let width = frame_header.color_sample_width() as usize;
    let height = frame_header.color_sample_height() as usize;
    let (width_rounded, height_rounded) = {
        let mut bw = (width + 7) / 8;
        let mut bh = (height + 7) / 8;
        let h_upsample = jpeg_upsampling.into_iter().any(|j| j == 1 || j == 2);
        let v_upsample = jpeg_upsampling.into_iter().any(|j| j == 1 || j == 3);
        if h_upsample {
            bw = (bw + 1) / 2 * 2;
        }
        if v_upsample {
            bh = (bh + 1) / 2 * 2;
        }
        (bw * 8, bh * 8)
    };

    let aligned_region = region.container_aligned(frame_header.group_dim());
    let aligned_lf_region = {
        // group_dim is multiple of 8
        let aligned_region_div8 = Region {
            left: aligned_region.left / 8,
            top: aligned_region.top / 8,
            width: aligned_region.width / 8,
            height: aligned_region.height / 8,
        };
        if frame_header.flags.skip_adaptive_lf_smoothing() {
            aligned_region_div8
        } else {
            aligned_region_div8.pad(1)
        }
        .container_aligned(frame_header.group_dim())
    };

    let aligned_region = aligned_region.intersection(Region::with_size(
        width_rounded as u32,
        height_rounded as u32,
    ));
    let aligned_lf_region = aligned_lf_region.intersection(Region::with_size(
        width_rounded as u32 / 8,
        height_rounded as u32 / 8,
    ));
    let modular_region =
        modular::compute_modular_region(frame_header, &gmodular, aligned_region, false);
    let modular_lf_region =
        modular::compute_modular_region(frame_header, &gmodular, aligned_lf_region, true)
            .intersection(Region::with_size(
                width_rounded as u32 / 8,
                height_rounded as u32 / 8,
            ));

    let mut fb_xyb = ImageWithRegion::from_region_and_tracker(3, modular_region, false, tracker)?;

    let mut modular_image = gmodular.modular.image_mut();
    let groups = modular_image
        .as_mut()
        .map(|x| x.prepare_groups(frame.pass_shifts()))
        .transpose()?;
    let (lf_group_image, pass_group_image) = groups.map(|x| (x.lf_groups, x.pass_groups)).unzip();
    let lf_group_image = lf_group_image.unwrap_or_else(Vec::new);
    let pass_group_image = pass_group_image.unwrap_or_else(|| {
        let passes = frame_header.passes.num_passes as usize;
        let mut ret = Vec::with_capacity(passes);
        ret.resize_with(passes, Vec::new);
        ret
    });

    let lf_groups = &mut cache.lf_groups;
    tracing::trace_span!("Load LF groups").in_scope(|| {
        util::load_lf_groups(
            frame,
            lf_global.vardct.as_ref(),
            lf_groups,
            gmodular.ma_config.as_ref(),
            lf_group_image,
            modular_lf_region,
            pool,
        )
    })?;

    let group_dim = frame_header.group_dim();
    let (hf_cfl_data, mut lf_xyb) =
        tracing::trace_span!("Copy LFQuant").in_scope(|| -> Result<_> {
            let mut hf_cfl_data = (!subsampled)
                .then(|| {
                    ImageWithRegion::from_region_and_tracker(
                        2,
                        modular_lf_region.downsample(3),
                        false,
                        tracker,
                    )
                })
                .transpose()?;

            let mut lf_xyb =
                ImageWithRegion::from_region_and_tracker(3, modular_lf_region, false, tracker)?;

            if let Some(x) = lf_frame {
                let lf_frame = std::sync::Arc::clone(&x.image).run_with_image()?;
                let lf_frame = lf_frame.blend(None, pool)?;
                lf_frame.clone_region_channel(modular_lf_region, 0, &mut lf_xyb.buffer_mut()[0]);
                lf_frame.clone_region_channel(modular_lf_region, 1, &mut lf_xyb.buffer_mut()[1]);
                lf_frame.clone_region_channel(modular_lf_region, 2, &mut lf_xyb.buffer_mut()[2]);
            }

            let lf_groups_per_row = frame_header.lf_groups_per_row();
            for idx in 0..frame_header.num_lf_groups() {
                let Some(lf_group) = lf_groups.get(&idx) else {
                    continue;
                };

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
                if modular_lf_region.intersection(lf_group_region).is_empty() {
                    continue;
                }

                let left = left - modular_lf_region.left as u32;
                let top = top - modular_lf_region.top as u32;

                if lf_frame.is_none() {
                    let quantizer = &lf_global_vardct.quantizer;
                    let lf_coeff = lf_group.lf_coeff.as_ref().unwrap();
                    let channel_data = lf_coeff.lf_quant.image().unwrap().image_channels();
                    let [lf_x, lf_y, lf_b] = lf_xyb.buffer_mut() else {
                        panic!()
                    };
                    copy_lf_dequant(
                        lf_x,
                        left as usize >> shifts_cbycr[0].hshift(),
                        top as usize >> shifts_cbycr[0].vshift(),
                        quantizer,
                        lf_global.lf_dequant.m_x_lf,
                        &channel_data[1],
                        lf_coeff.extra_precision,
                    );
                    copy_lf_dequant(
                        lf_y,
                        left as usize >> shifts_cbycr[1].hshift(),
                        top as usize >> shifts_cbycr[1].vshift(),
                        quantizer,
                        lf_global.lf_dequant.m_y_lf,
                        &channel_data[0],
                        lf_coeff.extra_precision,
                    );
                    copy_lf_dequant(
                        lf_b,
                        left as usize >> shifts_cbycr[2].hshift(),
                        top as usize >> shifts_cbycr[2].vshift(),
                        quantizer,
                        lf_global.lf_dequant.m_b_lf,
                        &channel_data[2],
                        lf_coeff.extra_precision,
                    );
                }

                let Some(hf_meta) = &lf_group.hf_meta else {
                    continue;
                };

                if let Some(cfl) = &mut hf_cfl_data {
                    let corr = &lf_global_vardct.lf_chan_corr;
                    let [x_from_y, b_from_y] = cfl.buffer_mut() else {
                        panic!()
                    };
                    let group_x_from_y = &hf_meta.x_from_y;
                    let group_b_from_y = &hf_meta.b_from_y;
                    let left = left as usize / 8;
                    let top = top as usize / 8;
                    for y in 0..group_x_from_y.height() {
                        for x in 0..group_x_from_y.width() {
                            let v = *group_x_from_y.get(x, y).unwrap();
                            let kx =
                                corr.base_correlation_x + (v as f32 / corr.colour_factor as f32);
                            *x_from_y.get_mut(left + x, top + y).unwrap() = kx;
                        }
                    }
                    for y in 0..group_b_from_y.height() {
                        for x in 0..group_b_from_y.width() {
                            let v = *group_b_from_y.get(x, y).unwrap();
                            let kb =
                                corr.base_correlation_b + (v as f32 / corr.colour_factor as f32);
                            *b_from_y.get_mut(left + x, top + y).unwrap() = kb;
                        }
                    }
                }
            }

            Ok((hf_cfl_data, lf_xyb))
        })?;

    if lf_frame.is_none() {
        if !subsampled {
            tracing::trace_span!("LF CfL").in_scope(|| {
                chroma_from_luma_lf(lf_xyb.buffer_mut(), &lf_global_vardct.lf_chan_corr);
            });
        }

        if !frame_header.flags.skip_adaptive_lf_smoothing() {
            tracing::trace_span!("Adaptive LF smoothing").in_scope(|| {
                adaptive_lf_smoothing(
                    lf_xyb.buffer_mut(),
                    &lf_global.lf_dequant,
                    &lf_global_vardct.quantizer,
                )
            })?;
        }
    }

    let hf_global = if let Some(x) = &cache.hf_global {
        Some(x)
    } else {
        cache.hf_global = frame.try_parse_hf_global(Some(lf_global)).transpose()?;
        cache.hf_global.as_ref()
    };

    tracing::trace_span!("Decode and transform").in_scope(|| -> Result<_> {
        struct PassGroupJob<'g, 'modular, 'lf, S: Sample> {
            group_idx: u32,
            grid_xyb: [CutGrid<'g, f32>; 3],
            pass_modular: HashMap<u32, TransformedModularSubimage<'modular, S>>,
            lf_group: &'lf LfGroup<S>,
        }

        let num_passes = frame_header.passes.num_passes;
        let groups_per_row = frame_header.groups_per_row();
        let mut it = fb_xyb
            .groups_with_group_id(frame_header)
            .into_iter()
            .filter_map(|(group_idx, grid_xyb)| {
                let lf_group_idx = frame_header.lf_group_idx_from_group_idx(group_idx);
                let Some(lf_group) = lf_groups.get(&lf_group_idx) else {
                    return None;
                };

                Some(PassGroupJob {
                    group_idx,
                    grid_xyb,
                    pass_modular: HashMap::new(),
                    lf_group,
                })
            })
            .collect::<Vec<_>>();

        for (pass_idx, pass_image) in pass_group_image.into_iter().enumerate() {
            let mut image_it = pass_image.into_iter().enumerate();
            for PassGroupJob {
                group_idx,
                pass_modular,
                ..
            } in &mut it
            {
                for (image_idx, modular) in &mut image_it {
                    if image_idx == *group_idx as usize {
                        pass_modular.insert(pass_idx as u32, modular);
                        break;
                    }
                }
            }
        }

        let result = std::sync::RwLock::new(Result::Ok(()));
        pool.for_each_vec(
            it,
            |PassGroupJob {
                 group_idx,
                 mut grid_xyb,
                 mut pass_modular,
                 lf_group,
             }| {
                if lf_group.hf_meta.is_none() || hf_global.is_none() {
                    transform_with_lf_grouped(
                        &lf_xyb,
                        &mut grid_xyb,
                        group_idx,
                        frame_header,
                        lf_groups,
                    );
                    return;
                }
                let hf_global = hf_global.unwrap();

                let transform_hf = {
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
                    !group_region.intersection(aligned_region).is_empty()
                };

                let global_ma_config = gmodular.ma_config.as_ref();
                for pass_idx in 0..num_passes {
                    let modular = pass_modular.remove(&pass_idx);

                    let bitstream = match frame.pass_group_bitstream(pass_idx, group_idx) {
                        Some(Ok(bitstream)) => bitstream,
                        Some(Err(e)) => {
                            *result.write().unwrap() = Err(e.into());
                            return;
                        }
                        None => continue,
                    };
                    let allow_partial = bitstream.partial;
                    let mut bitstream = bitstream.bitstream;

                    let vardct = Some(PassGroupParamsVardct {
                        lf_vardct: lf_global_vardct,
                        hf_global,
                        hf_coeff_output: &mut grid_xyb,
                    });

                    let r = jxl_frame::data::decode_pass_group(
                        &mut bitstream,
                        PassGroupParams {
                            frame_header,
                            lf_group,
                            pass_idx,
                            group_idx,
                            global_ma_config,
                            modular,
                            vardct,
                            allow_partial,
                            tracker,
                            pool,
                        },
                    );
                    if !allow_partial && r.is_err() {
                        *result.write().unwrap() = r.map_err(From::from);
                    }
                }

                if transform_hf {
                    dequant_hf_varblock_grouped(
                        &mut grid_xyb,
                        group_idx,
                        image_header,
                        frame_header,
                        lf_global,
                        lf_groups,
                        hf_global,
                    );

                    if let Some(cfl) = &hf_cfl_data {
                        chroma_from_luma_hf_grouped(&mut grid_xyb, group_idx, frame_header, cfl);
                    }

                    transform_with_lf_grouped(
                        &lf_xyb,
                        &mut grid_xyb,
                        group_idx,
                        frame_header,
                        lf_groups,
                    );
                }
            },
        );

        result.into_inner().unwrap()
    })?;

    if fb_xyb.region() != aligned_region {
        fb_xyb = fb_xyb.clone_intersection(aligned_region)?;
    }

    if let Some(modular_image) = modular_image {
        tracing::trace_span!("Extra channel inverse transform").in_scope(|| {
            modular_image.prepare_subimage().unwrap().finish(pool);
        });
    }

    Ok((fb_xyb, gmodular))
}

pub fn copy_lf_dequant<S: Sample>(
    out: &mut SimpleGrid<f32>,
    left: usize,
    top: usize,
    quantizer: &Quantizer,
    m_lf: f32,
    channel_data: &SimpleGrid<S>,
    extra_precision: u8,
) {
    debug_assert!(extra_precision < 4);
    let precision_scale = 1i32 << (9 - extra_precision);
    let scale_inv = quantizer.global_scale as u64 * quantizer.quant_lf as u64;
    let scale = (m_lf as f64 * precision_scale as f64 / scale_inv as f64) as f32;

    let width = channel_data.width();
    let height = channel_data.height();
    let stride = out.width();
    let buf = &mut out.buf_mut()[top * stride + left..];
    let mut grid = CutGrid::from_buf(buf, width, height, stride);
    let buf = channel_data.buf();
    for y in 0..height {
        let row = grid.get_row_mut(y);
        let quant = &buf[y * width..][..width];
        for (out, &q) in row.iter_mut().zip(quant) {
            *out = q.to_i32() as f32 * scale;
        }
    }
}

pub fn adaptive_lf_smoothing(
    lf_image: &mut [SimpleGrid<f32>],
    lf_dequant: &LfChannelDequantization,
    quantizer: &Quantizer,
) -> Result<()> {
    let scale_inv = quantizer.global_scale as u64 * quantizer.quant_lf as u64;
    let lf_x = (512.0 * lf_dequant.m_x_lf as f64 / scale_inv as f64) as f32;
    let lf_y = (512.0 * lf_dequant.m_y_lf as f64 / scale_inv as f64) as f32;
    let lf_b = (512.0 * lf_dequant.m_b_lf as f64 / scale_inv as f64) as f32;

    let [in_x, in_y, in_b] = lf_image else {
        panic!("lf_image should be three-channel image")
    };
    let tracker = in_x.tracker();
    let width = in_x.width();
    let height = in_x.height();

    let in_x = in_x.buf_mut();
    let in_y = in_y.buf_mut();
    let in_b = in_b.buf_mut();

    impls::adaptive_lf_smoothing_impl(
        width,
        height,
        [in_x, in_y, in_b],
        [lf_x, lf_y, lf_b],
        tracker.as_ref(),
    )
}

pub fn dequant_hf_varblock_grouped<S: Sample>(
    out: &mut [CutGrid<'_, f32>; 3],
    group_idx: u32,
    image_header: &ImageHeader,
    frame_header: &FrameHeader,
    lf_global: &LfGlobal<S>,
    lf_groups: &HashMap<u32, LfGroup<S>>,
    hf_global: &HfGlobal,
) {
    let shifts_cbycr: [_; 3] = std::array::from_fn(|idx| {
        ChannelShift::from_jpeg_upsampling(frame_header.jpeg_upsampling, idx)
    });
    let oim = &image_header.metadata.opsin_inverse_matrix;
    let quantizer = &lf_global.vardct.as_ref().unwrap().quantizer;
    let dequant_matrices = &hf_global.dequant_matrices;

    let qm_scale = [
        0.8f32.powi(frame_header.x_qm_scale as i32 - 2),
        1.0f32,
        0.8f32.powi(frame_header.b_qm_scale as i32 - 2),
    ];

    let quant_bias_numerator = oim.quant_bias_numerator;

    let group_dim = frame_header.group_dim();
    let groups_per_row = frame_header.groups_per_row();

    let group_x = group_idx % groups_per_row;
    let group_y = group_idx / groups_per_row;

    let lf_group_idx = frame_header.lf_group_idx_from_group_idx(group_idx);
    let Some(lf_group) = lf_groups.get(&lf_group_idx) else {
        return;
    };
    let left_in_lf = ((group_x % 8) * (group_dim / 8)) as usize;
    let top_in_lf = ((group_y % 8) * (group_dim / 8)) as usize;

    let Some(hf_meta) = &lf_group.hf_meta else {
        return;
    };

    let block_info = &hf_meta.block_info;
    let lf_width = (block_info.width() - left_in_lf).min(group_dim as usize / 8);
    let lf_height = (block_info.height() - top_in_lf).min(group_dim as usize / 8);
    let block_info = hf_meta.block_info.subgrid(
        left_in_lf..(left_in_lf + lf_width),
        top_in_lf..(top_in_lf + lf_height),
    );

    for (channel, coeff) in out.iter_mut().enumerate() {
        let quant_bias = oim.quant_bias[channel];
        let shift = shifts_cbycr[channel];
        for_each_varblocks(
            &block_info,
            shift,
            |VarblockInfo {
                 shifted_bx,
                 shifted_by,
                 dct_select,
                 hf_mul,
             }| {
                let (bw, bh) = dct_select.dct_select_size();
                let left = shifted_bx * 8;
                let top = shifted_by * 8;

                let bw = bw as usize;
                let bh = bh as usize;
                let width = bw * 8;
                let height = bh * 8;

                let need_transpose = dct_select.need_transpose();
                let mul =
                    65536.0 / (quantizer.global_scale as f32 * hf_mul as f32) * qm_scale[channel];

                let matrix = if need_transpose {
                    dequant_matrices.get_transposed(channel, dct_select)
                } else {
                    dequant_matrices.get(channel, dct_select)
                };

                let mut coeff = coeff.subgrid_mut(left..(left + width), top..(top + height));
                for (y, matrix_row) in matrix.chunks_exact(width).enumerate() {
                    let row = coeff.get_row_mut(y);
                    for (q, &m) in row.iter_mut().zip(matrix_row) {
                        let qn = q.to_bits() as i32;
                        *q = qn as f32;
                        if q.abs() <= 1.0 {
                            *q *= quant_bias;
                        } else {
                            *q -= quant_bias_numerator / *q;
                        }
                        *q *= m;
                        *q *= mul;
                    }
                }
            },
        );
    }
}

pub fn chroma_from_luma_lf(coeff_xyb: &mut [SimpleGrid<f32>], lf_chan_corr: &LfChannelCorrelation) {
    let LfChannelCorrelation {
        colour_factor,
        base_correlation_x,
        base_correlation_b,
        x_factor_lf,
        b_factor_lf,
        ..
    } = *lf_chan_corr;

    let x_factor = x_factor_lf as i32 - 128;
    let b_factor = b_factor_lf as i32 - 128;
    let kx = base_correlation_x + (x_factor as f32 / colour_factor as f32);
    let kb = base_correlation_b + (b_factor as f32 / colour_factor as f32);

    let [x, y, b] = coeff_xyb else {
        panic!("coeff_xyb should be three-channel image")
    };
    for ((x, y), b) in x.buf_mut().iter_mut().zip(y.buf()).zip(b.buf_mut()) {
        let y = *y;
        *x += kx * y;
        *b += kb * y;
    }
}

pub fn chroma_from_luma_hf_grouped(
    coeff_xyb: &mut [CutGrid<'_, f32>; 3],
    group_idx: u32,
    frame_header: &FrameHeader,
    cfl_grid: &ImageWithRegion,
) {
    let [coeff_x, coeff_y, coeff_b] = coeff_xyb;

    let cfl_grid_region = cfl_grid.region();
    let cfl_grid = cfl_grid.buffer();

    let group_dim = frame_header.group_dim();
    let groups_per_row = frame_header.groups_per_row();

    let group_x = group_idx % groups_per_row;
    let group_y = group_idx / groups_per_row;
    let gw = coeff_x.width();
    let gh = coeff_x.height();

    let cfl_left = (group_x * group_dim / 64)
        .checked_add_signed(-cfl_grid_region.left)
        .unwrap() as usize;
    let cfl_top = (group_y * group_dim / 64)
        .checked_add_signed(-cfl_grid_region.top)
        .unwrap() as usize;
    let cfl_width = (gw + 63) / 64;
    let cfl_height = (gh + 63) / 64;
    let x_from_y = cfl_grid[0].subgrid(
        cfl_left..(cfl_left + cfl_width),
        cfl_top..(cfl_top + cfl_height),
    );
    let b_from_y = cfl_grid[1].subgrid(
        cfl_left..(cfl_left + cfl_width),
        cfl_top..(cfl_top + cfl_height),
    );

    for y in 0..gh {
        let x_from_y = x_from_y.get_row(y / 64);
        let b_from_y = b_from_y.get_row(y / 64);

        let coeff_x = coeff_x.get_row_mut(y);
        let coeff_y = coeff_y.get_row_mut(y);
        let coeff_b = coeff_b.get_row_mut(y);

        for (x64, (kx, kb)) in x_from_y.iter().zip(b_from_y).enumerate() {
            for dx in 0..((gw - x64 * 64).min(64)) {
                let x = x64 * 64 + dx;
                let coeff_y = coeff_y[x];
                coeff_x[x] += kx * coeff_y;
                coeff_b[x] += kb * coeff_y;
            }
        }
    }
}

pub fn transform_with_lf_grouped<S: Sample>(
    lf: &ImageWithRegion,
    coeff_out: &mut [CutGrid<'_, f32>; 3],
    group_idx: u32,
    frame_header: &FrameHeader,
    lf_groups: &HashMap<u32, LfGroup<S>>,
) {
    use TransformType::*;

    let lf_region = lf.region();
    let lf = lf.buffer();
    let [lf_x, lf_y, lf_b, ..] = lf else { panic!() };
    let shifts_cbycr: [_; 3] = std::array::from_fn(|idx| {
        ChannelShift::from_jpeg_upsampling(frame_header.jpeg_upsampling, idx)
    });

    let group_dim = frame_header.group_dim();
    let groups_per_row = frame_header.groups_per_row();

    let group_x = group_idx % groups_per_row;
    let group_y = group_idx / groups_per_row;
    let lf_base_left = group_x * group_dim / 8;
    let lf_base_top = group_y * group_dim / 8;
    let lf_base_left = lf_base_left.checked_add_signed(-lf_region.left).unwrap();
    let lf_base_top = lf_base_top.checked_add_signed(-lf_region.top).unwrap();
    let lf_width = (lf_region.width - lf_base_left).min(group_dim / 8);
    let lf_height = (lf_region.height - lf_base_top).min(group_dim / 8);
    let lf_base_left = lf_base_left as usize;
    let lf_base_top = lf_base_top as usize;
    let lf = [
        (shifts_cbycr[0], lf_x),
        (shifts_cbycr[1], lf_y),
        (shifts_cbycr[2], lf_b),
    ]
    .map(|(shift, lf)| {
        let lf_base_left = lf_base_left >> shift.hshift();
        let lf_base_top = lf_base_top >> shift.vshift();
        let (lf_width, lf_height) = shift.shift_size((lf_width, lf_height));
        lf.subgrid(
            lf_base_left..(lf_base_left + lf_width as usize),
            lf_base_top..(lf_base_top + lf_height as usize),
        )
    });

    let lf_group_idx = frame_header.lf_group_idx_from_group_idx(group_idx);
    let Some(lf_group) = lf_groups.get(&lf_group_idx) else {
        return;
    };
    let left_in_lf = ((group_x % 8) * (group_dim / 8)) as usize;
    let top_in_lf = ((group_y % 8) * (group_dim / 8)) as usize;

    let Some(hf_meta) = &lf_group.hf_meta else {
        for (coeff, lf) in coeff_out.iter_mut().zip(lf) {
            for y in 0..coeff.height() {
                let coeff_row = coeff.get_row_mut(y);
                let lf_row = lf.get_row(y / 8);
                for (x, v) in coeff_row.iter_mut().enumerate() {
                    *v = lf_row[x / 8];
                }
            }
        }
        return;
    };

    let block_info = hf_meta.block_info.subgrid(
        left_in_lf..(left_in_lf + lf_width as usize),
        top_in_lf..(top_in_lf + lf_height as usize),
    );

    for (channel, (coeff, lf)) in coeff_out.iter_mut().zip(lf).enumerate() {
        let shift = shifts_cbycr[channel];
        for_each_varblocks(
            &block_info,
            shift,
            |VarblockInfo {
                 shifted_bx,
                 shifted_by,
                 dct_select,
                 ..
             }| {
                let (bw, bh) = dct_select.dct_select_size();
                let left = shifted_bx * 8;
                let top = shifted_by * 8;

                let bw = bw as usize;
                let bh = bh as usize;
                let logbw = bw.trailing_zeros() as usize;
                let logbh = bh.trailing_zeros() as usize;

                let mut out = coeff.subgrid_mut(left..(left + bw), top..(top + bh));
                if matches!(
                    dct_select,
                    Hornuss | Dct2 | Dct4 | Dct8x4 | Dct4x8 | Dct8 | Afv0 | Afv1 | Afv2 | Afv3
                ) {
                    debug_assert_eq!(bw * bh, 1);
                    *out.get_mut(0, 0) = *lf.get(shifted_bx, shifted_by);
                } else {
                    for y in 0..bh {
                        for x in 0..bw {
                            *out.get_mut(x, y) = *lf.get(shifted_bx + x, shifted_by + y);
                        }
                    }
                    dct::dct_2d(&mut out, dct::DctDirection::Forward);
                    for y in 0..bh {
                        for x in 0..bw {
                            *out.get_mut(x, y) /= scale_f(y, 5 - logbh) * scale_f(x, 5 - logbw);
                        }
                    }
                }

                let mut block = coeff.subgrid_mut(left..(left + bw * 8), top..(top + bh * 8));
                transform(&mut block, dct_select);
            },
        );
    }
}

#[derive(Debug)]
struct VarblockInfo {
    shifted_bx: usize,
    shifted_by: usize,
    dct_select: TransformType,
    hf_mul: i32,
}

fn for_each_varblocks(
    block_info: &SharedSubgrid<BlockInfo>,
    shift: ChannelShift,
    mut f: impl FnMut(VarblockInfo),
) {
    let w8 = block_info.width();
    let h8 = block_info.height();
    let vshift = shift.vshift();
    let hshift = shift.hshift();

    for by in 0..h8 {
        for bx in 0..w8 {
            let &BlockInfo::Data { dct_select, hf_mul } = block_info.get(bx, by) else {
                continue;
            };
            let shifted_bx = bx >> hshift;
            let shifted_by = by >> vshift;
            if hshift != 0 || vshift != 0 {
                if (shifted_bx << hshift) != bx || (shifted_by << vshift) != by {
                    continue;
                }
                if !matches!(
                    block_info.get(shifted_bx, shifted_by),
                    BlockInfo::Data { .. }
                ) {
                    continue;
                }
            }

            f(VarblockInfo {
                shifted_bx,
                shifted_by,
                dct_select,
                hf_mul,
            })
        }
    }
}

fn scale_f(c: usize, logb: usize) -> f32 {
    // Precomputed for c = 0..32, b = 256
    #[allow(clippy::excessive_precision)]
    const SCALE_F: [f32; 32] = [
        1.0000000000000000,
        0.9996047255830407,
        0.9984194528776054,
        0.9964458326264695,
        0.9936866130906366,
        0.9901456355893141,
        0.9858278282666936,
        0.9807391980963174,
        0.9748868211368796,
        0.9682788310563117,
        0.9609244059440204,
        0.9528337534340876,
        0.9440180941651672,
        0.9344896436056892,
        0.9242615922757944,
        0.9133480844001980,
        0.9017641950288744,
        0.8895259056651056,
        0.8766500784429904,
        0.8631544288990163,
        0.8490574973847023,
        0.8343786191696513,
        0.8191378932865928,
        0.8033561501721485,
        0.7870549181591013,
        0.7702563888779096,
        0.7529833816270532,
        0.7352593067735488,
        0.7171081282466044,
        0.6985543251889097,
        0.6796228528314652,
        0.6603391026591464,
    ];
    SCALE_F[c << logb]
}
