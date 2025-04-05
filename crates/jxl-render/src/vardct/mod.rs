#![allow(unsafe_op_in_unsafe_fn)]

use std::collections::HashMap;

use jxl_frame::{
    FrameHeader,
    data::{HfGlobal, LfGlobal, LfGroup, PassGroupParams, PassGroupParamsVardct},
};
use jxl_grid::{AlignedGrid, MutableSubgrid, SharedSubgrid};
use jxl_image::ImageHeader;
use jxl_modular::{ChannelShift, Sample};
use jxl_threadpool::JxlThreadPool;
use jxl_vardct::{
    BlockInfo, LfChannelCorrelation, LfChannelDequantization, Quantizer, TransformType,
};

use crate::{
    Error, ImageWithRegion, IndexedFrame, Reference, Region, RenderCache, Result,
    image::ImageBuffer, modular, util,
};

mod dct_common;
mod transform_common;

#[cfg(target_arch = "x86_64")]
mod x86_64;
#[cfg(target_arch = "x86_64")]
use x86_64 as impls;

#[cfg(target_arch = "aarch64")]
mod aarch64;
#[cfg(target_arch = "aarch64")]
use aarch64 as impls;

#[cfg(all(target_family = "wasm", target_feature = "simd128"))]
mod wasm32;
#[cfg(all(target_family = "wasm", target_feature = "simd128"))]
use wasm32 as impls;

mod generic;
#[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_family = "wasm", target_feature = "simd128")
)))]
use generic as impls;

pub(crate) fn render_vardct<S: Sample>(
    frame: &IndexedFrame,
    lf_frame: Option<&Reference<S>>,
    cache: &mut RenderCache<S>,
    region: Region,
    pool: &JxlThreadPool,
) -> Result<ImageWithRegion> {
    let span = tracing::span!(tracing::Level::TRACE, "Render VarDCT");
    let _guard = span.enter();

    let image_header = frame.image_header();
    let frame_header = frame.header();
    let tracker = frame.alloc_tracker();

    let jpeg_upsampling = frame_header.jpeg_upsampling;
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
        let mut bw = width.div_ceil(8);
        let mut bh = height.div_ceil(8);
        let h_upsample = jpeg_upsampling.into_iter().any(|j| j == 1 || j == 2);
        let v_upsample = jpeg_upsampling.into_iter().any(|j| j == 1 || j == 3);
        if h_upsample {
            bw = bw.div_ceil(2) * 2;
        }
        if v_upsample {
            bh = bh.div_ceil(2) * 2;
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

    let hf_global = &mut cache.hf_global;
    let lf_groups = &mut cache.lf_groups;
    let group_dim = frame_header.group_dim();

    let result = std::sync::RwLock::new(Result::Ok(()));
    let (mut fb, lf_xyb) = pool.scope(|scope| -> Result<_> {
        if hf_global.is_none() {
            scope.spawn(|_| {
                let ret = tracing::trace_span!("Parse HfGlobal").in_scope(|| -> Result<_> {
                    *hf_global = frame.try_parse_hf_global(Some(lf_global)).transpose()?;
                    Ok(())
                });
                if let Err(e) = ret {
                    *result.write().unwrap() = Err(e);
                }
            });
        }

        let lf_xyb = tracing::trace_span!("Load LF groups").in_scope(|| {
            util::load_lf_groups(
                frame,
                lf_global,
                lf_groups,
                lf_group_image,
                modular_lf_region,
                pool,
            )
        })?;

        let lf_xyb = if let Some(x) = lf_frame {
            tracing::trace_span!("Copy LFQuant").in_scope(|| -> Result<_> {
                let lf_frame = std::sync::Arc::clone(&x.image).run_with_image()?;
                let lf_frame = lf_frame.blend(None, pool)?.try_clone()?;
                Ok(lf_frame)
            })?
        } else {
            let mut lf_xyb = lf_xyb.unwrap();

            if !subsampled {
                tracing::trace_span!("LF CfL").in_scope(|| {
                    chroma_from_luma_lf(
                        lf_xyb.as_color_floats_mut(),
                        &lf_global_vardct.lf_chan_corr,
                    );
                });
            }

            if !frame_header.flags.skip_adaptive_lf_smoothing() {
                tracing::trace_span!("Adaptive LF smoothing").in_scope(|| {
                    adaptive_lf_smoothing(
                        lf_xyb.as_color_floats_mut(),
                        &lf_global.lf_dequant,
                        &lf_global_vardct.quantizer,
                    )
                })?;
            }

            lf_xyb
        };

        let fb = {
            let shifts_cbycr: [_; 3] = std::array::from_fn(|idx| {
                ChannelShift::from_jpeg_upsampling(frame_header.jpeg_upsampling, idx)
            });
            let Region { width, height, .. } = modular_region;

            let mut fb = ImageWithRegion::new(3, tracker);
            for shift in shifts_cbycr {
                let (w8, h8) = shift.shift_size((width.div_ceil(8), height.div_ceil(8)));
                let width = w8 * 8;
                let height = h8 * 8;
                let buffer =
                    AlignedGrid::with_alloc_tracker(width as usize, height as usize, tracker)?;
                fb.append_channel_shifted(ImageBuffer::F32(buffer), modular_region, shift);
            }
            fb
        };

        Ok((fb, lf_xyb))
    })?;
    result.into_inner().unwrap()?;

    let hf_global = cache.hf_global.as_ref();
    let lf_groups = &mut cache.lf_groups;

    let it = tracing::trace_span!("Prepare PassGroup").in_scope(|| {
        fb.color_groups_with_group_id(frame_header)
            .into_iter()
            .filter_map(|(group_idx, grid_xyb)| {
                let lf_group_idx = frame_header.lf_group_idx_from_group_idx(group_idx);
                let lf_group = lf_groups.get(&lf_group_idx)?;

                Some((group_idx, grid_xyb, lf_group))
            })
            .collect::<Vec<_>>()
    });

    tracing::trace_span!("Decode PassGroup").in_scope(|| {
        let Some(hf_global) = hf_global else {
            return Ok(());
        };

        let result = std::sync::RwLock::new(Result::Ok(()));

        pool.scope(|scope| {
            let global_ma_config = gmodular.ma_config.as_ref();

            for (pass_idx, pass_image) in pass_group_image.into_iter().enumerate() {
                let pass_idx = pass_idx as u32;
                let mut image_it = pass_image.into_iter().enumerate();
                for &(group_idx, ref grid_xyb, lf_group) in &it {
                    // SAFETY: All accesses to `grid_xyb` are atomic in this Rayon scope. The scope
                    // captures `fb_xyb` as a unique reference (via `it`), so `grid_xyb` is
                    // accessed exclusively in this scope.
                    let grid_xyb = grid_xyb
                        .each_ref()
                        .map(|grid| unsafe { grid.as_shared().as_atomic_i32() });

                    if lf_group.hf_meta.is_none() {
                        continue;
                    }

                    let bitstream = match frame.pass_group_bitstream(pass_idx, group_idx) {
                        Some(Ok(bitstream)) => bitstream,
                        Some(Err(e)) => {
                            *result.write().unwrap() = Err(e.into());
                            continue;
                        }
                        None => continue,
                    };
                    let allow_partial = bitstream.partial;
                    let mut bitstream = bitstream.bitstream;

                    let modular = image_it
                        .find(|(image_idx, _)| *image_idx == group_idx as usize)
                        .map(|(_, modular)| modular);

                    let result = &result;
                    scope.spawn(move |_| {
                        let vardct = Some(PassGroupParamsVardct {
                            lf_vardct: lf_global_vardct,
                            hf_global,
                            hf_coeff_output: &grid_xyb,
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
                    });
                }
            }
        });

        result.into_inner().unwrap()
    })?;

    tracing::trace_span!("Dequant and transform").in_scope(|| {
        let groups_per_row = frame_header.groups_per_row();

        pool.for_each_vec(it, |job| {
            let (group_idx, mut grid_xyb, lf_group) = job;
            let grid_xyb = &mut grid_xyb;
            let group_x = group_idx % groups_per_row;
            let group_y = group_idx / groups_per_row;

            let transform_hf = {
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

            if lf_group.hf_meta.is_none() || hf_global.is_none() || !transform_hf {
                transform_with_lf_grouped(&lf_xyb, grid_xyb, group_idx, frame_header, lf_groups);
                return;
            }

            let hf_global = hf_global.unwrap();

            dequant_hf_varblock_grouped(
                grid_xyb,
                group_idx,
                image_header,
                frame_header,
                lf_global,
                lf_groups,
                hf_global,
            );

            if !subsampled {
                let hf_meta = lf_group.hf_meta.as_ref().unwrap();
                let lf_chan_corr = &lf_global_vardct.lf_chan_corr;
                let cfl_base_x = ((group_x % 8) * group_dim / 64) as usize;
                let cfl_base_y = ((group_y % 8) * group_dim / 64) as usize;
                let gw = grid_xyb[0].width().div_ceil(64);
                let gh = grid_xyb[0].height().div_ceil(64);
                let x_from_y = hf_meta
                    .x_from_y
                    .as_subgrid()
                    .subgrid(cfl_base_x..(cfl_base_x + gw), cfl_base_y..(cfl_base_y + gh));
                let b_from_y = hf_meta
                    .b_from_y
                    .as_subgrid()
                    .subgrid(cfl_base_x..(cfl_base_x + gw), cfl_base_y..(cfl_base_y + gh));
                chroma_from_luma_hf_grouped(grid_xyb, &x_from_y, &b_from_y, lf_chan_corr);
            }

            transform_with_lf_grouped(&lf_xyb, grid_xyb, group_idx, frame_header, lf_groups);
        });
    });

    if let Some(modular_image) = modular_image {
        tracing::trace_span!("Extra channel inverse transform").in_scope(|| {
            modular_image.prepare_subimage().unwrap().finish(pool);
        });
        fb.extend_from_gmodular(gmodular);
    }

    Ok(fb)
}

pub fn copy_lf_dequant<S: Sample>(
    grid: &mut MutableSubgrid<f32>,
    quantizer: &Quantizer,
    m_lf: f32,
    channel_data: &AlignedGrid<S>,
    extra_precision: u8,
) {
    debug_assert!(extra_precision < 4);
    assert!(grid.width() >= channel_data.width());
    assert!(grid.height() >= channel_data.height());

    let precision_scale = 1i32 << (9 - extra_precision);
    let scale_inv = quantizer.global_scale as u64 * quantizer.quant_lf as u64;
    let scale = (m_lf as f64 * precision_scale as f64 / scale_inv as f64) as f32;

    let width = channel_data.width();
    let height = channel_data.height();
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
    lf_image: [&mut AlignedGrid<f32>; 3],
    lf_dequant: &LfChannelDequantization,
    quantizer: &Quantizer,
) -> Result<()> {
    let scale_inv = quantizer.global_scale as u64 * quantizer.quant_lf as u64;
    let lf_x = (512.0 * lf_dequant.m_x_lf as f64 / scale_inv as f64) as f32;
    let lf_y = (512.0 * lf_dequant.m_y_lf as f64 / scale_inv as f64) as f32;
    let lf_b = (512.0 * lf_dequant.m_b_lf as f64 / scale_inv as f64) as f32;

    let [in_x, in_y, in_b] = lf_image;
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
    out: &mut [MutableSubgrid<'_, f32>; 3],
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
    let block_info = hf_meta.block_info.as_subgrid().subgrid(
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

                let mut coeff = coeff
                    .borrow_mut()
                    .subgrid(left..(left + width), top..(top + height));
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

pub fn chroma_from_luma_lf(
    coeff_xyb: [&mut AlignedGrid<f32>; 3],
    lf_chan_corr: &LfChannelCorrelation,
) {
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

    let [x, y, b] = coeff_xyb;
    for ((x, y), b) in x.buf_mut().iter_mut().zip(y.buf()).zip(b.buf_mut()) {
        let y = *y;
        *x += kx * y;
        *b += kb * y;
    }
}

pub fn chroma_from_luma_hf_grouped(
    coeff_xyb: &mut [MutableSubgrid<'_, f32>; 3],
    x_from_y: &SharedSubgrid<i32>,
    b_from_y: &SharedSubgrid<i32>,
    lf_chan_corr: &LfChannelCorrelation,
) {
    let [coeff_x, coeff_y, coeff_b] = coeff_xyb;

    let gw = coeff_x.width();
    let gh = coeff_x.height();

    for y in 0..gh {
        let x_from_y = x_from_y.get_row(y / 64);
        let b_from_y = b_from_y.get_row(y / 64);

        let coeff_x = coeff_x.get_row_mut(y);
        let coeff_y = coeff_y.get_row_mut(y);
        let coeff_b = coeff_b.get_row_mut(y);

        for (x64, (&kx, &kb)) in x_from_y.iter().zip(b_from_y).enumerate() {
            let kx =
                lf_chan_corr.base_correlation_x + (kx as f32 / lf_chan_corr.colour_factor as f32);
            let kb =
                lf_chan_corr.base_correlation_b + (kb as f32 / lf_chan_corr.colour_factor as f32);

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
    coeff_out: &mut [MutableSubgrid<'_, f32>; 3],
    group_idx: u32,
    frame_header: &FrameHeader,
    lf_groups: &HashMap<u32, LfGroup<S>>,
) {
    let lf_regions = <[_; 3]>::try_from(&lf.regions_and_shifts()[..3]).unwrap();
    let [lf_x, lf_y, lf_b] = lf.as_color_floats();
    let shifts_cbycr: [_; 3] = std::array::from_fn(|idx| {
        ChannelShift::from_jpeg_upsampling(frame_header.jpeg_upsampling, idx)
    });

    let group_dim = frame_header.group_dim();
    let groups_per_row = frame_header.groups_per_row();
    let (group_width, group_height) = frame_header.group_size_for(group_idx);

    let group_x = group_idx % groups_per_row;
    let group_y = group_idx / groups_per_row;
    let lf_base_left = group_x * group_dim / 8;
    let lf_base_top = group_y * group_dim / 8;
    let lf = [
        (lf_regions[0], lf_x),
        (lf_regions[1], lf_y),
        (lf_regions[2], lf_b),
    ]
    .map(|((lf_region, shift), lf)| {
        let lf_base_left = lf_base_left.checked_add_signed(-lf_region.left).unwrap();
        let lf_base_top = lf_base_top.checked_add_signed(-lf_region.top).unwrap();
        let lf_width = (lf_region.width - lf_base_left).min(group_width.div_ceil(8));
        let lf_height = (lf_region.height - lf_base_top).min(group_height.div_ceil(8));
        let lf_base_left = lf_base_left as usize;
        let lf_base_top = lf_base_top as usize;

        let lf_base_left = lf_base_left >> shift.hshift();
        let lf_base_top = lf_base_top >> shift.vshift();
        let (lf_width, lf_height) = shift.shift_size((lf_width, lf_height));
        lf.as_subgrid().subgrid(
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

    let block_info = {
        let lf_region = lf_regions[0].0;
        let lf_base_left = lf_base_left.checked_add_signed(-lf_region.left).unwrap();
        let lf_base_top = lf_base_top.checked_add_signed(-lf_region.top).unwrap();
        let lf_width = (lf_region.width - lf_base_left).min(group_width.div_ceil(8));
        let lf_height = (lf_region.height - lf_base_top).min(group_height.div_ceil(8));

        hf_meta.block_info.as_subgrid().subgrid(
            left_in_lf..(left_in_lf + lf_width as usize),
            top_in_lf..(top_in_lf + lf_height as usize),
        )
    };

    impls::transform_varblocks(&lf, coeff_out, shifts_cbycr, &block_info);
}

#[derive(Debug)]
struct VarblockInfo {
    shifted_bx: usize,
    shifted_by: usize,
    dct_select: TransformType,
    hf_mul: i32,
}

#[inline(always)]
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
