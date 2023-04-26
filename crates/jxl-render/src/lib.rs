use std::{io::Read, collections::{HashSet, HashMap}};

use jxl_bitstream::{Bitstream, Bundle};
use jxl_color::ColourSpace;
use jxl_frame::{
    data::HfCoeff,
    filter::Gabor,
    header::{Encoding, FrameType},
    Frame,
    ProgressiveResult,
};
use jxl_grid::{Grid, SimpleGrid};
use jxl_image::{Headers, ImageMetadata};
use jxl_modular::ChannelShift;

mod blend;
mod cut_grid;
mod dct;
mod error;
mod filter;
mod vardct;
pub use error::{Error, Result};

#[derive(Debug)]
pub struct RenderContext<'a> {
    inner: ContextInner<'a>,
    state: RenderState,
    icc: Vec<u8>,
}

impl<'a> RenderContext<'a> {
    pub fn new(image_header: &'a Headers) -> Self {
        Self {
            inner: ContextInner::new(image_header),
            state: RenderState::new(),
            icc: Vec::new(),
        }
    }
}

impl RenderContext<'_> {
    #[inline]
    pub fn width(&self) -> u32 {
        self.inner.width()
    }

    #[inline]
    pub fn height(&self) -> u32 {
        self.inner.height()
    }

    #[inline]
    pub fn loaded_keyframes(&self) -> usize {
        self.inner.keyframes.len() + (self.inner.keyframe_in_progress.is_some() as usize)
    }
}

impl RenderContext<'_> {
    pub fn read_icc_if_exists<R: Read>(&mut self, bitstream: &mut Bitstream<R>) -> Result<&[u8]> {
        if self.inner.metadata().colour_encoding.want_icc {
            tracing::debug!("Image has ICC profile");
            let icc = jxl_color::icc::read_icc(bitstream)?;
            self.icc = jxl_color::icc::decode_icc(&icc)?;
        }

        Ok(&self.icc)
    }

    pub fn icc(&self) -> &[u8] {
        &self.icc
    }

    pub fn load_cropped<R: Read>(
        &mut self,
        bitstream: &mut Bitstream<R>,
        progressive: bool,
        region: Option<(u32, u32, u32, u32)>,
    ) -> Result<ProgressiveResult> {
        loop {
            let (result, frame) = self.inner.load_cropped_single(bitstream, progressive, region)?;
            if result != ProgressiveResult::FrameComplete {
                return Ok(result);
            }

            let is_last = frame.header().is_last;
            let toc = frame.toc();
            let bookmark = toc.bookmark() + (toc.total_byte_size() * 8);
            self.inner.preserve_current_frame();
            self.state.preserve_current_frame();
            if is_last {
                break;
            }

            bitstream.skip_to_bookmark(bookmark)?;
        }

        Ok(ProgressiveResult::FrameComplete)
    }

    pub fn load_until_keyframe<R: Read>(
        &mut self,
        bitstream: &mut Bitstream<R>,
        progressive: bool,
        region: Option<(u32, u32, u32, u32)>,
    ) -> Result<ProgressiveResult> {
        loop {
            let (result, frame) = self.inner.load_cropped_single(bitstream, progressive, region)?;
            if result != ProgressiveResult::FrameComplete {
                return Ok(result);
            }

            let is_keyframe = frame.header().is_keyframe();
            let toc = frame.toc();
            let bookmark = toc.bookmark() + (toc.total_byte_size() * 8);
            self.inner.preserve_current_frame();
            self.state.preserve_current_frame();
            if is_keyframe {
                break;
            }

            bitstream.skip_to_bookmark(bookmark)?;
        }

        Ok(ProgressiveResult::FrameComplete)
    }

    pub fn load_all_frames<R: Read>(
        &mut self,
        bitstream: &mut Bitstream<R>,
        progressive: bool,
    ) -> Result<ProgressiveResult> {
        self.load_cropped(bitstream, progressive, None)
    }
}

impl<'a> RenderContext<'a> {
    pub fn keyframe(&self, keyframe_idx: usize) -> Option<&Frame<'a>> {
        if keyframe_idx == self.inner.keyframes.len() {
            if let Some(idx) = self.inner.keyframe_in_progress {
                Some(&self.inner.frames[idx])
            } else {
                let Some(frame) = &self.inner.loading_frame else { return None; };
                frame.header().is_keyframe().then_some(frame)
            }
        } else if let Some(&idx) = self.inner.keyframes.get(keyframe_idx) {
            Some(&self.inner.frames[idx])
        } else {
            None
        }
    }
}

impl RenderContext<'_> {
    fn render_by_index(&mut self, index: usize, region: Option<(u32, u32, u32, u32)>) -> Result<()> {
        let span = tracing::span!(tracing::Level::TRACE, "RenderContext::render_by_index", index);
        let _guard = span.enter();

        if matches!(&self.state.renders[index], FrameRender::Done(_)) {
            return Ok(());
        }

        let deps = self.inner.frame_deps[index];
        for dep in deps.indices() {
            self.render_by_index(dep, None)?;
        }

        tracing::debug!(index, region = format_args!("{:?}", region), "Rendering frame");
        let frame = &self.inner.frames[index];
        let (prev, state) = self.state.renders.split_at_mut(index);
        let state = &mut state[0];
        let reference_frames = ReferenceFrames {
            lf: (deps.lf != usize::MAX).then(|| prev[deps.lf].as_grid().unwrap()),
            refs: deps.ref_slots.map(|r| (r != usize::MAX).then(|| prev[r].as_grid().unwrap())),
        };

        let cache = match state {
            FrameRender::Done(_) => return Ok(()),
            FrameRender::InProgress(cache) => cache,
            FrameRender::None => {
                *state = FrameRender::InProgress(Box::new(RenderCache::new(frame)));
                let FrameRender::InProgress(cache) = state else { unreachable!() };
                cache
            },
        };

        let grid = self.inner.render_frame(frame, reference_frames, cache, region)?;
        *state = FrameRender::Done(grid);

        let mut unref = |idx: usize| {
            tracing::debug!("Dereference frame #{idx}");
            let new_refcount = self.inner.refcounts[idx].saturating_sub(1);
            if new_refcount == 0 {
                tracing::debug!("Frame #{idx} is not referenced, dropping framebuffer");
                self.state.renders[idx] = FrameRender::None;
            }
        };

        if deps.lf != usize::MAX {
            unref(deps.lf);
        }
        for ref_idx in deps.ref_slots {
            if ref_idx != usize::MAX {
                unref(ref_idx);
            }
        }

        Ok(())
    }

    #[inline]
    pub fn render_cropped(
        &mut self,
        region: Option<(u32, u32, u32, u32)>,
    ) -> Result<Vec<SimpleGrid<f32>>> {
        self.render_keyframe_cropped(0, region)
    }

    pub fn render_keyframe_cropped(
        &mut self,
        keyframe_idx: usize,
        region: Option<(u32, u32, u32, u32)>,
    ) -> Result<Vec<SimpleGrid<f32>>> {
        let (frame, grid) = if let Some(&idx) = self.inner.keyframes.get(keyframe_idx) {
            self.render_by_index(idx, region)?;
            let FrameRender::Done(grid) = &self.state.renders[idx] else { panic!(); };
            let frame = &self.inner.frames[idx];
            (frame, grid.clone())
        } else {
            let mut current_frame_grid = None;
            if let Some(frame) = &self.inner.loading_frame {
                if frame.header().frame_type.is_normal_frame() {
                    let ret = self.render_loading_frame(region);
                    match ret {
                        Ok(grid) => current_frame_grid = Some(grid),
                        Err(Error::IncompleteFrame) => {},
                        Err(e) => return Err(e),
                    }
                }
            }

            if let Some(grid) = current_frame_grid {
                let frame = self.inner.loading_frame.as_ref().unwrap();
                (frame, grid)
            } else if let Some(idx) = self.inner.keyframe_in_progress {
                self.render_by_index(idx, region)?;
                let FrameRender::Done(grid) = &self.state.renders[idx] else { panic!(); };
                let frame = &self.inner.frames[idx];
                (frame, grid.clone())
            } else {
                return Err(Error::IncompleteFrame);
            }
        };

        let mut cropped = if let Some((l, t, w, h)) = region {
            let mut cropped = Vec::with_capacity(grid.len());
            for g in grid {
                let mut new_grid = SimpleGrid::new(w as usize, h as usize);
                for (idx, v) in new_grid.buf_mut().iter_mut().enumerate() {
                    let y = idx / w as usize;
                    let x = idx % w as usize;
                    *v = *g.get(x + l as usize, y + t as usize).unwrap();
                }
                cropped.push(new_grid);
            }
            cropped
        } else {
            grid
        };

        if frame.header().save_before_ct {
            frame.transform_color(&mut cropped);
        }
        Ok(cropped)
    }

    fn render_loading_frame(&mut self, region: Option<(u32, u32, u32, u32)>) -> Result<Vec<SimpleGrid<f32>>> {
        let frame = self.inner.loading_frame.as_ref().unwrap();
        let header = frame.header();
        if frame.data().lf_global.is_none() {
            return Err(Error::IncompleteFrame);
        }

        let lf_frame = if header.flags.use_lf_frame() {
            let lf_frame_idx = self.inner.lf_frame[header.lf_level as usize];
            self.render_by_index(lf_frame_idx, None)?;
            Some(self.state.renders[lf_frame_idx].as_grid().unwrap())
        } else {
            None
        };

        let frame = self.inner.loading_frame.as_ref().unwrap();
        if self.state.loading_render_cache.is_none() {
            self.state.loading_render_cache = Some(RenderCache::new(frame));
        }
        let Some(cache) = &mut self.state.loading_render_cache else { unreachable!() };

        let reference_frames = ReferenceFrames {
            lf: lf_frame,
            refs: [None; 4],
        };

        self.inner.render_frame(frame, reference_frames, cache, region)
    }
}

impl<'f> ContextInner<'f> {
    fn render_frame<'a>(
        &'a self,
        frame: &'a Frame<'f>,
        reference_frames: ReferenceFrames<'a>,
        cache: &mut RenderCache,
        mut region: Option<(u32, u32, u32, u32)>,
    ) -> Result<Vec<SimpleGrid<f32>>> {
        if let Some(region) = &mut region {
            frame.adjust_region(region);
        }

        let mut fb = match frame.header().encoding {
            Encoding::Modular => {
                self.render_modular(frame, cache, region)
            },
            Encoding::VarDct => {
                self.render_vardct(frame, reference_frames.lf, cache, region)
            },
        }?;

        let [a, b, c] = &mut fb;
        if frame.header().do_ycbcr {
            filter::apply_jpeg_upsampling([a, b, c], frame.header().jpeg_upsampling);
        }
        if let Gabor::Enabled(weights) = frame.header().restoration_filter.gab {
            filter::apply_gabor_like([a, b, c], weights);
        }
        filter::apply_epf([a, b, c], &frame.data().lf_group, frame.header());

        let [a, b, c] = fb;
        let mut ret = vec![a, b, c];
        self.append_extra_channels(frame, &mut ret);

        self.render_features(frame, &mut ret, reference_frames.refs)?;

        if !frame.header().save_before_ct {
            frame.transform_color(&mut ret);
        }

        Ok(if !frame.header().frame_type.is_normal_frame() {
            ret
        } else if frame.header().resets_canvas {
            let mut cropped = Vec::with_capacity(ret.len());
            let l = (-frame.header().x0) as usize;
            let t = (-frame.header().y0) as usize;
            let w = self.width() as usize;
            let h = self.height() as usize;
            for g in ret {
                if g.width() == w && g.height() == h {
                    cropped.push(g);
                    continue;
                }

                let mut new_grid = SimpleGrid::new(w, h);
                for (idx, v) in new_grid.buf_mut().iter_mut().enumerate() {
                    let y = idx / w;
                    let x = idx % w;
                    *v = *g.get(x + l, y + t).unwrap();
                }
                cropped.push(new_grid);
            }
            cropped
        } else {
            blend::blend(self.image_header, reference_frames.refs, frame, &ret)
        })
    }

    fn append_extra_channels<'a>(
        &'a self,
        frame: &'a Frame<'f>,
        fb: &mut Vec<SimpleGrid<f32>>,
    ) {
        tracing::debug!("Attaching extra channels");

        let frame_data = frame.data();
        let lf_global = frame_data.lf_global.as_ref().unwrap();
        let extra_channel_from = lf_global.gmodular.extra_channel_from();
        let gmodular = &lf_global.gmodular.modular;

        let channel_data = &gmodular.image().channel_data()[extra_channel_from..];

        let width = frame.header().sample_width() as usize;
        let height = frame.header().sample_height() as usize;

        for (g, ec_info) in channel_data.iter().zip(&self.image_header.metadata.ec_info) {
            let bit_depth = ec_info.bit_depth;

            let mut out = SimpleGrid::new(width, height);
            let buffer = out.buf_mut();

            let (gw, gh) = g.group_dim();
            let group_stride = g.groups_per_row();
            for (group_idx, g) in g.groups() {
                let base_x = (group_idx % group_stride) * gw;
                let base_y = (group_idx / group_stride) * gh;
                for (idx, &s) in g.buf().iter().enumerate() {
                    let y = base_y + idx / g.width();
                    if y >= height {
                        break;
                    }
                    let x = base_x + idx % g.width();
                    if x >= width {
                        continue;
                    }

                    buffer[y * width + x] = bit_depth.parse_integer_sample(s);
                }
            }

            fb.push(out);
        }
    }

    fn render_features<'a>(
        &'a self,
        frame: &'a Frame<'f>,
        grid: &mut [SimpleGrid<f32>],
        reference_grids: [Option<&[SimpleGrid<f32>]>; 4],
    ) -> Result<()> {
        let frame_data = frame.data();
        let lf_global = frame_data.lf_global.as_ref().unwrap();

        if let Some(patches) = &lf_global.patches {
            for patch in &patches.patches {
                let Some(ref_grid) = reference_grids[patch.ref_idx as usize] else {
                    return Err(Error::InvalidReference(patch.ref_idx));
                };
                blend::patch(self.image_header, grid, ref_grid, patch);
            }
        }

        if let Some(splines) = &lf_global.splines {
            let mut estimated_area = 0;
            let base_correlations_xb = lf_global.vardct.as_ref().map(|x| {
                (
                    x.lf_chan_corr.base_correlation_x,
                    x.lf_chan_corr.base_correlation_b,
                )
            });
            for quant_spline in &splines.quant_splines {
                let spline = quant_spline.dequant(
                    splines.quant_adjust,
                    base_correlations_xb,
                    &mut estimated_area,
                );
                // Maximum total_estimated_area_reached for Level 5
                if estimated_area
                    > (self.image_header.size.height * self.image_header.size.width + (1 << 18)).min(1 << 22) as u64
                {
                    tracing::warn!(
                        "Large estimated_area of splines, expect slower decoding: {}",
                        estimated_area
                    );
                }
                // Maximum total_estimated_area_reached for Level 10
                if estimated_area
                    > (64 * (self.image_header.size.height * self.image_header.size.width) as u64 + (1u64 << 34))
                        .min(1u64 << 38)
                {
                    return Err(crate::Error::Frame(
                        jxl_frame::Error::TooLargeEstimatedArea(estimated_area),
                    ));
                }
                blend::spline(self.image_header, grid, spline)?;
            }
        }

        if let Some(_noise) = &lf_global.noise {
            tracing::warn!("Noise is not supported");
        }

        Ok(())
    }

    fn render_modular<'a>(
        &'a self,
        frame: &'a Frame<'f>,
        _cache: &mut RenderCache,
        _region: Option<(u32, u32, u32, u32)>,
    ) -> Result<[SimpleGrid<f32>; 3]> {
        let metadata = self.metadata();
        let xyb_encoded = self.xyb_encoded();
        let frame_header = frame.header();
        let frame_data = frame.data();
        let lf_global = frame_data.lf_global.as_ref().ok_or(Error::IncompleteFrame)?;
        let gmodular = &lf_global.gmodular.modular;
        let jpeg_upsampling = frame_header.jpeg_upsampling;
        let shifts_cbycr = [0, 1, 2].map(|idx| {
            ChannelShift::from_jpeg_upsampling(jpeg_upsampling, idx)
        });
        let is_single_channel = !xyb_encoded && metadata.colour_encoding.colour_space == ColourSpace::Grey;

        let channel_data = gmodular.image().channel_data();
        if is_single_channel {
            tracing::error!("Single-channel modular image is not supported");
            return Err(Error::NotSupported("Single-channel modular image is not supported"));
        }

        let width = frame_header.sample_width() as usize;
        let height = frame_header.sample_height() as usize;
        let bit_depth = metadata.bit_depth;
        let mut fb_xyb = [
            SimpleGrid::new(width, height),
            SimpleGrid::new(width, height),
            SimpleGrid::new(width, height),
        ];

        for ((g, shift), buffer) in channel_data.iter().zip(shifts_cbycr).zip(fb_xyb.iter_mut()) {
            let buffer = buffer.buf_mut();
            let (gw, gh) = g.group_dim();
            let group_stride = g.groups_per_row();
            for (group_idx, g) in g.groups() {
                let base_x = (group_idx % group_stride) * gw;
                let base_y = (group_idx / group_stride) * gh;
                for (idx, &s) in g.buf().iter().enumerate() {
                    let y = base_y + idx / g.width();
                    let y = y << shift.vshift();
                    if y >= height {
                        break;
                    }
                    let x = base_x + idx % g.width();
                    let x = x << shift.hshift();
                    if x >= width {
                        continue;
                    }

                    buffer[y * width + x] = if xyb_encoded {
                        s as f32
                    } else {
                        bit_depth.parse_integer_sample(s)
                    };
                }
            }
        }

        if xyb_encoded {
            // Make Y'X'B' to X'Y'B'
            fb_xyb.swap(0, 1);
            let [x, y, b] = &mut fb_xyb;
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

        Ok(fb_xyb)
    }

    fn render_vardct<'a>(
        &'a self,
        frame: &'a Frame<'f>,
        lf_frame: Option<&'a [SimpleGrid<f32>]>,
        cache: &mut RenderCache,
        region: Option<(u32, u32, u32, u32)>,
    ) -> Result<[SimpleGrid<f32>; 3]> {
        let span = tracing::span!(tracing::Level::TRACE, "RenderContext::render_vardct");
        let _guard = span.enter();

        let metadata = self.metadata();
        let frame_header = frame.header();
        let frame_data = frame.data();
        let lf_global = frame_data.lf_global.as_ref().ok_or(Error::IncompleteFrame)?;
        let lf_global_vardct = lf_global.vardct.as_ref().unwrap();
        let hf_global = frame_data.hf_global.as_ref().ok_or(Error::IncompleteFrame)?;
        let hf_global = hf_global.as_ref().expect("HfGlobal not found for VarDCT frame");
        let jpeg_upsampling = frame_header.jpeg_upsampling;
        let shifts_cbycr: [_; 3] = std::array::from_fn(|idx| {
            ChannelShift::from_jpeg_upsampling(jpeg_upsampling, idx)
        });
        let subsampled = jpeg_upsampling.into_iter().any(|x| x != 0);

        // Modular extra channels are already merged into GlobalModular,
        // so it's okay to drop PassGroup modular
        for (&(pass_idx, group_idx), group_pass) in &frame_data.group_pass {
            if let Some(region) = region {
                if !frame_header.is_group_collides_region(group_idx, region) {
                    continue;
                }
            }
            if !cache.coeff_merged.insert((pass_idx, group_idx)) {
                continue;
            }

            let hf_coeff = group_pass.hf_coeff.as_ref().unwrap();
            cache.group_coeffs
                .entry(group_idx as usize)
                .or_insert_with(HfCoeff::empty)
                .merge(hf_coeff);
        }
        let group_coeffs = &cache.group_coeffs;

        let quantizer = &lf_global_vardct.quantizer;
        let oim = &metadata.opsin_inverse_matrix;
        let dequant_matrices = &hf_global.dequant_matrices;
        let lf_chan_corr = &lf_global_vardct.lf_chan_corr;

        let width = frame_header.sample_width() as usize;
        let height = frame_header.sample_height() as usize;
        let width_rounded = ((width + 7) / 8) * 8;
        let height_rounded = ((height + 7) / 8) * 8;
        let mut fb_xyb = [
            SimpleGrid::new(width_rounded, height_rounded),
            SimpleGrid::new(width_rounded, height_rounded),
            SimpleGrid::new(width_rounded, height_rounded),
        ];

        let mut subgrids = {
            let [x, y, b] = &mut fb_xyb;
            let group_dim = frame_header.group_dim() as usize;
            [
                cut_grid::cut_with_block_info(x, group_coeffs, group_dim, shifts_cbycr[0]),
                cut_grid::cut_with_block_info(y, group_coeffs, group_dim, shifts_cbycr[1]),
                cut_grid::cut_with_block_info(b, group_coeffs, group_dim, shifts_cbycr[2]),
            ]
        };

        let lf_group_it = frame_data.lf_group
            .iter()
            .filter(|(&lf_group_idx, _)| {
                let Some(region) = region else { return true; };
                frame_header.is_lf_group_collides_region(lf_group_idx, region)
            });
        let mut hf_meta_map = HashMap::new();
        let mut lf_image_changed = false;
        for (&lf_group_idx, data) in lf_group_it {
            let group_x = lf_group_idx % frame_header.lf_groups_per_row();
            let group_y = lf_group_idx / frame_header.lf_groups_per_row();

            let lf_group_idx = lf_group_idx as usize;
            hf_meta_map.insert(lf_group_idx, data.hf_meta.as_ref().unwrap());

            if lf_frame.is_some() {
                continue;
            }
            if !cache.inserted_lf_groups.insert(lf_group_idx) {
                continue;
            }

            let lf_coeff = data.lf_coeff.as_ref().unwrap();
            let mut dequant_lf = vardct::dequant_lf(
                &lf_global.lf_dequant,
                quantizer,
                lf_coeff,
            );
            if !subsampled {
                vardct::chroma_from_luma_lf(&mut dequant_lf, &lf_global_vardct.lf_chan_corr);
            }

            let group_dim = frame_header.group_dim();
            for ((image, mut g), shift) in cache.dequantized_lf.iter_mut().zip(dequant_lf).zip(shifts_cbycr) {
                let left = (group_x * group_dim) as isize >> shift.hshift();
                let top = (group_y * group_dim) as isize >> shift.vshift();
                image.insert_subgrid(&mut g, left, top);
            }
            lf_image_changed = true;
        }

        if lf_image_changed && lf_frame.is_none() {
            if frame_header.flags.skip_adaptive_lf_smoothing() {
                for (image, g) in cache.smoothed_lf.iter_mut().zip(&cache.dequantized_lf) {
                    let width = image.width();
                    let height = image.height();
                    let buf = image.buf_mut();
                    for y in 0..height {
                        for x in 0..width {
                            buf[y * width + x] = g.get(x, y).copied().unwrap_or(0.0);
                        }
                    }
                }
            } else {
                vardct::adaptive_lf_smoothing(
                    &cache.dequantized_lf,
                    &mut cache.smoothed_lf,
                    &lf_global.lf_dequant,
                    quantizer,
                );
            }
        }

        let dequantized_lf = if let Some(lf_frame) = lf_frame {
            lf_frame
        } else {
            &cache.smoothed_lf
        };

        let group_dim = frame_header.group_dim() as usize;
        let groups_per_row = frame_header.groups_per_row() as usize;

        for (group_idx, hf_coeff) in group_coeffs {
            let mut x = subgrids[0].remove(group_idx).unwrap();
            let mut y = subgrids[1].remove(group_idx).unwrap();
            let mut b = subgrids[2].remove(group_idx).unwrap();
            let lf_group_id = frame_header.lf_group_idx_from_group_idx(*group_idx as u32) as usize;
            let hf_meta = hf_meta_map.get(&lf_group_id).unwrap();
            let x_from_y = &hf_meta.x_from_y;
            let b_from_y = &hf_meta.b_from_y;

            let group_row = group_idx / groups_per_row;
            let group_col = group_idx % groups_per_row;

            for (coord, coeff_data) in &hf_coeff.data {
                let &(bx, by) = coord;
                let mut x = x.get_mut(coord);
                let mut y = y.get_mut(coord);
                let mut b = b.get_mut(coord);
                let dct_select = coeff_data.dct_select;

                if let Some(x) = &mut x {
                    vardct::dequant_hf_varblock(coeff_data, x, 0, oim, quantizer, dequant_matrices, Some(frame_header.x_qm_scale));
                }
                if let Some(y) = &mut y {
                    vardct::dequant_hf_varblock(coeff_data, y, 1, oim, quantizer, dequant_matrices, None);
                }
                if let Some(b) = &mut b {
                    vardct::dequant_hf_varblock(coeff_data, b, 2, oim, quantizer, dequant_matrices, Some(frame_header.b_qm_scale));
                }

                let lf_left = (group_col * group_dim) / 8 + bx;
                let lf_top = (group_row * group_dim) / 8 + by;
                if !subsampled {
                    let lf_left = (lf_left % group_dim) * 8;
                    let lf_top = (lf_top % group_dim) * 8;
                    let mut xyb = [
                        &mut **x.as_mut().unwrap(),
                        &mut **y.as_mut().unwrap(),
                        &mut **b.as_mut().unwrap(),
                    ];
                    vardct::chroma_from_luma_hf(&mut xyb, lf_left, lf_top, x_from_y, b_from_y, lf_chan_corr);
                }

                for ((coeff, lf_dequant), shift) in [x, y, b].into_iter().zip(dequantized_lf.iter()).zip(shifts_cbycr) {
                    let Some(coeff) = coeff else { continue; };

                    let s_lf_left = lf_left >> shift.hshift();
                    let s_lf_top = lf_top >> shift.vshift();
                    if s_lf_left << shift.hshift() != lf_left || s_lf_top << shift.vshift() != lf_top {
                        continue;
                    }

                    let llf = vardct::llf_from_lf(lf_dequant, s_lf_left, s_lf_top, dct_select);
                    for y in 0..llf.height() {
                        for x in 0..llf.width() {
                            *coeff.get_mut(x, y) = *llf.get(x, y).unwrap();
                        }
                    }

                    vardct::transform(coeff, dct_select);
                }
            }
        }

        if width == width_rounded && height == width_rounded {
            Ok(fb_xyb)
        } else {
            Ok(fb_xyb.map(|g| {
                let mut new_g = SimpleGrid::new(width, height);
                for (new_row, row) in new_g.buf_mut().chunks_exact_mut(width).zip(g.buf().chunks_exact(width_rounded)) {
                    new_row.copy_from_slice(&row[..width]);
                }
                new_g
            }))
        }
    }
}

#[derive(Debug)]
struct ContextInner<'a> {
    image_header: &'a Headers,
    frames: Vec<Frame<'a>>,
    keyframes: Vec<usize>,
    keyframe_in_progress: Option<usize>,
    refcounts: Vec<usize>,
    frame_deps: Vec<FrameDependence>,
    lf_frame: [usize; 4],
    reference: [usize; 4],
    loading_frame: Option<Frame<'a>>,
}

impl<'a> ContextInner<'a> {
    fn new(image_header: &'a Headers) -> Self {
        Self {
            image_header,
            frames: Vec::new(),
            keyframes: Vec::new(),
            keyframe_in_progress: None,
            refcounts: Vec::new(),
            frame_deps: Vec::new(),
            lf_frame: [usize::MAX; 4],
            reference: [usize::MAX; 4],
            loading_frame: None,
        }
    }
}

impl<'a> ContextInner<'a> {
    #[inline]
    fn width(&self) -> u32 {
        self.image_header.size.width
    }

    #[inline]
    fn height(&self) -> u32 {
        self.image_header.size.height
    }

    #[inline]
    fn metadata(&self) -> &'a ImageMetadata {
        &self.image_header.metadata
    }

    #[inline]
    fn xyb_encoded(&self) -> bool {
        self.image_header.metadata.xyb_encoded
    }

    fn preserve_current_frame(&mut self) {
        let Some(frame) = self.loading_frame.take() else { return; };

        let header = frame.header();
        let idx = self.frames.len();
        let is_last = header.is_last;

        self.refcounts.push(0);

        let lf = if header.flags.use_lf_frame() {
            let lf = self.lf_frame[header.lf_level as usize];
            self.refcounts[lf] += 1;
            lf
        } else {
            usize::MAX
        };
        for ref_idx in self.reference {
            if ref_idx != usize::MAX {
                self.refcounts[ref_idx] += 1;
            }
        }

        let deps = FrameDependence {
            lf,
            ref_slots: self.reference,
        };

        if !is_last && (header.duration == 0 || header.save_as_reference != 0) && header.frame_type != FrameType::LfFrame {
            let ref_idx = header.save_as_reference as usize;
            self.reference[ref_idx] = idx;
        }
        if header.lf_level != 0 {
            let lf_idx = header.lf_level as usize - 1;
            self.lf_frame[lf_idx] = idx;
        }

        if header.is_keyframe() {
            self.refcounts[idx] += 1;
            self.keyframes.push(idx);
            self.keyframe_in_progress = None;
        } else if header.frame_type.is_normal_frame() {
            self.keyframe_in_progress = Some(idx);
        }

        self.frames.push(frame);
        self.frame_deps.push(deps);
    }
}

impl ContextInner<'_> {
    fn load_cropped_single<R: Read>(
        &mut self,
        bitstream: &mut Bitstream<R>,
        progressive: bool,
        mut region: Option<(u32, u32, u32, u32)>,
    ) -> Result<(ProgressiveResult, &Frame)> {
        let image_header = self.image_header;

        let frame = match &mut self.loading_frame {
            Some(frame) => frame,
            slot => {
                let frame = Frame::parse(bitstream, image_header)?;
                *slot = Some(frame);
                slot.as_mut().unwrap()
            },
        };

        let header = frame.header();
        tracing::debug!(
            width = header.sample_width(),
            height = header.sample_height(),
            frame_type = format_args!("{:?}", header.frame_type),
            encoding = format_args!("{:?}", header.encoding),
            upsampling = header.upsampling,
            lf_level = header.lf_level,
            "Decoding {}x{} frame", header.sample_width(), header.sample_height()
        );

        if let Some(region) = &mut region {
            frame.adjust_region(region);
        };
        let filter = if region.is_some() {
            Box::new(frame.crop_filter_fn(region)) as Box<dyn FnMut(&_, &_, _) -> bool>
        } else {
            Box::new(|_: &_, _: &_, _| true)
        };

        if header.frame_type == FrameType::RegularFrame {
            let result = frame.load_with_filter(bitstream, progressive, filter)?;
            match result {
                ProgressiveResult::FrameComplete => frame.complete()?,
                result => return Ok((result, frame)),
            }
        } else {
            frame.load_all(bitstream)?;
            frame.complete()?;
        }

        Ok((ProgressiveResult::FrameComplete, frame))
    }
}

#[derive(Debug, Copy, Clone)]
struct FrameDependence {
    lf: usize,
    ref_slots: [usize; 4],
}

impl FrameDependence {
    pub fn indices(&self) -> impl Iterator<Item = usize> + 'static {
        std::iter::once(self.lf).chain(self.ref_slots).filter(|&v| v != usize::MAX)
    }
}

#[derive(Debug, Default)]
struct ReferenceFrames<'state> {
    lf: Option<&'state [SimpleGrid<f32>]>,
    #[allow(unused)]
    refs: [Option<&'state [SimpleGrid<f32>]>; 4],
}

#[derive(Debug)]
struct RenderState {
    renders: Vec<FrameRender>,
    loading_render_cache: Option<RenderCache>,
}

impl RenderState {
    fn new() -> Self {
        Self {
            renders: Vec::new(),
            loading_render_cache: None,
        }
    }
}

impl RenderState {
    fn preserve_current_frame(&mut self) {
        if let Some(cache) = self.loading_render_cache.take() {
            self.renders.push(FrameRender::InProgress(Box::new(cache)));
        } else {
            self.renders.push(FrameRender::None);
        }
    }
}

#[derive(Debug)]
enum FrameRender {
    None,
    InProgress(Box<RenderCache>),
    Done(Vec<SimpleGrid<f32>>),
}

impl FrameRender {
    fn as_grid(&self) -> Option<&[SimpleGrid<f32>]> {
        if let Self::Done(grid) = self {
            Some(grid)
        } else {
            None
        }
    }
}

#[derive(Debug)]
struct RenderCache {
    dequantized_lf: [Grid<f32>; 3],
    smoothed_lf: [SimpleGrid<f32>; 3],
    inserted_lf_groups: HashSet<usize>,
    group_coeffs: HashMap<usize, HfCoeff>,
    coeff_merged: HashSet<(u32, u32)>,
}

impl RenderCache {
    fn new(frame: &Frame<'_>) -> Self {
        let frame_header = frame.header();
        let jpeg_upsampling = frame_header.jpeg_upsampling;
        let shifts_cbycr: [_; 3] = std::array::from_fn(|idx| {
            ChannelShift::from_jpeg_upsampling(jpeg_upsampling, idx)
        });

        let lf_width = (frame_header.sample_width() + 7) / 8 * 8;
        let lf_height = (frame_header.sample_height() + 7) / 8 * 8;
        let group_dim = frame_header.lf_group_dim();
        let mut whd = [(lf_width, lf_height, group_dim, group_dim); 3];
        for ((w, h, dw, dh), shift) in whd.iter_mut().zip(shifts_cbycr) {
            *w >>= shift.hshift();
            *h >>= shift.vshift();
            *dw >>= shift.hshift();
            *dh >>= shift.vshift();
        }
        let dequantized_lf = whd.map(|(w, h, dw, dh)| Grid::new(w, h, dw, dh));
        let smoothed_lf = whd.map(|(w, h, _, _)| SimpleGrid::new(w as usize, h as usize));
        Self {
            dequantized_lf,
            smoothed_lf,
            inserted_lf_groups: HashSet::new(),
            group_coeffs: HashMap::new(),
            coeff_merged: HashSet::new(),
        }
    }
}
