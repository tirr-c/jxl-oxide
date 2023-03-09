use std::{io::Read, collections::BTreeMap};

use jxl_bitstream::{Bitstream, Bundle};
use jxl_frame::{
    data::HfCoeff,
    filter::Gabor,
    header::{Encoding, FrameType},
    Frame,
    ProgressiveResult,
};
use jxl_grid::SimpleGrid;
use jxl_image::{Headers, ImageMetadata, ColourSpace};
use jxl_modular::ChannelShift;

mod dct;
mod error;
mod filter;
mod vardct;
pub use error::{Error, Result};

#[derive(Debug)]
pub struct RenderContext<'a> {
    image_header: &'a Headers,
    frames: Vec<Frame<'a>>,
    lf_frame: Vec<usize>,
    reference: Vec<usize>,
    loading_frame: Option<Frame<'a>>,
}

impl<'a> RenderContext<'a> {
    pub fn new(image_header: &'a Headers) -> Self {
        Self {
            image_header,
            frames: Vec::new(),
            lf_frame: vec![usize::MAX; 4],
            reference: vec![usize::MAX; 4],
            loading_frame: None,
        }
    }

    fn metadata(&self) -> &'a ImageMetadata {
        &self.image_header.metadata
    }

    fn xyb_encoded(&self) -> bool {
        self.image_header.metadata.xyb_encoded
    }

    fn preserve_current_frame(&mut self) {
        let Some(frame) = self.loading_frame.take() else { return; };

        let header = frame.header();
        let idx = self.frames.len();
        let is_last = header.is_last;

        if !is_last && (header.duration == 0 || header.save_as_reference != 0) && header.frame_type != FrameType::LfFrame {
            let ref_idx = header.save_as_reference as usize;
            self.reference[ref_idx] = idx;
        }
        if header.lf_level != 0 {
            let lf_idx = header.lf_level as usize - 1;
            self.lf_frame[lf_idx] = idx;
        }
        self.frames.push(frame);
    }
}

impl RenderContext<'_> {
    pub fn width(&self) -> u32 {
        self.image_header.size.width
    }

    pub fn height(&self) -> u32 {
        self.image_header.size.height
    }
}

impl RenderContext<'_> {
    pub fn read_icc_if_exists<R: Read>(&mut self, bitstream: &mut Bitstream<R>) -> Result<()> {
        if self.metadata().colour_encoding.want_icc {
            tracing::info!("Image has ICC profile");
            let icc = jxl_color::icc::read_icc(bitstream)?;

            tracing::warn!("Discarding encoded ICC profile");
            drop(icc);
        }

        Ok(())
    }

    pub fn load_cropped<R: Read>(
        &mut self,
        bitstream: &mut Bitstream<R>,
        progressive: bool,
        mut region: Option<(u32, u32, u32, u32)>,
    ) -> Result<ProgressiveResult> {
        let image_header = self.image_header;

        loop {
            let frame = match &mut self.loading_frame {
                Some(frame) => frame,
                slot => {
                    let frame = Frame::parse(bitstream, image_header)?;
                    *slot = Some(frame);
                    slot.as_mut().unwrap()
                },
            };

            let header = frame.header();
            let is_last = header.is_last;
            tracing::info!(
                width = header.width,
                height = header.height,
                frame_type = format_args!("{:?}", header.frame_type),
                upsampling = header.upsampling,
                lf_level = header.lf_level,
                "Decoding {}x{} frame", header.width, header.height
            );

            if let Some(region) = &mut region {
                frame.adjust_region(region);
            };
            let filter = if region.is_some() {
                Box::new(frame.crop_filter_fn(region)) as Box<dyn FnMut(&_, &_, _) -> bool>
            } else {
                Box::new(|_: &_, _: &_, _| true)
            };

            if header.frame_type.is_normal_frame() {
                let result = frame.load_with_filter(bitstream, progressive, filter)?;
                match result {
                    ProgressiveResult::FrameComplete => frame.complete()?,
                    result => return Ok(result),
                }
            } else {
                frame.load_all(bitstream)?;
                frame.complete()?;
            }

            let toc = frame.toc();
            let bookmark = toc.bookmark() + (toc.total_byte_size() * 8);
            self.preserve_current_frame();
            if is_last {
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

impl RenderContext<'_> {
    pub fn render_cropped(&self, region: Option<(u32, u32, u32, u32)>) -> Result<Vec<SimpleGrid<f32>>> {
        let Some(target_frame) = self.frames
            .iter()
            .chain(self.loading_frame.as_ref())
            .find(|f| f.header().frame_type.is_normal_frame())
            else {
                panic!("No regular frame found");
            };

        self.render_frame(target_frame, region)
    }
}

impl<'f> RenderContext<'f> {
    fn render_frame<'a>(&'a self, frame: &'a Frame<'f>, mut region: Option<(u32, u32, u32, u32)>) -> Result<Vec<SimpleGrid<f32>>> {
        if let Some(region) = &mut region {
            frame.adjust_region(region);
        }

        let mut fb = match frame.header().encoding {
            Encoding::Modular => {
                self.render_modular(frame, region)
            },
            Encoding::VarDct => {
                self.render_vardct(frame, region)
            },
        }?;

        let [a, b, c] = &mut fb;
        if frame.header().do_ycbcr {
            jxl_color::ycbcr::ycbcr_upsample([a, b, c], frame.header().jpeg_upsampling);
            jxl_color::ycbcr::perform_inverse_ycbcr([a, b, c]);
        }
        if let Gabor::Enabled(weights) = frame.header().restoration_filter.gab {
            filter::apply_gabor_like([a, b, c], weights);
        }
        filter::apply_epf([a, b, c], &frame.data().lf_group, frame.header());

        let [a, b, c] = fb;
        Ok(vec![a, b, c])
    }

    fn render_modular<'a>(&'a self, frame: &'a Frame<'f>, _region: Option<(u32, u32, u32, u32)>) -> Result<[SimpleGrid<f32>; 3]> {
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

        let width = self.width() as usize;
        let height = self.height() as usize;
        let bit_depth = metadata.bit_depth;
        let mut fb_yxb = [
            SimpleGrid::new(width, height),
            SimpleGrid::new(width, height),
            SimpleGrid::new(width, height),
        ];
        let mut buffers = {
            let [a, b, c] = &mut fb_yxb;
            [a, b, c]
        };
        if frame_header.do_ycbcr {
            // make CbYCr into YCbCr
            buffers.swap(0, 1);
        }

        for ((g, shift), buffer) in channel_data.iter().zip(shifts_cbycr).zip(buffers.iter_mut()) {
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
            let [y, x, b] = buffers;
            let y = y.buf_mut();
            let x = x.buf_mut();
            let b = b.buf_mut();
            for ((y, x), b) in y.iter_mut().zip(x).zip(b) {
                *b += *y;
                *y *= lf_global.lf_dequant.m_y_lf_unscaled();
                *x *= lf_global.lf_dequant.m_x_lf_unscaled();
                *b *= lf_global.lf_dequant.m_b_lf_unscaled();
            }
        }

        Ok(fb_yxb)
    }

    fn render_vardct<'a>(&'a self, frame: &'a Frame<'f>, region: Option<(u32, u32, u32, u32)>) -> Result<[SimpleGrid<f32>; 3]> {
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
        let shifts_ycbcr = [1, 0, 2].map(|idx| {
            ChannelShift::from_jpeg_upsampling(jpeg_upsampling, idx)
        });
        let subsampled = jpeg_upsampling.into_iter().any(|x| x != 0);

        // Modular extra channels are already merged into GlobalModular,
        // so it's okay to drop PassGroup modular
        let mut group_coeffs: BTreeMap<_, HfCoeff> = BTreeMap::new();
        for (&(_, group_idx), group_pass) in &frame_data.group_pass {
            if let Some(region) = region {
                if !frame_header.is_group_collides_region(group_idx, region) {
                    continue;
                }
            }

            let hf_coeff = group_pass.hf_coeff.as_ref().unwrap();
            group_coeffs
                .entry(group_idx as usize)
                .or_insert_with(HfCoeff::empty)
                .merge(hf_coeff);
        }

        let quantizer = &lf_global_vardct.quantizer;
        let oim = &metadata.opsin_inverse_matrix;
        let dequant_matrices = &hf_global.dequant_matrices;
        let lf_chan_corr = &lf_global_vardct.lf_chan_corr;

        if frame_header.flags.use_lf_frame() {
            tracing::error!("LF frame is not supported");
            return Err(Error::NotSupported("LF frame is not supported"));
        }

        let dequantized_lf = frame_data.lf_group
            .iter()
            .filter(|(&lf_group_idx, _)| {
                let Some(region) = region else { return true; };
                frame_header.is_lf_group_collides_region(lf_group_idx, region)
            })
            .map(|(&lf_group_idx, data)| {
                let lf_coeff = data.lf_coeff.as_ref().unwrap();
                let hf_meta = data.hf_meta.as_ref().unwrap();
                let mut dequant_lf = vardct::dequant_lf(frame_header, &lf_global.lf_dequant, quantizer, lf_coeff);
                if !subsampled {
                    vardct::chroma_from_luma_lf(&mut dequant_lf, &lf_global_vardct.lf_chan_corr);
                }
                (lf_group_idx as usize, (dequant_lf, hf_meta))
            })
            .collect::<BTreeMap<_, _>>();

        let group_dim = frame_header.group_dim() as usize;
        let groups_per_row = frame_header.groups_per_row() as usize;

        // key is (y, x) to match raster order
        let varblocks = group_coeffs
            .into_iter()
            .flat_map(|(group_idx, hf_coeff)| {
                let lf_group_id = frame_header.lf_group_idx_from_group_idx(group_idx as u32) as usize;
                let (lf_dequant, hf_meta) = dequantized_lf.get(&lf_group_id).unwrap();
                let x_from_y = &hf_meta.x_from_y;
                let b_from_y = &hf_meta.b_from_y;

                let group_row = group_idx / groups_per_row;
                let group_col = group_idx % groups_per_row;
                let base_lf_left = (group_col % 8) * group_dim;
                let base_lf_top = (group_row % 8) * group_dim;

                hf_coeff.data
                    .into_iter()
                    .map(move |((bx, by), coeff_data)| {
                        let dct_select = coeff_data.dct_select;
                        let y = vardct::dequant_hf_varblock(&coeff_data, 1, oim, quantizer, dequant_matrices, None);
                        let x = vardct::dequant_hf_varblock(&coeff_data, 0, oim, quantizer, dequant_matrices, Some(frame_header.x_qm_scale));
                        let b = vardct::dequant_hf_varblock(&coeff_data, 2, oim, quantizer, dequant_matrices, Some(frame_header.b_qm_scale));
                        let mut yxb = [y, x, b];

                        let (bw, bh) = coeff_data.dct_select.dct_select_size();
                        let bw = bw as usize;
                        let bh = bh as usize;
                        let bx = bx * 8;
                        let by = by * 8;
                        let lf_left = base_lf_left + bx;
                        let lf_top = base_lf_top + by;
                        if !subsampled {
                            let transposed = bw < bh;
                            vardct::chroma_from_luma_hf(&mut yxb, lf_left, lf_top, x_from_y, b_from_y, lf_chan_corr, transposed);
                        }

                        for ((coeff, lf_dequant), shift) in yxb.iter_mut().zip(lf_dequant.iter()).zip(shifts_ycbcr) {
                            let lf_left = lf_left / 8;
                            let lf_top = lf_top / 8;
                            let s_lf_left = lf_left >> shift.hshift();
                            let s_lf_top = lf_top >> shift.vshift();
                            if s_lf_left << shift.hshift() != lf_left || s_lf_top << shift.vshift() != lf_top {
                                continue;
                            }

                            let lf_subgrid = lf_dequant.subgrid(s_lf_left, s_lf_top, bw, bh);
                            let llf = vardct::llf_from_lf(lf_subgrid, dct_select);
                            for y in 0..llf.height() {
                                for x in 0..llf.width() {
                                    *coeff.get_mut(x, y).unwrap() = *llf.get(x, y).unwrap();
                                }
                            }
                        }
                        ((group_row * group_dim + by, group_col * group_dim + bx), (dct_select, yxb))
                    })
            });

        let width = self.width() as usize;
        let height = self.height() as usize;
        let mut fb_yxb = [
            SimpleGrid::new(width, height),
            SimpleGrid::new(width, height),
            SimpleGrid::new(width, height),
        ];
        for ((y, x), (dct_select, mut yxb)) in varblocks {
            let y8 = y / 8;
            let x8 = x / 8;
            let (w8, h8) = dct_select.dct_select_size();
            let tw = w8 * 8;
            let th = h8 * 8;

            for (idx, (coeff, shift)) in yxb.iter_mut().zip(shifts_ycbcr).enumerate() {
                let sx8 = x8 >> shift.hshift();
                let sy8 = y8 >> shift.vshift();
                if sx8 << shift.hshift() != x8 || sy8 << shift.vshift() != y8 {
                    continue;
                }

                let hsize = width - x;
                let vsize = height - y;
                let (limit_w, limit_h) = shift.shift_size((hsize as u32, vsize as u32));
                let w = tw.min(limit_w) as usize;
                let h = th.min(limit_h) as usize;

                let fb = fb_yxb[idx].buf_mut();
                vardct::transform(coeff, dct_select);

                for (idx, &s) in coeff.buf().iter().enumerate() {
                    let iy = idx / tw as usize;
                    if iy >= h {
                        break;
                    }
                    let y = y + (iy << shift.vshift());
                    let ix = idx % tw as usize;
                    if ix >= w {
                        continue;
                    }
                    let x = x + (ix << shift.hshift());
                    fb[y * width + x] = s;
                }
            }
        }

        Ok(fb_yxb)
    }
}
