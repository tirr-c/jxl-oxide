use std::collections::HashMap;

use jxl_bitstream::{define_bundle, read_bits, header::{Headers, ColourSpace, ImageMetadata}, Bitstream, Bundle};
use jxl_grid::{Grid, Subgrid};
use jxl_modular::{ChannelShift, ModularChannelParams, Modular, ModularParams, MaConfig, MaContext};
use jxl_vardct::{TransformType, LfChannelDequantization, LfChannelCorrelation, HfBlockContext, Quantizer, HfPass};

use crate::{
    FrameHeader,
    Result,
    header::Encoding,
};

mod noise;
mod patch;
mod spline;
mod toc;
pub use noise::NoiseParameters;
pub use patch::Patches;
pub use spline::Splines;
pub use toc::{Toc, TocGroup, TocGroupKind};

#[derive(Debug)]
pub struct LfGlobal {
    pub patches: Option<Patches>,
    pub splines: Option<Splines>,
    pub noise: Option<NoiseParameters>,
    pub lf_dequant: LfChannelDequantization,
    pub vardct: Option<LfGlobalVarDct>,
    pub gmodular: GlobalModular,
}

impl Bundle<(&Headers, &FrameHeader)> for LfGlobal {
    type Error = crate::Error;

    fn parse<R: std::io::Read>(bitstream: &mut Bitstream<R>, (image_header, header): (&Headers, &FrameHeader)) -> Result<Self> {
        let patches = header.flags.patches().then(|| {
            Patches::parse(bitstream, image_header)
        }).transpose()?;
        let splines = header.flags.splines().then(|| {
            Splines::parse(bitstream, ())
        }).transpose()?;
        let noise = header.flags.noise().then(|| {
            NoiseParameters::parse(bitstream, ())
        }).transpose()?;
        let lf_dequant = read_bits!(bitstream, Bundle(LfChannelDequantization))?;
        let vardct = (header.encoding == crate::header::Encoding::VarDct).then(|| {
            read_bits!(bitstream, Bundle(LfGlobalVarDct))
        }).transpose()?;
        let gmodular = read_bits!(bitstream, Bundle(GlobalModular), (image_header, header))?;

        Ok(Self {
            patches,
            splines,
            noise,
            lf_dequant,
            vardct,
            gmodular,
        })
    }
}

impl LfGlobal {
    pub(crate) fn apply_modular_inverse_transform(&mut self) {
        self.gmodular.modular.inverse_transform();
    }
}

define_bundle! {
    #[derive(Debug)]
    pub struct LfGlobalVarDct error(crate::Error) {
        pub quantizer: ty(Bundle(Quantizer)),
        pub hf_block_ctx: ty(Bundle(HfBlockContext)),
        pub lf_chan_corr: ty(Bundle(LfChannelCorrelation)),
    }
}


#[derive(Debug)]
pub struct GlobalModular {
    pub ma_config: Option<MaConfig>,
    pub modular: Modular,
}

impl GlobalModular {
    pub fn make_context(&self) -> Option<MaContext> {
        Some(self.ma_config.as_ref()?.make_context())
    }
}

impl Bundle<(&Headers, &FrameHeader)> for GlobalModular {
    type Error = crate::Error;

    fn parse<R: std::io::Read>(bitstream: &mut Bitstream<R>, (image_header, header): (&Headers, &FrameHeader)) -> Result<Self> {
        let ma_config = bitstream.read_bool()?
            .then(|| read_bits!(bitstream, Bundle(MaConfig)))
            .transpose()?;
        let mut shifts = Vec::new();
        if header.encoding == Encoding::Modular {
            if header.do_ycbcr {
                // Cb, Y, Cr
                shifts.push(ChannelShift::from_jpeg_upsampling(header.jpeg_upsampling, 0));
                shifts.push(ChannelShift::from_jpeg_upsampling(header.jpeg_upsampling, 1));
                shifts.push(ChannelShift::from_jpeg_upsampling(header.jpeg_upsampling, 2));
            } else {
                let shift = ChannelShift::from_shift(0);
                let is_single_channel = !image_header.metadata.xyb_encoded && image_header.metadata.colour_encoding.colour_space == ColourSpace::Grey;
                let channels = if is_single_channel { 1 } else { 3 };
                shifts.extend(std::iter::repeat(shift).take(channels));
            }
        }

        for (&ec_upsampling, ec_info) in header.ec_upsampling.iter().zip(image_header.metadata.ec_info.iter()) {
            let dim_shift = ec_info.dim_shift;
            let shift = ChannelShift::from_upsampling_factor_and_shift(ec_upsampling, dim_shift);
            shifts.push(shift);
        }

        let group_dim = header.group_dim();
        let modular_params = ModularParams::new(
            header.sample_width(),
            header.sample_height(),
            group_dim,
            image_header.metadata.bit_depth.bits_per_sample(),
            shifts,
            ma_config.as_ref(),
        );
        let mut modular = read_bits!(bitstream, Bundle(Modular), modular_params)?;
        modular.decode_image_gmodular(bitstream)?;

        Ok(Self {
            ma_config,
            modular,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct LfGroupParams<'a> {
    frame_header: &'a FrameHeader,
    gmodular: &'a GlobalModular,
    lf_group_idx: u32,
}

impl<'a> LfGroupParams<'a> {
    pub fn new(frame_header: &'a FrameHeader, lf_global: &'a LfGlobal, lf_group_idx: u32) -> Self {
        Self {
            frame_header,
            gmodular: &lf_global.gmodular,
            lf_group_idx,
        }
    }
}

#[derive(Debug)]
pub struct LfGroup {
    pub lf_coeff: Option<LfCoeff>,
    pub mlf_group: Modular,
    pub hf_meta: Option<HfMetadata>,
}

#[derive(Debug)]
pub struct LfCoeff {
    pub extra_precision: u8,
    pub lf_quant: Modular,
}

#[derive(Debug)]
pub struct HfMetadata {
    pub x_from_y: Grid<i32>,
    pub b_from_y: Grid<i32>,
    pub block_info: Grid<BlockInfo>,
    pub sharpness: Grid<i32>,
}

#[derive(Debug, Default, Clone, Copy)]
pub enum BlockInfo {
    #[default]
    Uninit,
    Occupied,
    Data {
        dct_select: TransformType,
        hf_mul: i32,
    },
}

impl Bundle<LfGroupParams<'_>> for LfGroup {
    type Error = crate::Error;

    fn parse<R: std::io::Read>(bitstream: &mut Bitstream<R>, params: LfGroupParams<'_>) -> Result<Self> {
        let LfGroupParams { frame_header, gmodular, lf_group_idx } = params;

        let lf_coeff = (frame_header.encoding == Encoding::VarDct && !frame_header.flags.use_lf_frame())
            .then(|| read_bits!(bitstream, Bundle(LfCoeff), params))
            .transpose()?;

        let mlf_group_params = gmodular.modular
            .make_subimage_params_lf_group(gmodular.ma_config.as_ref(), lf_group_idx);
        let mut mlf_group = read_bits!(bitstream, Bundle(Modular), mlf_group_params.clone())?;
        mlf_group.decode_image(bitstream, 1 + frame_header.num_lf_groups() + lf_group_idx)?;
        mlf_group.inverse_transform();

        let hf_meta = (frame_header.encoding == Encoding::VarDct)
            .then(|| read_bits!(bitstream, Bundle(HfMetadata), params))
            .transpose()?;

        Ok(Self { lf_coeff, mlf_group, hf_meta })
    }
}

impl Bundle<LfGroupParams<'_>> for LfCoeff {
    type Error = crate::Error;

    fn parse<R: std::io::Read>(bitstream: &mut Bitstream<R>, group_params: LfGroupParams<'_>) -> Result<Self> {
        let extra_precision = bitstream.read_bits(2)? as u8;

        let lf_group_idx = group_params.lf_group_idx;
        let (width, height) = group_params.frame_header.lf_group_size_for(lf_group_idx);
        let width = (width + 7) / 8;
        let height = (height + 7) / 8;
        let channel_shifts = (0..3)
            .map(|idx| ChannelShift::from_jpeg_upsampling(group_params.frame_header.jpeg_upsampling, idx))
            .collect();
        let lf_quant_params = ModularParams::new(
            width,
            height,
            group_params.frame_header.group_dim(),
            group_params.frame_header.bit_depth.bits_per_sample(),
            channel_shifts,
            group_params.gmodular.ma_config.as_ref(),
        );
        let mut lf_quant = read_bits!(bitstream, Bundle(Modular), lf_quant_params)?;
        lf_quant.decode_image(bitstream, 1 + lf_group_idx)?;
        lf_quant.inverse_transform();
        Ok(Self { extra_precision, lf_quant })
    }
}

impl Bundle<LfGroupParams<'_>> for HfMetadata {
    type Error = crate::Error;

    fn parse<R: std::io::Read>(bitstream: &mut Bitstream<R>, group_params: LfGroupParams<'_>) -> Result<Self> {
        let lf_group_idx = group_params.lf_group_idx;
        let (width, height) = group_params.frame_header.lf_group_size_for(lf_group_idx);
        let bw = (width + 7) / 8;
        let bh = (height + 7) / 8;
        let nb_blocks = 1 + bitstream.read_bits((bw * bh).next_power_of_two().trailing_zeros())?;

        let channels = vec![
            ModularChannelParams::new((width + 63) / 64, (height + 63) / 64, 128),
            ModularChannelParams::new((width + 63) / 64, (height + 63) / 64, 128),
            ModularChannelParams::new(nb_blocks, 2, 128),
            ModularChannelParams::new(bw, bh, 128),
        ];
        let params = ModularParams::with_channels(
            128,
            group_params.frame_header.bit_depth.bits_per_sample(),
            channels,
            group_params.gmodular.ma_config.as_ref(),
        );
        let mut modular = read_bits!(bitstream, Bundle(Modular), params)?;
        modular.decode_image(bitstream, 1 + 2 * group_params.frame_header.num_lf_groups() + lf_group_idx)?;
        modular.inverse_transform();

        let image = modular.into_image().into_channel_data();
        let mut image_iter = image.into_iter();
        let x_from_y = image_iter.next().unwrap();
        let b_from_y = image_iter.next().unwrap();
        let block_info_raw = image_iter.next().unwrap();
        let sharpness = image_iter.next().unwrap();

        let mut block_info = Grid::<BlockInfo>::new(bw, bh, (bw, bh));
        let mut x;
        let mut y = 0u32;
        let mut data_idx = 0u32;
        while y < bh {
            x = 0u32;

            while x < bw {
                if !block_info[(x, y)].is_occupied() {
                    let dct_select = block_info_raw[(data_idx, 0)];
                    let dct_select = TransformType::try_from(dct_select as u8)?;
                    let hf_mul = block_info_raw[(data_idx, 1)];
                    let (dw, dh) = dct_select.dct_select_size();

                    for dy in 0..dh {
                        for dx in 0..dw {
                            debug_assert!(!block_info[(x + dx, y + dy)].is_occupied());
                            block_info[(x + dx, y + dy)] = if dx == 0 && dy == 0 {
                                BlockInfo::Data {
                                    dct_select,
                                    hf_mul: hf_mul + 1,
                                }
                            } else {
                                BlockInfo::Occupied
                            };
                        }
                    }
                    data_idx += 1;
                    x += dw;
                } else {
                    x += 1;
                }
            }

            y += 1;
        }

        Ok(Self {
            x_from_y,
            b_from_y,
            block_info,
            sharpness,
        })
    }
}

impl BlockInfo {
    fn is_occupied(self) -> bool {
        !matches!(self, Self::Uninit)
    }
}

#[derive(Debug, Copy, Clone)]
pub struct HfGlobalParams<'a> {
    metadata: &'a ImageMetadata,
    frame_header: &'a FrameHeader,
    ma_config: Option<&'a MaConfig>,
    hf_block_ctx: &'a HfBlockContext,
}

impl<'a> HfGlobalParams<'a> {
    pub fn new(metadata: &'a ImageMetadata, frame_header: &'a FrameHeader, lf_global: &'a LfGlobal) -> Self {
        let Some(lf_vardct) = &lf_global.vardct else { panic!("VarDCT not initialized") };
        Self {
            metadata,
            frame_header,
            ma_config: lf_global.gmodular.ma_config.as_ref(),
            hf_block_ctx: &lf_vardct.hf_block_ctx,
        }
    }
}

#[derive(Debug)]
pub struct HfGlobal {
    dequant_matrices: jxl_vardct::DequantMatrixSet,
    num_hf_presets: u32,
    hf_passes: Vec<HfPass>,
}

impl Bundle<HfGlobalParams<'_>> for HfGlobal {
    type Error = crate::Error;

    fn parse<R: std::io::Read>(bitstream: &mut Bitstream<R>, params: HfGlobalParams<'_>) -> Result<Self> {
        let HfGlobalParams { metadata, frame_header, ma_config, hf_block_ctx } = params;
        let dequant_matrix_params = jxl_vardct::DequantMatrixSetParams::new(
            metadata.bit_depth.bits_per_sample(),
            1 + frame_header.num_lf_groups() * 3,
            ma_config,
        );
        let dequant_matrices = jxl_vardct::DequantMatrixSet::parse(bitstream, dequant_matrix_params)?;

        let num_groups = frame_header.num_groups();
        let num_hf_presets = bitstream.read_bits(num_groups.next_power_of_two().trailing_zeros())? + 1;

        let hf_pass_params = jxl_vardct::HfPassParams::new(hf_block_ctx, num_hf_presets);
        let hf_passes = std::iter::repeat_with(|| HfPass::parse(bitstream, hf_pass_params))
            .take(frame_header.passes.num_passes as usize)
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(Self {
            dequant_matrices,
            num_hf_presets,
            hf_passes,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PassGroupParams<'a> {
    frame_header: &'a FrameHeader,
    gmodular: &'a GlobalModular,
    lf_vardct: Option<&'a LfGlobalVarDct>,
    lf_group: &'a LfGroup,
    hf_global: Option<&'a HfGlobal>,
    pass_idx: u32,
    group_idx: u32,
    shift: Option<(i32, i32)>,
}

impl<'a> PassGroupParams<'a> {
    pub fn new(
        frame_header: &'a FrameHeader,
        lf_global: &'a LfGlobal,
        lf_group: &'a LfGroup,
        hf_global: Option<&'a HfGlobal>,
        pass_idx: u32,
        group_idx: u32,
        shift: Option<(i32, i32)>,
    ) -> Self {
        Self {
            frame_header,
            gmodular: &lf_global.gmodular,
            lf_vardct: lf_global.vardct.as_ref(),
            lf_group,
            hf_global,
            pass_idx,
            group_idx,
            shift,
        }
    }
}

#[derive(Debug)]
pub struct PassGroup {
    pub hf_coeff: Option<HfCoeff>,
    pub modular: Modular,
}

#[derive(Debug)]
struct HfCoeffParams<'a> {
    num_hf_presets: u32,
    hf_block_ctx: &'a HfBlockContext,
    block_info: Subgrid<'a, BlockInfo>,
    jpeg_upsampling: [u32; 3],
    lf_quant: Option<[Subgrid<'a, i32>; 3]>,
    hf_pass: &'a HfPass,
}

impl Bundle<PassGroupParams<'_>> for PassGroup {
    type Error = crate::Error;

    fn parse<R: std::io::Read>(bitstream: &mut Bitstream<R>, params: PassGroupParams<'_>) -> Result<Self> {
        let PassGroupParams {
            frame_header,
            gmodular,
            lf_vardct,
            lf_group,
            hf_global,
            pass_idx,
            group_idx,
            shift,
        } = params;

        let hf_coeff = lf_vardct
            .zip(lf_group.lf_coeff.as_ref().zip(lf_group.hf_meta.as_ref()))
            .zip(hf_global)
            .map(|((lf_vardct, (lf_coeff, hf_meta)), hf_global)| -> Result<HfCoeff> {
                let hf_pass = &hf_global.hf_passes[pass_idx as usize];

                let group_col = group_idx % frame_header.groups_per_row();
                let group_row = group_idx / frame_header.groups_per_row();
                let lf_col = group_col % 8;
                let lf_row = group_row % 8;
                let group_dim_blocks = frame_header.group_dim() / 8;

                let lf_quant_channels = lf_coeff.lf_quant.image().channel_data();
                let block_info = &hf_meta.block_info;

                let block_left = lf_col * group_dim_blocks;
                let block_top = lf_row * group_dim_blocks;
                let block_width = (block_info.width() - block_left).min(group_dim_blocks);
                let block_height = (block_info.height() - block_top).min(group_dim_blocks);

                let jpeg_upsampling = frame_header.jpeg_upsampling;
                let block_info = block_info.subgrid(block_left as i32, block_top as i32, block_width, block_height);
                let lf_quant: Option<[_; 3]> = (!frame_header.flags.use_lf_frame()).then(|| {
                    std::array::from_fn(|idx| {
                        let lf_quant = &lf_quant_channels[idx];
                        let shift = ChannelShift::from_jpeg_upsampling(jpeg_upsampling, idx);

                        let block_left = block_left >> shift.hshift();
                        let block_top = block_top >> shift.vshift();
                        let block_width = block_width >> shift.hshift();
                        let block_height = block_height >> shift.vshift();
                        lf_quant.subgrid(block_left as i32, block_top as i32, block_width, block_height)
                    })
                });

                let params = HfCoeffParams {
                    num_hf_presets: hf_global.num_hf_presets,
                    hf_block_ctx: &lf_vardct.hf_block_ctx,
                    block_info,
                    jpeg_upsampling,
                    lf_quant,
                    hf_pass,
                };
                HfCoeff::parse(bitstream, params)
            })
            .transpose()?;

        let modular = if let Some((minshift, maxshift)) = shift {
            let modular_params = gmodular.modular.make_subimage_params_pass_group(gmodular.ma_config.as_ref(), group_idx, minshift, maxshift);
            let mut modular = read_bits!(bitstream, Bundle(Modular), modular_params)?;
            modular.decode_image(bitstream, 1 + 3 * frame_header.num_lf_groups() + 17 + pass_idx * frame_header.num_groups() + group_idx)?;
            modular.inverse_transform();
            modular
        } else {
            Modular::empty()
        };

        Ok(Self {
            hf_coeff,
            modular,
        })
    }
}

#[derive(Debug)]
pub struct HfCoeff {
    data: HashMap<(u32, u32), CoeffData>,
}

#[derive(Debug)]
struct CoeffData {
    dct_select: TransformType,
    coeff: [Grid<i32>; 3],
}

impl Bundle<HfCoeffParams<'_>> for HfCoeff {
    type Error = crate::Error;

    fn parse<R: std::io::Read>(bitstream: &mut Bitstream<R>, params: HfCoeffParams<'_>) -> Result<Self> {
        const COEFF_FREQ_CONTEXT: [u32; 64] = [
            0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
            15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22,
            23, 23, 23, 23, 24, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 26,
            27, 27, 27, 27, 28, 28, 28, 28, 29, 29, 29, 29, 30, 30, 30, 30,
        ];
        const COEFF_NUM_NONZERO_CONTEXT: [u32; 64] = [
            0,     0,  31,  62,  62,  93,  93,  93,  93, 123, 123, 123, 123,
            152, 152, 152, 152, 152, 152, 152, 152, 180, 180, 180, 180, 180,
            180, 180, 180, 180, 180, 180, 180, 206, 206, 206, 206, 206, 206,
            206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206,
            206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206,
        ];

        let mut data = HashMap::new();

        let HfCoeffParams {
            num_hf_presets,
            hf_block_ctx,
            block_info,
            jpeg_upsampling,
            lf_quant,
            hf_pass,
        } = params;
        let mut dist = hf_pass.clone_decoder();

        let HfBlockContext {
            qf_thresholds,
            lf_thresholds,
            block_ctx_map,
            num_block_clusters,
        } = hf_block_ctx;
        let upsampling_shifts: [_; 3] = std::array::from_fn(|idx| ChannelShift::from_jpeg_upsampling(jpeg_upsampling, idx));

        let hfp_bits = num_hf_presets.next_power_of_two().trailing_zeros();
        let hfp = bitstream.read_bits(hfp_bits)?;
        let ctx_offset = 495 * *num_block_clusters * hfp;

        dist.begin(bitstream)?;

        let width = block_info.width();
        let height = block_info.height();
        let mut non_zeros_grid = [
            Grid::new(width, height, (width, height)),
            Grid::new(width, height, (width, height)),
            Grid::new(width, height, (width, height)),
        ];
        let predict_non_zeros = |grid: &Grid<u32>, x: u32, y: u32| {
            if x == 0 && y == 0 {
                32u32
            } else if x == 0 {
                grid[(x, y - 1)]
            } else if y == 0 {
                grid[(x - 1, y)]
            } else {
                (grid[(x - 1, y)] + grid[(x, y - 1)] + 1) >> 1
            }
        };

        for y in 0..height {
            for x in 0..width {
                let BlockInfo::Data { dct_select, hf_mul: qf } = block_info[(x, y)] else {
                    continue;
                };
                let (w8, h8) = dct_select.dct_select_size();
                let coeff_size = dct_select.dequant_matrix_size();
                let num_blocks = w8 * h8;
                let order_id = dct_select.order_id();
                let qdc: Option<[_; 3]> = lf_quant.as_ref().map(|lf_quant| {
                    std::array::from_fn(|idx| {
                        let shift = upsampling_shifts[idx];
                        let x = x >> shift.hshift();
                        let y = y >> shift.vshift();
                        lf_quant[idx][(x, y)]
                    })
                });

                let hf_idx = {
                    let mut idx = 0usize;
                    for &threshold in qf_thresholds {
                        if qf > threshold as i32 {
                            idx += 1;
                        }
                    }
                    idx
                };
                let lf_idx = if let Some(qdc) = qdc {
                    let mut idx = 0usize;
                    for c in [0, 2, 1] {
                        let lf_thresholds = &lf_thresholds[c];
                        idx *= lf_thresholds.len() + 1;

                        let q = qdc[c];
                        for &threshold in lf_thresholds {
                            if q > threshold {
                                idx += 1;
                            }
                        }
                    }
                    idx
                } else {
                    0
                };
                let lf_idx_mul = (lf_thresholds[0].len() + 1) * (lf_thresholds[1].len() + 1) * (lf_thresholds[2].len() + 1);

                let mut coeff = [
                    Grid::new(coeff_size.0, coeff_size.1, coeff_size),
                    Grid::new(coeff_size.0, coeff_size.1, coeff_size),
                    Grid::new(coeff_size.0, coeff_size.1, coeff_size),
                ];
                for c in [1, 0, 2] { // y, x, b
                    let shift = upsampling_shifts[c];
                    let sx = x >> shift.hshift();
                    let sy = y >> shift.vshift();
                    if sx << shift.hshift() != x || sy << shift.vshift() != y {
                        continue;
                    }

                    let ch_idx = [1, 0, 2][c] * 13 + order_id as usize;
                    let idx = (ch_idx * (qf_thresholds.len() + 1) + hf_idx) * lf_idx_mul + lf_idx;
                    let block_ctx = block_ctx_map[idx] as u32;
                    let non_zeros_ctx = {
                        let predicted = predict_non_zeros(&non_zeros_grid[c], x, y).min(64);
                        let idx = if predicted >= 8 {
                            4 + predicted / 2
                        } else {
                            predicted
                        };
                        block_ctx + idx * num_block_clusters
                    };

                    let mut non_zeros = dist.read_varint(bitstream, ctx_offset + non_zeros_ctx)?;
                    let non_zeros_val = (non_zeros + num_blocks - 1) / num_blocks;
                    let non_zeros_grid = &mut non_zeros_grid[c];
                    for dy in 0..h8 {
                        for dx in 0..w8 {
                            non_zeros_grid[(x + dx, y + dy)] = non_zeros_val;
                        }
                    }

                    let size = (w8 * 8) * (h8 * 8);
                    let coeff_grid = &mut coeff[c];
                    let mut prev_coeff = (non_zeros <= size / 16) as i32;
                    let order_it = hf_pass.order(order_id as usize, c);
                    for (idx, coeff_coord) in order_it.enumerate().skip(num_blocks as usize) {
                        if non_zeros == 0 {
                            break;
                        }

                        let idx = idx as u32;
                        let coeff_ctx = {
                            let prev = (prev_coeff != 0) as u32;
                            let non_zeros = (non_zeros + num_blocks - 1) / num_blocks;
                            let idx = idx / num_blocks;
                            (COEFF_NUM_NONZERO_CONTEXT[non_zeros as usize] + COEFF_FREQ_CONTEXT[idx as usize]) * 2 +
                                prev + block_ctx * 458 + 37 * num_block_clusters
                        };
                        let ucoeff = dist.read_varint(bitstream, ctx_offset + coeff_ctx)?;
                        let coeff = jxl_bitstream::unpack_signed(ucoeff);
                        let (x, y) = coeff_coord;
                        coeff_grid[(x as u32, y as u32)] = coeff;
                        prev_coeff = coeff;

                        if coeff != 0 {
                            non_zeros -= 1;
                        }
                    }
                }

                data.insert((x, y), CoeffData {
                    dct_select,
                    coeff,
                });
            }
        }

        dist.finalize()?;

        Ok(Self { data })
    }
}
