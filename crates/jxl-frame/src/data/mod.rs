use jxl_bitstream::{define_bundle, read_bits, header::{Headers, ColourSpace}, Bitstream, Bundle};
use jxl_grid::Grid;
use jxl_modular::{ChannelShift, ModularChannelParams, Modular, ModularParams, MaConfig, MaContext};

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
        dbg!(&patches);
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
    pub struct LfChannelDequantization error(crate::Error) {
        all_default: ty(Bool) default(true),
        pub m_x_lf: ty(F16) cond(!all_default) default(1.0 / 32.0),
        pub m_y_lf: ty(F16) cond(!all_default) default(1.0 / 4.0),
        pub m_b_lf: ty(F16) cond(!all_default) default(1.0 / 2.0),
    }
}

define_bundle! {
    #[derive(Debug)]
    pub struct LfGlobalVarDct error(crate::Error) {
        pub quantizer: ty(Bundle(Quantizer)),
        pub hf_block_ctx: ty(Bundle(HfBlockContext)),
        pub lf_chan_corr: ty(Bundle(LfChannelCorrelation)),
    }

    #[derive(Debug)]
    pub struct Quantizer error(crate::Error) {
        pub global_scale: ty(U32(1 + u(11), 2049 + u(11), 4097 + u(12), 8193 + u(16))),
        pub quant_lf: ty(U32(16, 1 + u(5), 1 + u(8), 1 + u(16))),
    }

    #[derive(Debug)]
    pub struct LfChannelCorrelation error(crate::Error) {
        all_default: ty(Bool) default(true),
        pub colour_factor: ty(U32(84,256, 2 + u(8), 258 + u(16))) cond(!all_default) default(84),
        pub base_correlation_x: ty(F16) cond(!all_default) default(0.0),
        pub base_correlation_b: ty(F16) cond(!all_default) default(1.0),
        pub x_factor_lf: ty(u(8)) cond(!all_default) default(128),
        pub b_factor_lf: ty(u(8)) cond(!all_default) default(128),
    }
}

#[derive(Debug, Default)]
pub struct HfBlockContext {
    pub qf_thresholds: Vec<u32>,
    pub lf_thresholds: [Vec<i32>; 3],
    pub block_ctx_map: Vec<u8>,
    pub num_block_clusters: u32,
}

impl<Ctx> Bundle<Ctx> for HfBlockContext {
    type Error = crate::Error;

    fn parse<R: std::io::Read>(bitstream: &mut Bitstream<R>, _: Ctx) -> crate::Result<Self> {
        let mut qf_thresholds = Vec::new();
        let mut lf_thresholds = [Vec::new(), Vec::new(), Vec::new()];
        let (num_block_clusters, block_ctx_map) = if bitstream.read_bool()? {
            (15, vec![
                0, 1, 2, 2, 3, 3, 4, 5, 6, 6, 6, 6, 6,
                7, 8, 9, 9, 10, 11, 12, 13, 14, 14, 14, 14, 14,
                7, 8, 9, 9, 10, 11, 12, 13, 14, 14, 14, 14, 14,
            ])
        } else {
            let mut bsize = 1;
            for thr in &mut lf_thresholds {
                let num_lf_thresholds = bitstream.read_bits(4)?;
                bsize *= num_lf_thresholds + 1;
                for _ in 0..num_lf_thresholds {
                    let t = read_bits!(
                        bitstream,
                        U32(u(4), 16 + u(8), 272 + u(16), 65808 + u(32)); UnpackSigned
                    )?;
                    thr.push(t);
                }
            }
            let num_qf_thresholds = bitstream.read_bits(4)?;
            bsize *= num_qf_thresholds + 1;
            for _ in 0..num_qf_thresholds {
                let t = read_bits!(bitstream, U32(u(2), 4 + u(3), 12 + u(5), 44 + u(8)))?;
                qf_thresholds.push(1 + t);
            }

            jxl_coding::read_clusters(bitstream, bsize * 39)?
        };

        Ok(Self {
            qf_thresholds,
            lf_thresholds,
            block_ctx_map,
            num_block_clusters,
        })
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
                shifts.push(ChannelShift::from_jpeg_upsampling(header.jpeg_upsampling[1]));
                shifts.push(ChannelShift::from_jpeg_upsampling(header.jpeg_upsampling[0]));
                shifts.push(ChannelShift::from_jpeg_upsampling(header.jpeg_upsampling[2]));
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
        dct_select: u8,
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
        let channel_shifts = group_params.frame_header.jpeg_upsampling
            .into_iter()
            .map(ChannelShift::from_jpeg_upsampling)
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

        let block_info = Grid::<BlockInfo>::new(bw, bh, (bw, bh));
        let mut x = 0i32;
        let mut y = 0i32;
        let mut data_idx = 0i32;
        while y < bh as i32 {
            x = 0i32;

            while x < bw as i32 {
                if !block_info[(x, y)].is_occupied() {
                    let dct_select = block_info_raw[(data_idx, 0)];
                    let hf_mul = block_info_raw[(data_idx, 1)];
                    // TODO: parse DctSelect
                    data_idx += 1;
                }

                x += 1;
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

#[derive(Debug)]
pub struct HfGlobal {
}

#[derive(Debug, Clone, Copy)]
pub struct PassGroupParams<'a> {
    frame_header: &'a FrameHeader,
    gmodular: &'a GlobalModular,
    hf_global: Option<&'a HfGlobal>,
    pass_idx: u32,
    group_idx: u32,
    shift: Option<(i32, i32)>,
}

impl<'a> PassGroupParams<'a> {
    pub fn new(
        frame_header: &'a FrameHeader,
        lf_global: &'a LfGlobal,
        hf_global: Option<&'a HfGlobal>,
        pass_idx: u32,
        group_idx: u32,
        shift: Option<(i32, i32)>,
    ) -> Self {
        Self {
            frame_header,
            gmodular: &lf_global.gmodular,
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
pub struct HfCoeff {
}

impl Bundle<PassGroupParams<'_>> for PassGroup {
    type Error = crate::Error;

    fn parse<R: std::io::Read>(bitstream: &mut Bitstream<R>, params: PassGroupParams<'_>) -> Result<Self> {
        let PassGroupParams { frame_header, gmodular, hf_global, pass_idx, group_idx, shift } = params;

        let hf_coeff = (frame_header.encoding == Encoding::VarDct)
            .then(|| -> Result<HfCoeff> { todo!() })
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
