use jxl_bitstream::{define_bundle, read_bits, header::{Headers, ColourSpace}, Bitstream, Bundle};

use crate::{
    FrameHeader,
    Result,
    header::Encoding,
    encoding::{modular::ChannelShift, Modular, ModularParams},
};

mod noise;
mod patch;
mod spline;
mod toc;
pub use toc::{Toc, TocGroup, TocGroupKind};

#[derive(Debug)]
pub struct LfGlobal {
    patches: Option<patch::Patches>,
    splines: Option<spline::Splines>,
    noise: Option<noise::NoiseParameters>,
    lf_dequant: LfChannelDequantization,
    vardct: Option<LfGlobalVarDct>,
    gmodular: GlobalModular,
}

impl Bundle<(&Headers, &FrameHeader)> for LfGlobal {
    type Error = crate::Error;

    fn parse<R: std::io::Read>(bitstream: &mut Bitstream<R>, (image_header, header): (&Headers, &FrameHeader)) -> Result<Self> {
        let patches = header.flags.patches().then(|| {
            todo!()
        });
        let splines = header.flags.splines().then(|| {
            todo!()
        });
        let noise = header.flags.noise().then(|| {
            todo!()
        });
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

define_bundle! {
    #[derive(Debug)]
    struct LfChannelDequantization error(crate::Error) {
        all_default: ty(Bool) default(true),
        m_x_lf: ty(F16) cond(!all_default) default(1.0 / 32.0),
        m_y_lf: ty(F16) cond(!all_default) default(1.0 / 4.0),
        m_b_lf: ty(F16) cond(!all_default) default(1.0 / 2.0),
    }
}

define_bundle! {
    #[derive(Debug)]
    struct LfGlobalVarDct error(crate::Error) {
        quantizer: ty(Bundle(Quantizer)),
        hf_block_ctx: ty(Bundle(HfBlockContext)),
        lf_chan_corr: ty(Bundle(LfChannelCorrelation)),
    }

    #[derive(Debug)]
    struct Quantizer error(crate::Error) {
        global_scale: ty(U32(1 + u(11), 2049 + u(11), 4097 + u(12), 8193 + u(16))),
        quant_lf: ty(U32(16, 1 + u(5), 1 + u(8), 1 + u(16))),
    }

    #[derive(Debug)]
    struct LfChannelCorrelation error(crate::Error) {
        all_default: ty(Bool) default(true),
        colour_factor: ty(U32(84,256, 2 + u(8), 258 + u(16))) cond(!all_default) default(84),
        base_correlation_x: ty(F16) cond(!all_default) default(0.0),
        base_correlation_b: ty(F16) cond(!all_default) default(1.0),
        x_factor_lf: ty(u(8)) cond(!all_default) default(128),
        b_factor_lf: ty(u(8)) cond(!all_default) default(128),
    }
}

#[derive(Debug, Default)]
struct HfBlockContext {
    qf_thresholds: Vec<u32>,
    lf_thresholds: [Vec<i32>; 3],
    block_ctx_map: Vec<u8>,
    num_block_clusters: u32,
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
    ma_config: Option<crate::encoding::modular::MaConfig>,
    modular: Modular,
}

impl GlobalModular {
    pub fn make_context(&self) -> Option<crate::encoding::modular::MaContext> {
        Some(self.ma_config.as_ref()?.make_context())
    }
}

impl Bundle<(&Headers, &FrameHeader)> for GlobalModular {
    type Error = crate::Error;

    fn parse<R: std::io::Read>(bitstream: &mut Bitstream<R>, (image_header, header): (&Headers, &FrameHeader)) -> Result<Self> {
        let ma_config = bitstream.read_bool()?
            .then(|| read_bits!(bitstream, Bundle(crate::encoding::modular::MaConfig)))
            .transpose()?;
        let mut shifts = Vec::new();
        if header.encoding == Encoding::Modular {
            if header.do_ycbcr {
                shifts.extend(
                    header.jpeg_upsampling
                        .iter()
                        .copied()
                        .map(ChannelShift::from_jpeg_upsampling)
                );
            } else {
                let shift = ChannelShift::from_upsampling_factor(header.upsampling);
                let is_single_channel = !image_header.metadata.xyb_encoded && image_header.metadata.colour_encoding.colour_space == ColourSpace::Grey;
                let channels = if is_single_channel { 3 } else { 1 };
                shifts.extend(std::iter::repeat(shift).take(channels));
            }
        }

        for (&ec_upsampling, ec_info) in header.ec_upsampling.iter().zip(image_header.metadata.ec_info.iter()) {
            let dim_shift = ec_info.dim_shift;
            let shift = ChannelShift::from_upsampling_factor_and_shift(ec_upsampling, dim_shift);
            shifts.push(shift);
        }

        let modular_params = ModularParams::new(header.width, header.height, shifts, ma_config.as_ref());
        let modular = read_bits!(bitstream, Bundle(Modular), modular_params)?;

        Ok(Self {
            ma_config,
            modular,
        })
    }
}

#[derive(Debug)]
pub struct LfGroup {
}

#[derive(Debug)]
pub struct HfGlobal {
}

#[derive(Debug)]
pub struct PassGroup {
}
