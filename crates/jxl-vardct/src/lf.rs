use jxl_bitstream::{define_bundle, read_bits, Bitstream, Bundle};
use jxl_modular::{MaConfig, Modular, ChannelShift, ModularParams};

use crate::Result;

define_bundle! {
    /// Dequantization information for each channel.
    #[derive(Debug)]
    pub struct LfChannelDequantization error(crate::Error) {
        all_default: ty(Bool) default(true),
        pub m_x_lf: ty(F16) cond(!all_default) default(1.0 / 32.0),
        pub m_y_lf: ty(F16) cond(!all_default) default(1.0 / 4.0),
        pub m_b_lf: ty(F16) cond(!all_default) default(1.0 / 2.0),
    }

    /// Global quantizer multipliers.
    #[derive(Debug)]
    pub struct Quantizer error(crate::Error) {
        pub global_scale: ty(U32(1 + u(11), 2049 + u(11), 4097 + u(12), 8193 + u(16))),
        pub quant_lf: ty(U32(16, 1 + u(5), 1 + u(8), 1 + u(16))),
    }

    /// Channel correlation data, used by chroma-from-luma procedure.
    #[derive(Debug)]
    pub struct LfChannelCorrelation error(crate::Error) {
        all_default: ty(Bool) default(true),
        pub colour_factor: ty(U32(84, 256, 2 + u(8), 258 + u(16))) cond(!all_default) default(84),
        pub base_correlation_x: ty(F16) cond(!all_default) default(0.0),
        pub base_correlation_b: ty(F16) cond(!all_default) default(1.0),
        pub x_factor_lf: ty(u(8)) cond(!all_default) default(128),
        pub b_factor_lf: ty(u(8)) cond(!all_default) default(128),
    }
}

impl LfChannelDequantization {
    #[inline]
    pub fn m_x_lf_unscaled(&self) -> f32 {
        self.m_x_lf / 128.0
    }

    #[inline]
    pub fn m_y_lf_unscaled(&self) -> f32 {
        self.m_y_lf / 128.0
    }

    #[inline]
    pub fn m_b_lf_unscaled(&self) -> f32 {
        self.m_b_lf / 128.0
    }
}

/// Context information for the entropy decoder of HF coefficients.
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

            if bsize > 64 {
                tracing::warn!(bsize, "bsize > 64");
            }

            let (num_clusters, ctx_map) = jxl_coding::read_clusters(bitstream, bsize * 39)?;
            if num_clusters > 16 {
                tracing::warn!(num_clusters, "num_clusters > 16");
            }

            (num_clusters, ctx_map)
        };

        Ok(Self {
            qf_thresholds,
            lf_thresholds,
            block_ctx_map,
            num_block_clusters,
        })
    }
}

/// Paramters for decoding `LfCoeff`.
#[derive(Debug)]
pub struct LfCoeffParams<'ma> {
    pub lf_group_idx: u32,
    pub lf_width: u32,
    pub lf_height: u32,
    pub jpeg_upsampling: [u32; 3],
    pub bits_per_sample: u32,
    pub global_ma_config: Option<&'ma MaConfig>,
}

/// Quantized LF image.
#[derive(Debug)]
pub struct LfCoeff {
    pub extra_precision: u8,
    pub lf_quant: Modular,
}

impl Bundle<LfCoeffParams<'_>> for LfCoeff {
    type Error = crate::Error;

    fn parse<R: std::io::Read>(bitstream: &mut Bitstream<R>, params: LfCoeffParams<'_>) -> Result<Self> {
        let LfCoeffParams {
            lf_group_idx,
            lf_width,
            lf_height,
            jpeg_upsampling,
            bits_per_sample,
            global_ma_config,
        } = params;

        let extra_precision = bitstream.read_bits(2)? as u8;

        let width = (lf_width + 7) / 8;
        let height = (lf_height + 7) / 8;
        let channel_shifts = [1, 0, 2]
            .into_iter()
            .map(|idx| ChannelShift::from_jpeg_upsampling(jpeg_upsampling, idx))
            .collect();
        let lf_quant_params = ModularParams::new(
            width,
            height,
            0,
            bits_per_sample,
            channel_shifts,
            global_ma_config,
        );
        let mut lf_quant = Modular::parse(bitstream, lf_quant_params)?;
        lf_quant.decode_image(bitstream, 1 + lf_group_idx)?;
        lf_quant.inverse_transform();
        Ok(Self { extra_precision, lf_quant })
    }
}
