use jxl_bitstream::{read_bits, Bitstream, Bundle};
use jxl_grid::{Grid, SimpleGrid};
use jxl_modular::{ChannelShift, ModularChannelParams, Modular, ModularParams};
use jxl_vardct::{TransformType, Quantizer};

use crate::{
    FrameHeader,
    GlobalModular,
    LfGlobal,
    Result,
    filter::EdgePreservingFilter,
    header::Encoding,
};

#[derive(Debug, Clone, Copy)]
pub struct LfGroupParams<'a> {
    frame_header: &'a FrameHeader,
    quantizer: Option<&'a Quantizer>,
    gmodular: &'a GlobalModular,
    lf_group_idx: u32,
}

impl<'a> LfGroupParams<'a> {
    pub fn new(frame_header: &'a FrameHeader, lf_global: &'a LfGlobal, lf_group_idx: u32) -> Self {
        Self {
            frame_header,
            quantizer: lf_global.vardct.as_ref().map(|vardct| &vardct.quantizer),
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
    pub x_from_y: SimpleGrid<i32>,
    pub b_from_y: SimpleGrid<i32>,
    pub block_info: Grid<BlockInfo>,
    pub epf_sigma: SimpleGrid<f32>,
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
        let LfGroupParams { frame_header, gmodular, lf_group_idx, .. } = params;

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
        let channel_shifts = [1, 0, 2]
            .into_iter()
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
        let mut bw = ((width + 7) / 8) as usize;
        let mut bh = ((height + 7) / 8) as usize;
        let upsample = group_params.frame_header.need_jpeg_upscale();
        if upsample.0 {
            bw = (bw + 1) / 2 * 2;
        }
        if upsample.1 {
            bh = (bh + 1) / 2 * 2;
        }
        let nb_blocks = 1 + bitstream.read_bits((bw * bh).next_power_of_two().trailing_zeros())?;

        let channels = vec![
            ModularChannelParams::new((width + 63) / 64, (height + 63) / 64),
            ModularChannelParams::new((width + 63) / 64, (height + 63) / 64),
            ModularChannelParams::new(nb_blocks, 2),
            ModularChannelParams::new(bw as u32, bh as u32),
        ];
        let params = ModularParams::with_channels(
            0,
            group_params.frame_header.bit_depth.bits_per_sample(),
            channels,
            group_params.gmodular.ma_config.as_ref(),
        );
        let mut modular = read_bits!(bitstream, Bundle(Modular), params)?;
        modular.decode_image(bitstream, 1 + 2 * group_params.frame_header.num_lf_groups() + lf_group_idx)?;
        modular.inverse_transform();

        let image = modular.into_image().into_channel_data();
        let mut image_iter = image.into_iter().map(|g| g.into_simple().unwrap());
        let x_from_y = image_iter.next().unwrap();
        let b_from_y = image_iter.next().unwrap();
        let block_info_raw = image_iter.next().unwrap();
        let sharpness = image_iter.next().unwrap();

        let sharpness = sharpness.buf();

        let mut epf_sigma = SimpleGrid::new(bw, bh);
        let epf_sigma_buf = epf_sigma.buf_mut();
        let epf = if let EdgePreservingFilter::Enabled { sigma, sharp_lut, .. } = &group_params.frame_header.restoration_filter.epf {
            let quantizer = group_params.quantizer.unwrap();
            Some((sigma.quant_mul * 65536.0 / quantizer.global_scale as f32, sharp_lut))
        } else {
            None
        };

        let mut block_info = Grid::<BlockInfo>::new_usize(bw, bh, 0, 0);
        let mut x;
        let mut y = 0usize;
        let mut data_idx = 0usize;
        while y < bh {
            x = 0usize;

            while x < bw {
                if !block_info.get(x, y).unwrap().is_occupied() {
                    let dct_select = *block_info_raw.get(data_idx, 0).unwrap();
                    let dct_select = TransformType::try_from(dct_select as u8)?;
                    let mul = *block_info_raw.get(data_idx, 1).unwrap();
                    let hf_mul = mul + 1;
                    let (dw, dh) = dct_select.dct_select_size();

                    let epf = epf.map(|(quant_mul, sharp_lut)| (quant_mul / hf_mul as f32, sharp_lut));
                    for dy in 0..dh as usize {
                        for dx in 0..dw as usize {
                            debug_assert!(!block_info.get(x + dx, y + dy).unwrap().is_occupied());
                            block_info.set(x + dx, y + dy, if dx == 0 && dy == 0 {
                                BlockInfo::Data {
                                    dct_select,
                                    hf_mul,
                                }
                            } else {
                                BlockInfo::Occupied
                            });

                            if let Some((sigma, sharp_lut)) = epf {
                                let sharpness = sharpness[(y + dy) * bw + (x + dx)];
                                let sigma = sigma * sharp_lut[sharpness as usize];
                                epf_sigma_buf[(y + dy) * bw + (x + dx)] = sigma;
                            }
                        }
                    }
                    data_idx += 1;
                    x += dw as usize;
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
            epf_sigma,
        })
    }
}

impl BlockInfo {
    fn is_occupied(self) -> bool {
        !matches!(self, Self::Uninit)
    }
}
