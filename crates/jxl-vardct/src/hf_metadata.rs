use jxl_bitstream::{Bitstream, Bundle};
use jxl_grid::SimpleGrid;
use jxl_modular::{MaConfig, Modular, ModularChannelParams, ModularParams};

use crate::{Result, TransformType};

/// Parameters for decoding `HfMetadata`.
#[derive(Debug)]
pub struct HfMetadataParams<'ma, 'pool> {
    pub num_lf_groups: u32,
    pub lf_group_idx: u32,
    pub lf_width: u32,
    pub lf_height: u32,
    pub jpeg_upsampling: [u32; 3],
    pub bits_per_sample: u32,
    pub global_ma_config: Option<&'ma MaConfig>,
    pub epf: Option<(f32, [f32; 8])>,
    pub quantizer_global_scale: u32,
    pub pool: &'pool jxl_threadpool::JxlThreadPool,
}

/// Data for decoding and rendering varblocks within an LF group.
#[derive(Debug)]
pub struct HfMetadata {
    /// Chroma-from-luma correlation grid for X channel.
    pub x_from_y: SimpleGrid<i32>,
    /// Chroma-from-luma correlation grid for B channel.
    pub b_from_y: SimpleGrid<i32>,
    /// Varblock information in an LF group.
    pub block_info: SimpleGrid<BlockInfo>,
    /// Sigma parameter grid for edge-preserving filter.
    pub epf_sigma: SimpleGrid<f32>,
}

/// Varblock grid information.
#[derive(Debug, Default, Clone, Copy)]
pub enum BlockInfo {
    /// The block is not initialized yet.
    #[default]
    Uninit,
    /// The block is occupied by a varblock.
    Occupied,
    /// The block is the top-left block of a varblock.
    Data {
        dct_select: TransformType,
        hf_mul: i32,
    },
}

impl Bundle<HfMetadataParams<'_, '_>> for HfMetadata {
    type Error = crate::Error;

    fn parse(bitstream: &mut Bitstream, params: HfMetadataParams<'_, '_>) -> Result<Self> {
        let HfMetadataParams {
            num_lf_groups,
            lf_group_idx,
            lf_width,
            lf_height,
            jpeg_upsampling,
            bits_per_sample,
            global_ma_config,
            epf,
            quantizer_global_scale,
            pool,
        } = params;

        let mut bw = ((lf_width + 7) / 8) as usize;
        let mut bh = ((lf_height + 7) / 8) as usize;

        let h_upsample = jpeg_upsampling.into_iter().any(|j| j == 1 || j == 2);
        let v_upsample = jpeg_upsampling.into_iter().any(|j| j == 1 || j == 3);
        if h_upsample {
            bw = (bw + 1) / 2 * 2;
        }
        if v_upsample {
            bh = (bh + 1) / 2 * 2;
        }

        let nb_blocks =
            1 + bitstream.read_bits((bw * bh).next_power_of_two().trailing_zeros() as usize)?;

        let channels = vec![
            ModularChannelParams::new((lf_width + 63) / 64, (lf_height + 63) / 64),
            ModularChannelParams::new((lf_width + 63) / 64, (lf_height + 63) / 64),
            ModularChannelParams::new(nb_blocks, 2),
            ModularChannelParams::new(bw as u32, bh as u32),
        ];
        let params = ModularParams::with_channels(0, bits_per_sample, channels, global_ma_config);
        let mut modular = Modular::parse(bitstream, params)?;
        let image = modular.image_mut().unwrap();
        let mut subimage = image.prepare_subimage()?;
        subimage.decode(bitstream, 1 + 2 * num_lf_groups + lf_group_idx, false)?;
        subimage.finish(pool);

        let image = modular.into_image().unwrap().into_image_channels();
        let mut image_iter = image.into_iter();
        let x_from_y = image_iter.next().unwrap();
        let b_from_y = image_iter.next().unwrap();
        let block_info_raw = image_iter.next().unwrap();
        let sharpness = image_iter.next().unwrap();

        let sharpness = sharpness.buf();

        let mut epf_sigma = SimpleGrid::new(bw, bh);
        let epf_sigma_buf = epf_sigma.buf_mut();
        let epf = epf.map(|(quant_mul, sharp_lut)| {
            (
                quant_mul * 65536.0 / quantizer_global_scale as f32,
                sharp_lut,
            )
        });

        let mut block_info = SimpleGrid::<BlockInfo>::new(bw, bh);
        let mut x;
        let mut y = 0usize;
        let mut data_idx = 0usize;
        while y < bh {
            x = 0usize;

            while x < bw {
                if !block_info.get(x, y).unwrap().is_occupied() {
                    let Some(&dct_select) = block_info_raw.get(data_idx, 0) else {
                        tracing::error!(lf_group_idx, x, y, "BlockInfo doesn't fill LF group");
                        return Err(jxl_bitstream::Error::ValidationFailed(
                            "BlockInfo doesn't fill LF group",
                        )
                        .into());
                    };
                    let dct_select = TransformType::try_from(dct_select as u8)?;
                    let mul = *block_info_raw.get(data_idx, 1).unwrap();
                    let hf_mul = mul + 1;
                    let (dw, dh) = dct_select.dct_select_size();

                    let epf =
                        epf.map(|(quant_mul, sharp_lut)| (quant_mul / hf_mul as f32, sharp_lut));
                    for dy in 0..dh as usize {
                        for dx in 0..dw as usize {
                            if let Some(info) = block_info.get(x + dx, y + dy) {
                                if info.is_occupied() {
                                    tracing::error!(
                                        lf_group_idx,
                                        base_x = x,
                                        base_y = y,
                                        dct_select = format_args!("{:?}", dct_select),
                                        x = x + dx,
                                        y = y + dy,
                                        "Varblocks overlap",
                                    );
                                    return Err(jxl_bitstream::Error::ValidationFailed(
                                        "Varblocks overlap",
                                    )
                                    .into());
                                }
                            } else {
                                tracing::error!(
                                    lf_group_idx,
                                    base_x = x,
                                    base_y = y,
                                    dct_select = format_args!("{:?}", dct_select),
                                    "Varblock doesn't fit in an LF group",
                                );
                                return Err(jxl_bitstream::Error::ValidationFailed(
                                    "Varblock doesn't fit in an LF group",
                                )
                                .into());
                            };

                            *block_info.get_mut(x + dx, y + dy).unwrap() = if dx == 0 && dy == 0 {
                                BlockInfo::Data { dct_select, hf_mul }
                            } else {
                                BlockInfo::Occupied
                            };

                            if let Some((sigma, sharp_lut)) = epf {
                                let sharpness = sharpness[(y + dy) * bw + (x + dx)];
                                if !(0..8).contains(&sharpness) {
                                    return Err(jxl_bitstream::Error::ValidationFailed(
                                        "Invalid EPF sharpness value",
                                    )
                                    .into());
                                }
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
