use jxl_bitstream::{Bitstream, Bundle, BundleDefault};
use jxl_modular::{Modular, ModularParams};

use crate::{Result, TransformType};

#[derive(Debug)]
struct DequantMatrixParams {
    dct_select: TransformType,
    encoding: DequantMatrixParamsEncoding,
}

#[derive(Debug)]
enum DequantMatrixParamsEncoding {
    Hornuss([[f32; 3]; 3]),
    Dct2([[f32; 6]; 3]),
    Dct4 {
        params: [[f32; 2]; 3],
        dct_params: [Vec<f32>; 3],
    },
    Dct4x8 {
        params: [[f32; 1]; 3],
        dct_params: [Vec<f32>; 3],
    },
    Afv {
        params: [[f32; 9]; 3],
        dct_params: [Vec<f32>; 3],
        dct4x4_params: [Vec<f32>; 3],
    },
    Dct([Vec<f32>; 3]),
    Raw {
        denominator: f32,
        params: Modular,
    },
}

impl DequantMatrixParamsEncoding {
    const SEQ_A: [f32; 7] = [-1.025, -0.78, -0.65012, -0.19041574, -0.20819396, -0.421064, -0.32733846];
    const SEQ_B: [f32; 7] = [-0.30419582, 0.36330363, -0.3566038, -0.34430745, -0.33699593, -0.30180866, -0.27321684];
    const SEQ_C: [f32; 7] = [-1.2, -1.2, -0.8, -0.7, -0.7, -0.4, -0.5];
    const DCT4X8_PARAMS: [[f32; 4]; 3] = [
        [2198.0505, -0.96269625, -0.7619425, -0.65511405],
        [764.36554, -0.926302, -0.967523, -0.2784529],
        [527.10754, -1.4594386, -1.4500821, -1.5843723],
    ];
    const DCT4_PARAMS: [[f32; 4]; 3] = [
        [2200.0, 0.0, 0.0, 0.0],
        [392.0, 0.0, 0.0, 0.0],
        [112.0, -0.25, -0.25, -0.5],
    ];

    fn make_dct_param_common_seq(a: f32, b: f32, c: f32) -> Self {
        Self::Dct([
            { let mut x = vec![a]; x.extend_from_slice(&Self::SEQ_A); x },
            { let mut x = vec![b]; x.extend_from_slice(&Self::SEQ_B); x },
            { let mut x = vec![c]; x.extend_from_slice(&Self::SEQ_C); x },
        ])
    }

    fn default_with(dct_select: TransformType) -> Self {
        use TransformType::*;

        match dct_select {
            Dct8 => Self::Dct([
                vec![3150.0, 0.0, -0.4, -0.4, -0.4, -2.0],
                vec![560.0, 0.0, -0.3, -0.3, -0.3, -0.3],
                vec![512.0, -2.0, -1.0, 0.0, -1.0, -2.0],
            ]),
            Hornuss => Self::Hornuss([
                [280.0, 3160.0, 3160.0],
                [60.0, 864.0, 864.0],
                [18.0, 200.0, 200.0],
            ]),
            Dct2 => Self::Dct2([
                [3840.0, 2560.0, 1280.0, 640.0, 480.0, 300.0],
                [960.0, 640.0, 320.0, 180.0, 140.0, 120.0],
                [640.0, 320.0, 128.0, 64.0, 32.0, 16.0],
            ]),
            Dct4 => Self::Dct4 {
                params: [[1.0; 2]; 3],
                dct_params: Self::DCT4_PARAMS.map(|v| v.to_vec()),
            },
            Dct16 => Self::Dct([
                vec![8996.873, -1.3000778, -0.4942453, -0.43909377, -0.6350102, -0.9017726, -1.6162099],
                vec![3191.4836, -0.67424583, -0.80745816, -0.4492584, -0.3586544, -0.3132239, -0.37615025],
                vec![1157.504, -2.0531423, -1.4, -0.5068713, -0.4270873, -1.4856834, -4.920914],
            ]),
            Dct32 => Self::Dct([
                vec![15718.408, -1.025, -0.98, -0.9012, -0.4, -0.48819396, -0.421064, -0.27],
                vec![7305.7637, -0.8041958, -0.76330364, -0.5566038, -0.49785304, -0.43699592, -0.40180868, -0.27321684],
                vec![3803.5317, -3.0607336, -2.041327, -2.023565, -0.54953897, -0.4, -0.4, -0.3],
            ]),
            Dct8x16 | Dct16x8 => Self::Dct([
                vec![7240.7734, -0.7, -0.7, -0.2, -0.2, -0.2, -0.5],
                vec![1448.1547, -0.5, -0.5, -0.5, -0.2, -0.2, -0.2],
                vec![506.85413, -1.4, -0.2, -0.5, -0.5, -1.5, -3.6],
            ]),
            Dct8x32 | Dct32x8 => Self::Dct([
                vec![16283.249, -1.7812846, -1.6309059, -1.0382179, -0.85, -0.7, -0.9, -1.2360638],
                vec![5089.1577, -0.3200494, -0.3536285, -0.3034, -0.61, -0.5, -0.5, -0.6],
                vec![3397.7761, -0.32132736, -0.3450762, -0.7034, -0.9, -1.0, -1.0, -1.1754606],
            ]),
            Dct16x32 | Dct32x16 => Self::Dct([
                vec![13844.971, -0.971138, -0.658, -0.42026, -0.22712, -0.2206, -0.226, -0.6],
                vec![4798.964, -0.6112531, -0.8377079, -0.7901486, -0.26927274, -0.38272768, -0.22924222, -0.20719099],
                vec![1807.2369, -1.2, -1.2, -0.7, -0.7, -0.7, -0.4, -0.5],
            ]),
            Dct4x8 | Dct8x4 => Self::Dct4x8 {
                params: [[1.0]; 3],
                dct_params: Self::DCT4X8_PARAMS.map(|v| v.to_vec()),
            },
            Afv0 | Afv1 | Afv2 | Afv3 => Self::Afv {
                params: [
                    [3072.0, 3072.0, 256.0, 256.0, 256.0, 414.0, 0.0, 0.0, 0.0],
                    [1024.0, 1024.0, 50.0, 50.0, 50.0, 58.0, 0.0, 0.0, 0.0],
                    [384.0, 384.0, 12.0, 12.0, 12.0, 22.0, -0.25, -0.25, -0.25],
                ],
                dct_params: Self::DCT4X8_PARAMS.map(|v| v.to_vec()),
                dct4x4_params: Self::DCT4_PARAMS.map(|v| v.to_vec()),
            },
            Dct64 => Self::make_dct_param_common_seq(23966.166, 8380.191, 4493.024),
            Dct32x64 | Dct64x32 => Self::make_dct_param_common_seq(15358.898, 5597.3604, 2919.9617),
            Dct128 => Self::make_dct_param_common_seq(47932.332, 16760.383, 8986.048),
            Dct64x128 | Dct128x64 => Self::make_dct_param_common_seq(30717.797, 11194.721, 5839.9233),
            Dct256 => Self::make_dct_param_common_seq(95864.664, 33520.766, 17972.096),
            Dct128x256 | Dct256x128 => Self::make_dct_param_common_seq(61435.594, 24209.441, 12979.847),
        }
    }
}

impl DequantMatrixParams {
    fn default_with(dct_select: TransformType) -> Self {
        Self {
            dct_select,
            encoding: DequantMatrixParamsEncoding::default_with(dct_select),
        }
    }

    fn into_matrix(self) -> [Vec<Vec<f32>>; 3] {
        use DequantMatrixParamsEncoding::*;

        fn interpolate(pos: f32, max: f32, bands: &[f32]) -> f32 {
            let len = bands.len();
            assert!(len > 0);
            assert!(pos >= 0.0);
            assert!(max > 0.0);

            if let &[val] = bands {
                return val;
            }

            let scaled_pos = pos * (len - 1) as f32 / max;
            let scaled_index = scaled_pos as usize; // scaled_pos >= 0.0
            let frac_index = scaled_pos - scaled_index as f32;

            let a = bands[scaled_index];
            let b = bands[scaled_index + 1];
            a * (b / a).powf(frac_index)
        }

        fn mult(x: f32) -> f32 {
            if x > 0.0 {
                1.0 + x
            } else {
                1.0 / (1.0 - x)
            }
        }

        fn dct_quant_weights(params: &[f32], width: u32, height: u32) -> Vec<Vec<f32>> {
            let mut bands = Vec::with_capacity(params.len());
            let mut last_band = params[0];
            bands.push(last_band);
            for &val in &params[1..] {
                let band = last_band * mult(val);
                // TODO: test band > 0.0
                bands.push(band);
                last_band = band;
            }

            let mut ret = Vec::with_capacity(height as usize);
            for y in 0..height {
                let mut row = Vec::with_capacity(width as usize);
                for x in 0..width {
                    let dx = x as f32 / (width - 1) as f32;
                    let dy = y as f32 / (height - 1) as f32;
                    let distance = (dx * dx + dy * dy).sqrt();
                    let weight = interpolate(distance, std::f32::consts::SQRT_2 + 1e-6, &bands);
                    row.push(weight);
                }
                ret.push(row);
            }

            ret
        }

        let dct_select = self.dct_select;
        let need_recip = !matches!(self.encoding, Raw { .. });
        let mut weights = match self.encoding {
            Dct(dct_params) => {
                let (width, height) = dct_select.dequant_matrix_size();
                dct_params.map(|params| dct_quant_weights(&params, width, height))
            },
            Hornuss(params) => {
                params.map(|params| {
                    let mut ret = vec![vec![params[0]; 8]; 8];
                    ret[0][0] = 1.0;
                    ret[0][1] = params[1];
                    ret[1][0] = params[1];
                    ret[1][1] = params[2];
                    ret
                })
            },
            Dct2(params) => {
                params.map(|params| {
                    let mut ret = vec![vec![0.0f32; 8]; 8];
                    for (idx, val) in params.into_iter().enumerate() {
                        let shift = idx / 2;
                        let dim = 1usize << shift;
                        if idx % 2 == 0 {
                            for w in ret[..dim].iter_mut().flat_map(|v| &mut v[dim..][..dim]) {
                                *w = val;
                            }
                            for w in ret[dim..][..dim].iter_mut().flat_map(|v| &mut v[..dim]) {
                                *w = val;
                            }
                        } else {
                            for w in ret[dim..][..dim].iter_mut().flat_map(|v| &mut v[dim..][..dim]) {
                                *w = val;
                            }
                        }
                    }
                    ret
                })
            },
            Dct4 { params, dct_params } => {
                let mut ret = [Vec::new(), Vec::new(), Vec::new()];
                for (ret, (params, dct_params)) in ret.iter_mut().zip(params.into_iter().zip(dct_params)) {
                    let mat = dct_quant_weights(&dct_params, 4, 4);
                    *ret = mat.into_iter()
                        .map(|v| vec![v[0], v[0], v[1], v[1], v[2], v[2], v[3], v[3]])
                        .flat_map(|v| [v.clone(), v])
                        .collect();
                    ret[0][1] /= params[0];
                    ret[1][0] /= params[0];
                    ret[1][1] /= params[1];
                }
                ret
            },
            Dct4x8 { params, dct_params } => {
                let mut ret = [Vec::new(), Vec::new(), Vec::new()];
                for (ret, (params, dct_params)) in ret.iter_mut().zip(params.into_iter().zip(dct_params)) {
                    let mat = dct_quant_weights(&dct_params, 8, 4);
                    *ret = mat.into_iter()
                        .flat_map(|v| [v.clone(), v])
                        .collect();
                    ret[1][0] /= params[0];
                }
                ret
            },
            Afv { params, dct_params, dct4x4_params } => {
                const FREQS: [f32; 16] = [
                    0.0, 0.0, 0.8517779, 5.3777843, 0.0, 0.0, 4.734748, 5.4492455,
                    1.659827, 4.0, 7.275749, 10.423227, 2.6629324, 7.6306577, 8.962389, 12.971662,
                ];
                const FREQ_LO: f32 = FREQS[2];
                const FREQ_HI: f32 = FREQS[15];

                let mut ret = [Vec::new(), Vec::new(), Vec::new()];
                for (ret, ((params, dct_params), dct4x4_params)) in ret.iter_mut().zip(params.into_iter().zip(dct_params).zip(dct4x4_params)) {
                    let weights_4x8 = dct_quant_weights(&dct_params, 8, 4);
                    let weights_4x4 = dct_quant_weights(&dct4x4_params, 4, 4);
                    let mut bands = [params[5], 0.0, 0.0, 0.0];
                    let mut prev_band = bands[0];
                    for (band, &param) in bands[1..].iter_mut().zip(&params[6..]) {
                        *band = prev_band * mult(param);
                        prev_band = *band;
                    }

                    *ret = vec![vec![0.0; 8]; 8];
                    for y in 0..4 {
                        for x in 0..4 {
                            ret[2 * y][2 * x] = match (x, y) {
                                (0, 0) => 1.0,
                                (0, 1) => params[2],
                                (1, 0) => params[3],
                                (1, 1) => params[4],
                                (x, y) => interpolate(
                                    FREQS[y * 4 + x] - FREQ_LO,
                                    FREQ_HI + FREQ_LO + 1e-6,
                                    &bands,
                                )
                            };
                        }
                    }
                    for (y, ((rows, weights_8), weights_4)) in ret.chunks_exact_mut(2).zip(weights_4x8).zip(weights_4x4).enumerate() {
                        let [row0, row1] = rows else { unreachable!() };
                        for (x, (w, dct_weight)) in row1.iter_mut().zip(weights_8).enumerate() {
                            *w = if y == 0 && x == 0 {
                                params[0]
                            } else {
                                dct_weight
                            };
                        }
                        for (x, (pair, dct_weight)) in row0.chunks_exact_mut(2).zip(weights_4).enumerate() {
                            pair[1] = if y == 0 && x == 0 {
                                params[1]
                            } else {
                                dct_weight
                            };
                        }
                    }
                }
                ret
            },
            Raw { denominator, params } => {
                let (width, height) = dct_select.dequant_matrix_size();
                let channel_data = params.image().channel_data();
                [0usize, 1, 2].map(|c| {
                    let channel = &channel_data[c];
                    let mut ret = vec![vec![0.0f32; width as usize]; height as usize];
                    for y in 0..height {
                        for x in 0..width {
                            ret[y as usize][x as usize] = channel[(x, y)] as f32 * denominator;
                        }
                    }
                    ret
                })
            },
        };

        if need_recip {
            for w in weights.iter_mut().flatten().flatten() {
                *w = 1.0 / *w;
            }
        }

        weights
    }
}

#[derive(Debug, Copy, Clone)]
pub struct DequantMatrixSetParams<'a> {
    dct_select: TransformType,
    bit_depth: u32,
    stream_index: u32,
    global_ma_config: Option<&'a jxl_modular::MaConfig>,
}

impl<'a> DequantMatrixSetParams<'a> {
    pub fn new(
        bit_depth: u32,
        stream_index_base: u32,
        global_ma_config: Option<&'a jxl_modular::MaConfig>,
    ) -> Self {
        Self {
            dct_select: TransformType::Dct8,
            bit_depth,
            stream_index: stream_index_base,
            global_ma_config,
        }
    }
}

impl Bundle<DequantMatrixSetParams<'_>> for DequantMatrixParams {
    type Error = crate::Error;

    fn parse<R: std::io::Read>(bitstream: &mut Bitstream<R>, params: DequantMatrixSetParams) -> Result<Self> {
        use DequantMatrixParamsEncoding::*;

        let span = tracing::span!(
            tracing::Level::DEBUG,
            "DequantMatrixParams::parse",
            dct_select = format_args!("{:?}", params.dct_select),
        );
        let _guard = span.enter();

        fn read_fixed<const N: usize, R: std::io::Read>(bitstream: &mut Bitstream<R>) -> Result<[[f32; N]; 3]> {
            let mut out = [[0.0f32; N]; 3];
            for val in out.iter_mut().flatten() {
                *val = bitstream.read_f16_as_f32()?;
            }
            Ok(out)
        }

        fn read_dct_params<R: std::io::Read>(bitstream: &mut Bitstream<R>) -> Result<[Vec<f32>; 3]> {
            let num_params = bitstream.read_bits(4)? as usize + 1;
            let mut params = [
                vec![0.0f32; num_params],
                vec![0.0f32; num_params],
                vec![0.0f32; num_params],
            ];
            for val in params.iter_mut().flatten() {
                *val = bitstream.read_f16_as_f32()?;
            }
            for val in params.iter_mut().map(|v| v.first_mut().unwrap()) {
                *val *= 64.0;
            }
            Ok(params)
        }

        let DequantMatrixSetParams {
            dct_select,
            bit_depth,
            stream_index,
            global_ma_config,
        } = params;

        let encoding_mode = bitstream.read_bits(3)?;
        if encoding_mode != 0 {
            tracing::debug!(
                dct_select = format_args!("{:?}", dct_select),
                bit_depth,
                stream_index,
                encoding_mode,
                "Reading dequant matrix params"
            );
        }
        let encoding = match encoding_mode {
            0 => DequantMatrixParamsEncoding::default_with(dct_select),
            1 => Hornuss(read_fixed(bitstream)?),
            2 => Dct2(read_fixed(bitstream)?),
            3 => Dct4 {
                params: read_fixed(bitstream)?,
                dct_params: read_dct_params(bitstream)?,
            },
            4 => Dct4x8 {
                params: read_fixed(bitstream)?,
                dct_params: read_dct_params(bitstream)?,
            },
            5 => todo!(),
            6 => Dct(read_dct_params(bitstream)?),
            7 => {
                let (width, height) = dct_select.dequant_matrix_size();

                let denominator = bitstream.read_f16_as_f32()?;
                let modular_params = ModularParams::new(
                    width,
                    height,
                    256,
                    bit_depth,
                    vec![jxl_modular::ChannelShift::from_shift(0); 3],
                    global_ma_config,
                );
                let mut params = Modular::parse(bitstream, modular_params)?;
                params.decode_image(bitstream, stream_index)?;
                params.inverse_transform();

                Raw { denominator, params }
            },
            _ => unreachable!(),
        };

        Ok(Self { dct_select, encoding })
    }
}

impl BundleDefault<TransformType> for DequantMatrixParams {
    fn default_with_context(dct_select: TransformType) -> Self {
        Self::default_with(dct_select)
    }
}

#[derive(Debug)]
pub struct DequantMatrixSet {
    matrices: Vec<[Vec<Vec<f32>>; 3]>,
}

impl Bundle<DequantMatrixSetParams<'_>> for DequantMatrixSet {
    type Error = crate::Error;

    fn parse<R: std::io::Read>(bitstream: &mut Bitstream<R>, params: DequantMatrixSetParams<'_>) -> Result<Self> {
        use TransformType::*;
        const DCT_SELECT_LIST: [TransformType; 17] = [
            Dct8,
            Hornuss,
            Dct2,
            Dct4,
            Dct16,
            Dct32,
            Dct8x16,
            Dct8x32,
            Dct16x32,
            Dct4x8,
            Afv0,
            Dct64,
            Dct32x64,
            Dct128,
            Dct64x128,
            Dct256,
            Dct128x256,
        ];

        let param_list: Vec<_> = if bitstream.read_bool()? {
            DCT_SELECT_LIST.into_iter().map(DequantMatrixParams::default_with).collect()
        } else {
            DCT_SELECT_LIST.into_iter().enumerate().map(|(idx, dct_select)| {
                let local_params = DequantMatrixSetParams {
                    dct_select,
                    stream_index: params.stream_index + idx as u32,
                    ..params
                };
                DequantMatrixParams::parse(bitstream, local_params)
            }).collect::<Result<_>>()?
        };

        let matrices = param_list.into_iter()
            .map(|params| params.into_matrix())
            .collect();
        Ok(Self { matrices })
    }
}

impl DequantMatrixSet {
    pub fn get(&self, channel: usize, dct_select: TransformType) -> &[Vec<f32>] {
        use TransformType::*;

        let idx = match dct_select {
            Dct8 => 0,
            Hornuss => 1,
            Dct2 => 2,
            Dct4 => 3,
            Dct16 => 4,
            Dct32 => 5,
            Dct8x16 | Dct16x8 => 6,
            Dct8x32 | Dct32x8 => 7,
            Dct16x32 | Dct32x16 => 8,
            Dct4x8 | Dct8x4 => 9,
            Afv0 | Afv1 | Afv2 | Afv3 => 10,
            Dct64 => 11,
            Dct32x64 | Dct64x32 => 12,
            Dct128 => 13,
            Dct64x128 | Dct128x64 => 14,
            Dct256 => 15,
            Dct128x256 | Dct256x128 => 16,
        };
        &self.matrices[idx][channel]
    }
}
