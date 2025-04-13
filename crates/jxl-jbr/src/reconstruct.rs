use std::io::Write;

use jxl_bitstream::Bitstream;
use jxl_frame::data::{
    HfGlobal, LfGroup, PassGroupParams, PassGroupParamsVardct, decode_pass_group,
};
use jxl_frame::header::Encoding;
use jxl_frame::{Frame, FrameHeader};
use jxl_grid::{AlignedGrid, SharedSubgrid};
use jxl_modular::ChannelShift;

use super::*;
use crate::huffman::BuiltHuffmanTable;

mod scan;

use scan::*;

const CFL_FIXED_POINT_BITS: usize = 11;
const CFL_DEFAULT_COLOR_FACTOR: i32 = 84;

/// JPEG bitstream reconstruction context.
pub struct JpegBitstreamReconstructor<'jbrd, 'frame, 'meta> {
    parsed: ParsedFrameData,
    is_progressive: bool,
    restart_interval: Option<u32>,
    dc_tables: [Option<BuiltHuffmanTable>; 4],
    ac_tables: [Option<BuiltHuffmanTable>; 4],

    header: &'jbrd JpegBitstreamHeader,
    frame: &'frame Frame,
    marker_ptr: usize,
    app_marker_ptr: std::slice::Iter<'jbrd, AppMarker>,
    next_icc_marker: usize,
    icc_marker_offset: usize,
    num_icc_markers: usize,
    app_data: &'jbrd [u8],
    com_length: std::slice::Iter<'jbrd, u32>,
    com_data: &'jbrd [u8],
    intermarker_length: std::slice::Iter<'jbrd, u32>,
    intermarker_data: &'jbrd [u8],
    huffman_code_ptr: &'jbrd [HuffmanCode],
    quant_ptr: &'jbrd [QuantTable],
    last_quant_val: Option<Box<[u16; 64]>>,
    padding_bitstream: Option<Bitstream<'jbrd>>,
    scan_info_ptr: usize,
    tail_data: &'jbrd [u8],

    icc_profile: &'meta [u8],
    exif: &'meta [u8],
    xmp: &'meta [u8],
}

impl<'jbrd, 'frame, 'meta> JpegBitstreamReconstructor<'jbrd, 'frame, 'meta> {
    pub(crate) fn new(
        header: &'jbrd JpegBitstreamHeader,
        data: &'jbrd [u8],
        frame: &'frame Frame,
        icc_profile: &'meta [u8],
        exif: &'meta [u8],
        xmp: &'meta [u8],
        pool: &jxl_threadpool::JxlThreadPool,
    ) -> Result<Self> {
        let expected_icc_len = header.expected_icc_len();
        let expected_exif_len = header.expected_exif_len();
        let expected_xmp_len = header.expected_xmp_len();
        if expected_icc_len > 0 && expected_icc_len != icc_profile.len() {
            tracing::error!(
                icc_profile_len = icc_profile.len(),
                expected = expected_icc_len,
                "ICC profile length doesn't match expected length"
            );
            return Err(Error::InvalidData);
        }
        if expected_exif_len > 0 && expected_exif_len != exif.len() {
            tracing::error!(
                icc_profile_len = exif.len(),
                expected = expected_exif_len,
                "Exif metadata length doesn't match expected length"
            );
            return Err(Error::InvalidData);
        }
        if expected_xmp_len > 0 && expected_xmp_len != xmp.len() {
            tracing::error!(
                icc_profile_len = xmp.len(),
                expected = expected_xmp_len,
                "XMP metadata length doesn't match expected length"
            );
            return Err(Error::InvalidData);
        }
        let com_data_start = header.app_data_len();
        let intermarker_data_start = com_data_start + header.com_data_len();
        let tail_data_start = intermarker_data_start + header.intermarker_data_len();

        if frame.image_header().metadata.xyb_encoded {
            return Err(Error::IncompatibleFrame);
        }

        let frame_header = frame.header();
        if frame_header.encoding != Encoding::VarDct {
            return Err(Error::IncompatibleFrame);
        }
        if !frame_header.frame_type.is_normal_frame() {
            return Err(Error::IncompatibleFrame);
        }
        if frame_header.flags.use_lf_frame() || !frame_header.flags.skip_adaptive_lf_smoothing() {
            return Err(Error::IncompatibleFrame);
        }

        let lf_global = frame
            .try_parse_lf_global::<i16>()
            .ok_or(Error::FrameDataIncomplete)??;
        let lf_global_vardct = lf_global.vardct.as_ref().unwrap();
        let global_ma_config = lf_global.gmodular.ma_config();

        let mut jpeg_upsampling_ycbcr = frame_header.jpeg_upsampling;
        jpeg_upsampling_ycbcr.swap(0, 1);
        let is_subsampled = jpeg_upsampling_ycbcr.iter().any(|&x| x != 0);
        let upsampling_shifts_ycbcr: [_; 3] = std::array::from_fn(|idx| {
            ChannelShift::from_jpeg_upsampling(jpeg_upsampling_ycbcr, idx)
        });

        if !is_subsampled {
            let lf_chan_corr = &lf_global_vardct.lf_chan_corr;
            if lf_chan_corr.colour_factor != CFL_DEFAULT_COLOR_FACTOR as u32
                || lf_chan_corr.base_correlation_x != 0.0
                || lf_chan_corr.base_correlation_b != 0.0
            {
                return Err(Error::IncompatibleFrame);
            }
        }

        let num_lf_groups = frame.header().num_lf_groups();
        let mut lf_groups = Vec::with_capacity(num_lf_groups as usize);
        lf_groups.resize_with(num_lf_groups as usize, || None);

        let hf_global_out = std::sync::Mutex::new(None);
        let result = std::sync::RwLock::new(Result::Ok(()));
        pool.scope(|scope| {
            scope.spawn(|_| {
                let hf_global = match frame.try_parse_hf_global(Some(&lf_global)).transpose() {
                    Ok(Some(x)) => x,
                    Ok(None) => {
                        *result.write().unwrap() = Err(Error::FrameDataIncomplete);
                        return;
                    }
                    Err(e) => {
                        *result.write().unwrap() = Err(Error::FrameParse(e));
                        return;
                    }
                };

                for c in 0..3 {
                    if hf_global.dequant_matrices.jpeg_quant_values(c).is_none() {
                        *result.write().unwrap() = Err(Error::IncompatibleFrame);
                        return;
                    }
                }

                *hf_global_out.lock().unwrap() = Some(hf_global);
            });

            for (lf_group_idx, out) in lf_groups.iter_mut().enumerate() {
                let result = &result;
                let lf_group_idx = lf_group_idx as u32;
                scope.spawn(move |_| {
                    let r = frame
                        .try_parse_lf_group(
                            Some(lf_global_vardct),
                            global_ma_config,
                            None,
                            lf_group_idx,
                        )
                        .transpose()
                        .map_err(Error::FrameParse);

                    match r {
                        Ok(x) => {
                            *out = x;
                        }
                        Err(e) => {
                            *result.write().unwrap() = Err(e);
                        }
                    }
                });
            }
        });

        result.into_inner().unwrap()?;
        let hf_global = hf_global_out.into_inner().unwrap().unwrap();
        let lf_groups = lf_groups
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or(Error::FrameDataIncomplete)?;

        let num_groups = frame_header.num_groups();
        let mut pass_groups = Vec::with_capacity(num_groups as usize);
        for group_idx in 0..num_groups {
            let (w, h) = frame_header.group_size_for(group_idx);

            pass_groups.push(upsampling_shifts_ycbcr.map(|shift| {
                let (w8, h8) = shift.shift_size((w.div_ceil(8), h.div_ceil(8)));
                let w = w8 * 8;
                let h = h8 * 8;
                AlignedGrid::<i32>::with_alloc_tracker(w as usize, h as usize, None).unwrap()
            }));
        }

        let mut hf_coeff_output = pass_groups
            .iter_mut()
            .map(|pass_group| {
                let [cb, y, cr] = pass_group.each_mut();
                [y, cb, cr].map(|grid| grid.as_subgrid_mut())
            })
            .collect::<Vec<_>>();

        for pass_idx in 0..frame_header.passes.num_passes {
            let mut pass_group_params = Vec::with_capacity(num_groups as usize);

            for (group_idx, hf_coeff_output) in hf_coeff_output.iter_mut().enumerate() {
                let group_idx = group_idx as u32;
                let pass_group = frame
                    .pass_group_bitstream(pass_idx, group_idx)
                    .ok_or(Error::FrameDataIncomplete)
                    .and_then(|r| r.map_err(Error::FrameParse))?;
                if pass_group.partial {
                    return Err(Error::FrameDataIncomplete);
                }
                let bitstream = pass_group.bitstream;

                let lf_group_idx = frame_header.lf_group_idx_from_group_idx(group_idx);
                let lf_group = &lf_groups[lf_group_idx as usize];
                let params = PassGroupParams {
                    frame_header,
                    lf_group,
                    pass_idx,
                    group_idx,
                    global_ma_config,
                    modular: None,
                    vardct: Some(PassGroupParamsVardct {
                        lf_vardct: lf_global_vardct,
                        hf_global: &hf_global,
                        hf_coeff_output,
                    }),
                    allow_partial: false,
                    tracker: None,
                    pool,
                };
                pass_group_params.push((bitstream, params));
            }

            let result = std::sync::RwLock::new(Result::Ok(()));
            pool.for_each_vec(pass_group_params, |(mut bitstream, params)| {
                if let Err(e) = decode_pass_group(&mut bitstream, params) {
                    *result.write().unwrap() = Err(Error::FrameParse(e));
                }
            });
            result.into_inner().unwrap()?;
        }

        if !header.is_gray && !is_subsampled {
            Self::integer_cfl(frame_header, &hf_global, &lf_groups, &mut pass_groups, pool);
        }

        let dc_offset = if frame_header.do_ycbcr {
            [0; 3]
        } else {
            let dequant_x = hf_global.dequant_matrices.jpeg_quant_values(0).unwrap();
            let dequant_y = hf_global.dequant_matrices.jpeg_quant_values(1).unwrap();
            let dequant_b = hf_global.dequant_matrices.jpeg_quant_values(2).unwrap();
            let dc_dequant = [dequant_y[0], dequant_x[0], dequant_b[0]];
            dc_dequant.map(|q| (1024 / q) as i16)
        };

        Ok(Self {
            parsed: ParsedFrameData {
                hf_global,
                lf_groups,
                pass_groups,
                dc_offset,
            },
            is_progressive: false,
            restart_interval: None,
            dc_tables: [None, None, None, None],
            ac_tables: [None, None, None, None],

            header,
            frame,
            marker_ptr: 0,
            app_marker_ptr: header.app_markers.iter(),
            next_icc_marker: 0,
            icc_marker_offset: 0,
            num_icc_markers: header.app_markers.iter().filter(|am| am.ty == 1).count(),
            app_data: &data[..com_data_start],
            com_length: header.com_lengths.iter(),
            com_data: &data[com_data_start..intermarker_data_start],
            intermarker_length: header.intermarker_lengths.iter(),
            intermarker_data: &data[intermarker_data_start..tail_data_start],
            huffman_code_ptr: &header.huffman_codes,
            quant_ptr: &header.quant_tables,
            last_quant_val: None,
            padding_bitstream: header
                .padding_bits
                .as_ref()
                .map(|padding| Bitstream::new(&padding.bits)),
            scan_info_ptr: 0,
            tail_data: &data[tail_data_start..],

            icc_profile,
            exif,
            xmp,
        })
    }

    // Integer chroma-from-luma, copied from libjxl.
    fn integer_cfl(
        frame_header: &FrameHeader,
        hf_global: &HfGlobal,
        lf_groups: &[LfGroup<i16>],
        pass_groups: &mut [[AlignedGrid<i32>; 3]],
        pool: &jxl_threadpool::JxlThreadPool,
    ) {
        let dequant_x = hf_global.dequant_matrices.jpeg_quant_values(0).unwrap();
        let dequant_y = hf_global.dequant_matrices.jpeg_quant_values(1).unwrap();
        let dequant_b = hf_global.dequant_matrices.jpeg_quant_values(2).unwrap();

        let dequant_yx = std::iter::zip(dequant_y, dequant_x)
            .map(|(&y, &x)| (1 << CFL_FIXED_POINT_BITS) * y / x)
            .collect::<Vec<_>>();
        let dequant_yb = std::iter::zip(dequant_y, dequant_b)
            .map(|(&y, &b)| (1 << CFL_FIXED_POINT_BITS) * y / b)
            .collect::<Vec<_>>();
        let quant_ratio = [dequant_yx, dequant_yb];

        let groups_per_row = frame_header.groups_per_row();
        pool.scope(|scope| {
            for (group_idx, [coeff_y, coeff_x, coeff_b]) in pass_groups.iter_mut().enumerate() {
                let group_idx = group_idx as u32;

                let lf_group_idx = frame_header.lf_group_idx_from_group_idx(group_idx);
                let lf_group = &lf_groups[lf_group_idx as usize];

                let group_x_in_lf = (group_idx % groups_per_row) % 8;
                let group_y_in_lf = (group_idx / groups_per_row) % 8;
                let cfl_x = group_x_in_lf as usize * 4;
                let cfl_y = group_y_in_lf as usize * 4;
                let x_from_y = &lf_group.hf_meta.as_ref().unwrap().x_from_y;
                let b_from_y = &lf_group.hf_meta.as_ref().unwrap().b_from_y;
                let cfl_x_end = (cfl_x + 4).min(x_from_y.width());
                let cfl_y_end = (cfl_y + 4).min(x_from_y.height());
                let x_from_y = x_from_y
                    .as_subgrid()
                    .subgrid(cfl_x..cfl_x_end, cfl_y..cfl_y_end);
                let b_from_y = b_from_y
                    .as_subgrid()
                    .subgrid(cfl_x..cfl_x_end, cfl_y..cfl_y_end);

                let cfl_factor = [x_from_y, b_from_y];
                let coeff = [coeff_x, coeff_b];
                let it = std::iter::zip(cfl_factor, coeff).zip(&quant_ratio);

                let width = coeff_y.width();
                let height = coeff_y.height();
                let rounding_const = 1i32 << (CFL_FIXED_POINT_BITS - 1);

                scope.spawn(move |_| {
                    for ((cfl_factor, coeff), quant_ratio) in it {
                        for y in 0..height {
                            let cfl_y = y / 64;
                            let q_y = y % 8;
                            for x in 0..width {
                                let cfl_x = x / 64;
                                let factor = cfl_factor.get(cfl_x, cfl_y);
                                let coeff_y = coeff_y.get(x, y);

                                // Dequant matrix is transposed.
                                let q_x = x % 8;
                                let q = quant_ratio[q_y + 8 * q_x];

                                let scale_factor =
                                    factor * (1 << CFL_FIXED_POINT_BITS) / CFL_DEFAULT_COLOR_FACTOR;
                                let q_scale =
                                    (q * scale_factor + rounding_const) >> CFL_FIXED_POINT_BITS;
                                let cfl_factor =
                                    (coeff_y * q_scale + rounding_const) >> CFL_FIXED_POINT_BITS;
                                *coeff.get_mut(x, y) += cfl_factor;
                            }
                        }
                    }
                });
            }
        });
    }
}

impl JpegBitstreamReconstructor<'_, '_, '_> {
    /// Writes reconstructed JPEG bitstream to the writer.
    pub fn write(mut self, mut writer: impl Write) -> Result<()> {
        writer
            .write_all(&[0xff, 0xd8])
            .map_err(Error::ReconstructionWrite)?;

        while self.marker_ptr < self.header.markers.len() {
            self.process_next(&mut writer)?;
            self.marker_ptr += 1;
        }

        Ok(())
    }

    fn process_next(&mut self, mut writer: impl Write) -> Result<()> {
        let marker = self.header.markers[self.marker_ptr];
        match marker {
            // SOF
            0xc0 | 0xc1 | 0xc2 | 0xc9 | 0xca => {
                self.is_progressive = marker == 0xc2 || marker == 0xca;

                let width = self.frame.image_header().size.width;
                let height = self.frame.image_header().size.height;
                let width_bytes = (width as u16).to_be_bytes();
                let height_bytes = (height as u16).to_be_bytes();

                let num_comps = self.header.components.len();
                let encoded_len = 8 + num_comps * 3;
                let encoded_len_bytes = (encoded_len as u16).to_be_bytes();

                let mut header = [0xff, marker, 0, 0, 8, 0, 0, 0, 0, num_comps as u8];
                header[2..4].copy_from_slice(&encoded_len_bytes);
                header[5..7].copy_from_slice(&height_bytes);
                header[7..9].copy_from_slice(&width_bytes);
                writer
                    .write_all(&header)
                    .map_err(Error::ReconstructionWrite)?;

                let mut jpeg_upsampling_ycbcr = self.frame.header().jpeg_upsampling;
                jpeg_upsampling_ycbcr.swap(0, 1);

                for (idx, comp) in self.header.components.iter().enumerate() {
                    let sampling_factor = jpeg_upsampling_ycbcr.get(idx).copied().unwrap_or(0);
                    let sampling_val = match sampling_factor {
                        0 => 0b01_0001,
                        1 => 0b10_0010,
                        2 => 0b10_0001,
                        3 => 0b01_0010,
                        _ => 0b01_0001,
                    };
                    let component_bytes = [comp.id, sampling_val, comp.q_idx];
                    writer
                        .write_all(&component_bytes)
                        .map_err(Error::ReconstructionWrite)?;
                }
            }

            // DHT
            0xc4 => {
                let last_idx = self.huffman_code_ptr.iter().position(|hc| hc.is_last);
                let num_tables = last_idx.expect("is_last not found") + 1;
                let (hcs, remainder) = self.huffman_code_ptr.split_at(num_tables);
                self.huffman_code_ptr = remainder;

                let encoded_len = 2 + hcs.iter().map(|hc| hc.encoded_len()).sum::<usize>();
                let mut header = [0xff, 0xc4, 0, 0];
                header[2..].copy_from_slice(&(encoded_len as u16).to_be_bytes());
                writer
                    .write_all(&header)
                    .map_err(Error::ReconstructionWrite)?;

                for hc in hcs {
                    let mut id_and_counts = [0u8; 17];
                    id_and_counts[0] = hc.id | if hc.is_ac { 0x10 } else { 0 };
                    id_and_counts[1..].copy_from_slice(&hc.counts[1..]);
                    if let Some(x) = id_and_counts[1..].iter_mut().rev().find(|x| **x != 0) {
                        *x -= 1;
                    }

                    writer
                        .write_all(&id_and_counts)
                        .map_err(Error::ReconstructionWrite)?;
                    writer
                        .write_all(&hc.values[..hc.values.len() - 1])
                        .map_err(Error::ReconstructionWrite)?;

                    let table = hc.build();
                    if hc.is_ac {
                        self.ac_tables[hc.id as usize] = Some(table);
                    } else {
                        self.dc_tables[hc.id as usize] = Some(table);
                    }
                }
            }

            // RSTn
            0xd0..=0xd7 => {
                writer
                    .write_all(&[0xff, marker])
                    .map_err(Error::ReconstructionWrite)?;
            }

            // EOI
            0xd9 => {
                writer
                    .write_all(&[0xff, 0xd9])
                    .map_err(Error::ReconstructionWrite)?;
                writer
                    .write_all(self.tail_data)
                    .map_err(Error::ReconstructionWrite)?;
            }

            // SOS
            0xda => {
                let frame_header = self.frame.header();

                let idx = self.scan_info_ptr;
                let si = &self.header.scan_info[idx];
                let smi = &self.header.scan_more_info[idx];
                self.scan_info_ptr += 1;
                if si.component_info.is_empty() {
                    tracing::error!("No component in SOS marker");
                    return Err(Error::InvalidData);
                }

                let num_comps = si.num_comps();
                let header_len_bytes = (6 + 2 * num_comps as u16).to_be_bytes();
                let header = [
                    0xff,
                    0xda,
                    header_len_bytes[0],
                    header_len_bytes[1],
                    num_comps,
                ];
                writer
                    .write_all(&header)
                    .map_err(Error::ReconstructionWrite)?;

                let comps = &si.component_info;
                for c in comps {
                    let id = self.header.components[c.comp_idx as usize].id;
                    let table = (c.dc_tbl_idx << 4) | c.ac_tbl_idx;
                    writer
                        .write_all(&[id, table])
                        .map_err(Error::ReconstructionWrite)?;
                }

                writer
                    .write_all(&[si.ss, si.se, (si.ah << 4) | si.al])
                    .map_err(Error::ReconstructionWrite)?;

                let mut jpeg_upsampling_ycbcr = self.frame.header().jpeg_upsampling;
                jpeg_upsampling_ycbcr.swap(0, 1);
                let upsampling_shifts_ycbcr: [_; 3] = std::array::from_fn(|idx| {
                    jxl_modular::ChannelShift::from_jpeg_upsampling(jpeg_upsampling_ycbcr, idx)
                });

                let mut hsamples = comps
                    .iter()
                    .map(|c| [1u32, 2, 2, 1][jpeg_upsampling_ycbcr[c.comp_idx as usize] as usize])
                    .collect::<Vec<_>>();
                let mut vsamples = comps
                    .iter()
                    .map(|c| [1u32, 2, 1, 2][jpeg_upsampling_ycbcr[c.comp_idx as usize] as usize])
                    .collect::<Vec<_>>();

                let mut max_hsample = hsamples.iter().copied().max().unwrap().trailing_zeros();
                let mut max_vsample = vsamples.iter().copied().max().unwrap().trailing_zeros();
                let mut w8 = (frame_header.width.div_ceil(8) + max_hsample) >> max_hsample;
                let mut h8 = (frame_header.height.div_ceil(8) + max_vsample) >> max_vsample;

                if num_comps == 1 {
                    let full_w8 = frame_header.width.div_ceil(8);
                    let full_h8 = frame_header.height.div_ceil(8);
                    if (1 << max_hsample) == hsamples[0] {
                        w8 = full_w8;
                        max_hsample = 0;
                    }
                    if (1 << max_vsample) == vsamples[0] {
                        h8 = full_h8;
                        max_vsample = 0;
                    }

                    hsamples = vec![1];
                    vsamples = vec![1];
                }

                let params = ScanParams {
                    si,
                    smi,
                    upsampling_shifts_ycbcr,
                    hsamples,
                    vsamples,
                    max_hsample,
                    max_vsample,
                    w8,
                    h8,
                };

                if !self.is_progressive {
                    if si.ss != 0 || si.se != 0x3f || si.al != 0 || si.ah != 0 {
                        tracing::error!(
                            si.ss,
                            si.se,
                            si.al,
                            si.ah,
                            "Progressive parameter set for sequential JPEG"
                        );
                        return Err(Error::InvalidData);
                    }
                    self.process_scan::<0>(params, writer)?;
                } else if si.ah == 0 {
                    self.process_scan::<1>(params, writer)?;
                } else {
                    self.process_scan::<2>(params, writer)?;
                }
            }

            // DQT
            0xdb => {
                let do_ycbcr = self.frame.header().do_ycbcr;
                let hf_global = &self.parsed.hf_global;

                let last_idx = self.quant_ptr.iter().position(|qt| qt.is_last);
                let num_tables = last_idx.expect("is_last not found") + 1;
                let (qts, remainder) = self.quant_ptr.split_at(num_tables);
                self.quant_ptr = remainder;

                let encoded_len =
                    2 + 65 * num_tables + 64 * qts.iter().filter(|qt| qt.precision != 0).count();
                let mut header = [0xff, 0xdb, 0, 0];
                header[2..].copy_from_slice(&(encoded_len as u16).to_be_bytes());
                writer
                    .write_all(&header)
                    .map_err(Error::ReconstructionWrite)?;

                for qt in qts {
                    let channel = self
                        .header
                        .components
                        .iter()
                        .position(|c| c.q_idx == qt.index);
                    let q = channel.and_then(|mut channel| {
                        if do_ycbcr && channel <= 1 {
                            channel ^= 1;
                        }
                        hf_global.dequant_matrices.jpeg_quant_values(channel)
                    });
                    if let Some(q) = q {
                        if self.last_quant_val.is_none() {
                            self.last_quant_val = Some(Box::new([0; 64]));
                        }
                        let q_val = self.last_quant_val.as_deref_mut().unwrap();

                        // Transposed for DCT8
                        for (&(y, x), q_val) in
                            std::iter::zip(jxl_vardct::DCT8_NATURAL_ORDER, q_val)
                        {
                            *q_val = q[x as usize + 8 * y as usize] as u16
                        }
                    }

                    let Some(q_val) = self.last_quant_val.as_deref() else {
                        return Err(Error::InvalidData);
                    };

                    let buf = if qt.precision == 0 {
                        let mut buf = vec![0u8; 65];
                        buf[0] = qt.index;
                        for (&q, out) in std::iter::zip(q_val, &mut buf[1..]) {
                            *out = q as u8;
                        }
                        buf
                    } else {
                        let mut buf = vec![0u8; 129];
                        buf[0] = qt.index | (qt.precision << 4);
                        for (&q, out) in std::iter::zip(q_val, buf[1..].chunks_exact_mut(2)) {
                            out.copy_from_slice(&q.to_be_bytes());
                        }
                        buf
                    };

                    writer.write_all(&buf).map_err(Error::ReconstructionWrite)?;
                }
            }

            // DRI
            0xdd => {
                let interval = (self.header.restart_interval as u16).to_be_bytes();
                let bytes = [0xff, 0xdd, 0, 4, interval[0], interval[1]];
                writer
                    .write_all(&bytes)
                    .map_err(Error::ReconstructionWrite)?;
                if self.header.restart_interval != 0 {
                    self.restart_interval = Some(self.header.restart_interval);
                }
            }

            // APPn
            0xe0..=0xef => {
                let am = self.app_marker_ptr.next().unwrap();
                let encoded_len = ((am.length - 1) as u16).to_be_bytes();
                match am.ty {
                    0 => {
                        writer
                            .write_all(&[0xff])
                            .map_err(Error::ReconstructionWrite)?;
                        let (app_data, next) = self.app_data.split_at(am.length as usize);
                        self.app_data = next;
                        writer
                            .write_all(app_data)
                            .map_err(Error::ReconstructionWrite)?;
                    }
                    1 => {
                        let header = [0xff, 0xe2, encoded_len[0], encoded_len[1]];
                        writer
                            .write_all(&header)
                            .map_err(Error::ReconstructionWrite)?;
                        writer
                            .write_all(HEADER_ICC)
                            .map_err(Error::ReconstructionWrite)?;
                        let curr = self.next_icc_marker as u8 + 1;
                        let total = self.num_icc_markers as u8;
                        writer
                            .write_all(&[curr, total])
                            .map_err(Error::ReconstructionWrite)?;

                        let from = self.icc_marker_offset;
                        let len = am.length as usize - 5 - HEADER_ICC.len();
                        self.next_icc_marker += 1;
                        self.icc_marker_offset += len;

                        writer
                            .write_all(&self.icc_profile[from..][..len])
                            .map_err(Error::ReconstructionWrite)?;
                    }
                    2 => {
                        let header = [0xff, 0xe1, encoded_len[0], encoded_len[1]];
                        writer
                            .write_all(&header)
                            .map_err(Error::ReconstructionWrite)?;
                        writer
                            .write_all(HEADER_EXIF)
                            .map_err(Error::ReconstructionWrite)?;
                        writer
                            .write_all(self.exif)
                            .map_err(Error::ReconstructionWrite)?;
                    }
                    3 => {
                        let header = [0xff, 0xe1, encoded_len[0], encoded_len[1]];
                        writer
                            .write_all(&header)
                            .map_err(Error::ReconstructionWrite)?;
                        writer
                            .write_all(HEADER_XMP)
                            .map_err(Error::ReconstructionWrite)?;
                        writer
                            .write_all(self.xmp)
                            .map_err(Error::ReconstructionWrite)?;
                    }
                    _ => unreachable!(),
                }
            }

            // COM
            0xfe => {
                let length = *self.com_length.next().unwrap();
                let (com_data, next) = self.com_data.split_at(length as usize);
                self.com_data = next;
                writer
                    .write_all(&[0xff, 0xfe])
                    .map_err(Error::ReconstructionWrite)?;
                writer
                    .write_all(com_data)
                    .map_err(Error::ReconstructionWrite)?;
            }

            // Unrecognized
            0xff => {
                let length = *self.intermarker_length.next().unwrap();
                let (data, next) = self.intermarker_data.split_at(length as usize);
                self.intermarker_data = next;
                writer.write_all(data).map_err(Error::ReconstructionWrite)?;
            }

            _ => {
                tracing::error!(marker, "Unknown marker");
                return Err(Error::InvalidData);
            }
        }

        Ok(())
    }
}

struct ParsedFrameData {
    hf_global: HfGlobal,
    lf_groups: Vec<LfGroup<i16>>,
    pass_groups: Vec<[AlignedGrid<i32>; 3]>,
    dc_offset: [i16; 3],
}

impl ParsedFrameData {
    fn hf_coeff(&self, group_idx: u32) -> [SharedSubgrid<'_, i32>; 3] {
        self.pass_groups[group_idx as usize]
            .each_ref()
            .map(|g| g.as_subgrid())
    }

    fn lf_quant(&self, lf_group_idx: u32) -> &[AlignedGrid<i16>] {
        let lf_coeff = self.lf_groups[lf_group_idx as usize]
            .lf_coeff
            .as_ref()
            .unwrap();
        lf_coeff.lf_quant.image().unwrap().image_channels()
    }
}
