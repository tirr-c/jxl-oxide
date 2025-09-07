use std::io::Write;

use jxl_bitstream::Bitstream;
use jxl_modular::ChannelShift;

use crate::bit_writer::BitWriter;
use crate::huffman::{self, BuiltHuffmanTable};
use crate::{Error, Result, ScanInfo, ScanMoreInfo};

#[derive(Debug)]
pub(super) struct ScanParams<'jbrd> {
    pub si: &'jbrd ScanInfo,
    pub smi: &'jbrd ScanMoreInfo,
    pub upsampling_shifts_ycbcr: [ChannelShift; 3],
    pub hsamples: Vec<u32>,
    pub vsamples: Vec<u32>,
    pub max_hsample: u32,
    pub max_vsample: u32,
    pub w8: u32,
    pub h8: u32,
}

struct ScanState<'recon> {
    bit_writer: BitWriter,
    dc_pred: Vec<i16>,
    eobrun: u32,
    last_ac_table: Option<&'recon BuiltHuffmanTable>,
    refinement_bitlen: Vec<u8>,
    refinement_bits: Vec<u64>,
    rst_m: u8,
}

impl<'recon> ScanState<'recon> {
    fn new(num_comps: usize) -> Self {
        Self {
            bit_writer: BitWriter::new(),
            dc_pred: vec![0; num_comps],
            eobrun: 0,
            last_ac_table: None,
            refinement_bitlen: Vec::new(),
            refinement_bits: Vec::new(),
            rst_m: 0,
        }
    }

    fn try_init_ac_table(&mut self, ac_table: &'recon BuiltHuffmanTable) {
        self.last_ac_table.get_or_insert(ac_table);
    }

    fn update_ac_table(&mut self, ac_table: &'recon BuiltHuffmanTable) {
        self.last_ac_table = Some(ac_table);
    }

    fn update_dc_pred(&mut self, comp_idx: usize, coeff: i16) -> i16 {
        let diff = coeff.wrapping_sub(self.dc_pred[comp_idx]);
        self.dc_pred[comp_idx] = coeff;
        diff
    }

    fn buffer_refinement_bits(&mut self, bits: u64, bitlen: u8) {
        self.refinement_bits.push(bits);
        self.refinement_bitlen.push(bitlen);
    }

    fn emit_eobrun(&mut self) -> Result<()> {
        if self.eobrun == 0 {
            return Ok(());
        }

        let ac_table = self.last_ac_table.expect("last_ac_table not initialized");

        let eobn = 31 - self.eobrun.leading_zeros();
        let (len, bits) = ac_table.lookup((eobn as u8) << 4)?;
        self.bit_writer.write_huffman(bits, len);
        let mask = (1u32 << eobn) - 1;
        self.bit_writer
            .write_raw((self.eobrun & mask) as u64, eobn as u8);
        self.eobrun = 0;

        let refinement_bits = std::mem::take(&mut self.refinement_bits);
        let refinement_bitlen = std::mem::take(&mut self.refinement_bitlen);
        for (bits, bitlen) in std::iter::zip(refinement_bits, refinement_bitlen) {
            self.bit_writer.write_raw(bits, bitlen);
        }

        Ok(())
    }

    fn flush_bit_writer(
        &mut self,
        padding_bitstream: Option<&mut Bitstream>,
        mut writer: impl Write,
    ) -> Result<()> {
        self.emit_eobrun()?;

        let mut bit_writer = std::mem::replace(&mut self.bit_writer, BitWriter::new());
        let padding_needed = bit_writer.padding_bits();
        if padding_needed != 0 {
            let bits = if let Some(padding_bitstream) = padding_bitstream {
                padding_bitstream
                    .read_bits(padding_needed)
                    .map_err(|_| Error::InvalidData)?
            } else {
                (-1i32) as u32
            };
            bit_writer.write_raw(bits as u64, padding_needed as u8);
        }

        let bytes = bit_writer.finalize();
        writer
            .write_all(&bytes)
            .map_err(Error::ReconstructionWrite)?;

        Ok(())
    }

    fn restart(
        &mut self,
        padding_bitstream: Option<&mut Bitstream>,
        mut writer: impl Write,
    ) -> Result<()> {
        self.dc_pred.fill(0);
        self.flush_bit_writer(padding_bitstream, &mut writer)?;

        let rst = [0xff, 0xd0 + self.rst_m];
        writer.write_all(&rst).map_err(Error::ReconstructionWrite)?;
        self.rst_m = (self.rst_m + 1) % 8;

        Ok(())
    }
}

fn process_sequential(
    state: &mut ScanState,
    component_idx: usize,
    dc_table: &BuiltHuffmanTable,
    ac_table: &BuiltHuffmanTable,
    dc: i16,
    ac: &[i16],
    extra_zero_runs: Option<u32>,
) -> Result<()> {
    let diff = state.update_dc_pred(component_idx, dc);

    let is_neg = diff < 0;
    let bits = if is_neg { -diff } else { diff };
    let bitlen = 16 - bits.leading_zeros();
    let raw_bits = if is_neg { -bits - 1 } else { bits };

    let (len, bits) = dc_table.lookup(bitlen as u8)?;
    state.bit_writer.write_huffman(bits, len);
    state.bit_writer.write_raw(raw_bits as u64, bitlen as u8);

    let mut remaining = ac;
    while let Some(mut nonzero_idx) = remaining.iter().position(|x| *x != 0) {
        let coeff = remaining[nonzero_idx];
        remaining = &remaining[nonzero_idx + 1..];

        while nonzero_idx >= 16 {
            let (len, bits) = ac_table.lookup(0xf0)?;
            state.bit_writer.write_huffman(bits, len);
            nonzero_idx -= 16;
        }

        let (raw_bits, bitlen) = if coeff < 0 {
            let coeff = -coeff;
            (!coeff, 16 - coeff.leading_zeros())
        } else {
            (coeff, 16 - coeff.leading_zeros())
        };

        let nonzero_idx = nonzero_idx as u8;
        let sym = (nonzero_idx << 4) | bitlen as u8;
        let (len, bits) = ac_table.lookup(sym)?;
        state.bit_writer.write_huffman(bits, len);
        state.bit_writer.write_raw(raw_bits as u64, bitlen as u8);
    }

    let mut num_zeros = remaining.len() as i32;
    if let Some(ezr) = extra_zero_runs {
        let (len, bits) = ac_table.lookup(0xf0)?;
        for _ in 0..ezr {
            state.bit_writer.write_huffman(bits, len);
        }

        num_zeros -= ezr as i32 * 16;
    }

    if num_zeros > 0 {
        let (len, bits) = ac_table.lookup(0)?;
        state.bit_writer.write_huffman(bits, len);
    }

    Ok(())
}

fn process_progressive_first<'recon>(
    state: &mut ScanState<'recon>,
    component_idx: usize,
    dc_table: &'recon BuiltHuffmanTable,
    ac_table: &'recon BuiltHuffmanTable,
    dc: Option<i16>,
    ac: &[i16],
    extra_zero_runs: Option<u32>,
) -> Result<()> {
    if let Some(dc) = dc {
        let diff = state.update_dc_pred(component_idx, dc);

        let is_neg = diff < 0;
        let bits = if is_neg { -diff } else { diff };
        let bitlen = 16 - bits.leading_zeros();
        let raw_bits = if is_neg { -bits - 1 } else { bits };

        state.emit_eobrun()?;
        let (len, bits) = dc_table.lookup(bitlen as u8)?;
        state.bit_writer.write_huffman(bits, len);
        state.bit_writer.write_raw(raw_bits as u64, bitlen as u8);
    }

    let mut remaining = ac;
    while let Some(mut nonzero_idx) = remaining.iter().position(|x| *x != 0) {
        state.emit_eobrun()?;

        let coeff = remaining[nonzero_idx];
        remaining = &remaining[nonzero_idx + 1..];

        while nonzero_idx >= 16 {
            let (len, bits) = ac_table.lookup(0xf0)?;
            state.bit_writer.write_huffman(bits, len);
            nonzero_idx -= 16;
        }

        let (raw_bits, bitlen) = if coeff < 0 {
            let coeff = -coeff;
            (!coeff, 16 - coeff.leading_zeros())
        } else {
            (coeff, 16 - coeff.leading_zeros())
        };

        let nonzero_idx = nonzero_idx as u8;
        let sym = (nonzero_idx << 4) | bitlen as u8;
        let (len, bits) = ac_table.lookup(sym)?;
        state.bit_writer.write_huffman(bits, len);
        state.bit_writer.write_raw(raw_bits as u64, bitlen as u8);
    }

    let mut num_zeros = remaining.len() as i32;
    if let Some(ezr) = extra_zero_runs {
        state.emit_eobrun()?;
        let (len, bits) = ac_table.lookup(0xf0)?;
        for _ in 0..ezr {
            state.bit_writer.write_huffman(bits, len);
        }

        num_zeros -= ezr as i32 * 16;
    }

    if state.eobrun == 0 {
        state.update_ac_table(ac_table);
    }

    if num_zeros > 0 {
        state.eobrun += 1;
        if state.eobrun >= 32767 {
            state.emit_eobrun()?;
        }
    }

    Ok(())
}

fn process_progressive_refinement<'recon>(
    state: &mut ScanState<'recon>,
    ac_table: &'recon BuiltHuffmanTable,
    dc: Option<i16>,
    ac: &[i16],
    extra_zero_runs: Option<u32>,
) -> Result<()> {
    if let Some(dc) = dc {
        state.emit_eobrun()?;
        state.bit_writer.write_raw(dc as u64, 1);
    }

    let mut remaining = ac;
    while let Some(nonzero_idx) = remaining.iter().position(|x| *x == 1 || *x == -1) {
        state.emit_eobrun()?;

        let mut zero_runs = 0u8;
        let mut refinement_bitlen = 0u8;
        let mut refinement_bits = 0u64;
        for &coeff in &remaining[..nonzero_idx] {
            if coeff == 0 {
                zero_runs += 1;
                if zero_runs == 16 {
                    let (len, bits) = ac_table.lookup(0xf0)?;
                    state.bit_writer.write_huffman(bits, len);
                    state
                        .bit_writer
                        .write_raw(refinement_bits, refinement_bitlen);
                    zero_runs = 0;
                    refinement_bitlen = 0;
                }
            } else {
                refinement_bits = (refinement_bits << 1) | (coeff & 1) as u64;
                refinement_bitlen += 1;
            }
        }

        let coeff = remaining[nonzero_idx];
        remaining = &remaining[nonzero_idx + 1..];

        let bit = (coeff == 1) as u64;
        let sym = (zero_runs << 4) | 1;
        let (len, bits) = ac_table.lookup(sym)?;
        state.bit_writer.write_huffman(bits, len);
        state.bit_writer.write_raw(bit, 1);
        state
            .bit_writer
            .write_raw(refinement_bits, refinement_bitlen);
    }

    let mut remaining_zrl = extra_zero_runs.unwrap_or(0);
    if remaining_zrl > 0 {
        state.emit_eobrun()?;
    }

    let (zrl_len, zrl_bits) = if remaining_zrl > 0 {
        ac_table.lookup(0xf0)?
    } else {
        (0, 0)
    };

    let mut zero_runs = 0u8;
    let mut refinement_bitlen = 0u8;
    let mut refinement_bits = 0u64;
    for &coeff in remaining {
        if coeff == 0 {
            zero_runs += 1;
            if remaining_zrl > 0 && zero_runs == 16 {
                state.bit_writer.write_huffman(zrl_bits, zrl_len);
                state
                    .bit_writer
                    .write_raw(refinement_bits, refinement_bitlen);
                zero_runs = 0;
                refinement_bitlen = 0;
                remaining_zrl -= 1;
            }
        } else {
            refinement_bits = (refinement_bits << 1) | (coeff & 1) as u64;
            refinement_bitlen += 1;
        }
    }

    for _ in 0..remaining_zrl {
        state.bit_writer.write_huffman(zrl_bits, zrl_len);
        state
            .bit_writer
            .write_raw(refinement_bits, refinement_bitlen);
        zero_runs = 0;
        refinement_bitlen = 0;
    }

    if state.eobrun == 0 {
        state.update_ac_table(ac_table);
    }

    if zero_runs > 0 || refinement_bitlen > 0 {
        state.eobrun += 1;
        state.buffer_refinement_bits(refinement_bits, refinement_bitlen);
        if state.eobrun >= 32767 {
            state.emit_eobrun()?;
        }
    }

    Ok(())
}

impl super::JpegBitstreamReconstructor<'_, '_, '_> {
    pub(super) fn process_scan<const TYPE: usize>(
        &mut self,
        params: ScanParams,
        mut writer: impl Write,
    ) -> Result<()> {
        assert!(TYPE < 3);

        let ScanParams {
            si,
            smi,
            upsampling_shifts_ycbcr,
            hsamples,
            vsamples,
            max_hsample,
            max_vsample,
            w8,
            h8,
        } = params;
        let comps = &si.component_info;
        let frame_header = self.frame.header();
        let dc_offset = self.parsed.dc_offset;

        let ss = si.ss.max(1);
        let se = si.se + 1;
        let al = si.al;

        let group_dim = frame_header.group_dim();

        let mut state = ScanState::new(comps.len());
        let mut block_idx = 0u32;
        for y8 in 0..h8 {
            for x8 in 0..w8 {
                let mcu_idx = x8 + w8 * y8;
                if let Some(restart_interval) = self.restart_interval
                    && mcu_idx != 0
                    && mcu_idx % restart_interval == 0
                {
                    state.restart(self.padding_bitstream.as_mut(), &mut writer)?;
                }

                let group_idx = frame_header
                    .group_idx_from_coord(x8 << (3 + max_hsample), y8 << (3 + max_vsample))
                    .unwrap();
                let lf_group_idx = frame_header.lf_group_idx_from_group_idx(group_idx);

                let hf_coeff = self.parsed.hf_coeff(group_idx);
                let lf_quant = self.parsed.lf_quant(lf_group_idx);

                let it = comps
                    .iter()
                    .enumerate()
                    .zip(std::iter::zip(&hsamples, &vsamples));
                for ((cidx, c), (&hs, &vs)) in it {
                    let dc_table = self.dc_tables[c.dc_tbl_idx as usize]
                        .as_ref()
                        .unwrap_or(&huffman::EMPTY_TABLE);
                    let ac_table = self.ac_tables[c.ac_tbl_idx as usize]
                        .as_ref()
                        .unwrap_or(&huffman::EMPTY_TABLE);

                    state.try_init_ac_table(ac_table);

                    let idx = if frame_header.do_ycbcr {
                        c.comp_idx as usize
                    } else {
                        [1, 0, 2][c.comp_idx as usize]
                    };
                    let lf_quant = &lf_quant[idx];
                    let hf_coeff = hf_coeff[idx];
                    let dc_offset = dc_offset[idx];
                    let shift = upsampling_shifts_ycbcr[idx];
                    let (group_width, group_height) = shift.shift_size((group_dim, group_dim));
                    let group_width_mask = group_width - 1;
                    let group_height_mask = group_height - 1;

                    for dy8 in 0..vs {
                        let y_dc = y8 * vs + dy8;
                        let y_ac_start = y_dc * 8;

                        let y_dc = (y_dc & group_height_mask) as usize;
                        let y_ac_start = (y_ac_start & group_height_mask) as usize;
                        for dx8 in 0..hs {
                            let x_dc = x8 * hs + dx8;
                            let x_ac_start = x_dc * 8;

                            let x_dc = (x_dc & group_width_mask) as usize;
                            let x_ac_start = (x_ac_start & group_width_mask) as usize;

                            let hf_coeff = hf_coeff.subgrid(
                                x_ac_start..(x_ac_start + 8),
                                y_ac_start..(y_ac_start + 8),
                            );

                            let dc_coeff = (si.ss == 0).then(|| {
                                let dc_coeff = lf_quant
                                    .get(x_dc, y_dc)
                                    .saturating_sub(dc_offset)
                                    .clamp(-2047, 2047);
                                dc_coeff >> al
                            });

                            let mut ac_coeffs: Vec<i16> = Vec::with_capacity((se - ss) as usize);
                            for &(x, y) in &jxl_vardct::DCT8_NATURAL_ORDER[ss as usize..se as usize]
                            {
                                let coeff = hf_coeff.get(x as usize, y as usize) as i16;
                                let coeff = if coeff < 0 {
                                    -((-coeff) >> al)
                                } else {
                                    coeff >> al
                                };
                                ac_coeffs.push(coeff);
                            }

                            let extra_zero_runs = smi.extra_zero_runs.get(&block_idx).copied();

                            if smi.reset_points.contains(&block_idx) {
                                state.emit_eobrun()?;
                            }

                            match TYPE {
                                0 => process_sequential(
                                    &mut state,
                                    cidx,
                                    dc_table,
                                    ac_table,
                                    dc_coeff.unwrap(),
                                    &ac_coeffs,
                                    extra_zero_runs,
                                )?,
                                1 => process_progressive_first(
                                    &mut state,
                                    cidx,
                                    dc_table,
                                    ac_table,
                                    dc_coeff,
                                    &ac_coeffs,
                                    extra_zero_runs,
                                )?,
                                2 => process_progressive_refinement(
                                    &mut state,
                                    ac_table,
                                    dc_coeff,
                                    &ac_coeffs,
                                    extra_zero_runs,
                                )?,
                                _ => unreachable!(),
                            };

                            block_idx += 1;
                        }
                    }
                }
            }
        }

        state.flush_bit_writer(self.padding_bitstream.as_mut(), &mut writer)?;
        Ok(())
    }
}
