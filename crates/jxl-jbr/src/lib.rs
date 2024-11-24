use std::collections::{HashMap, HashSet};
use std::io::Write;

use bit_writer::BitWriter;
use brotli_decompressor::DecompressorWriter;
use jxl_bitstream::{Bitstream, U};
use jxl_frame::data::{decode_pass_group, HfGlobal, LfGroup, PassGroupParams, PassGroupParamsVardct};
use jxl_frame::Frame;
use jxl_grid::AlignedGrid;
use jxl_modular::ChannelShift;
use jxl_oxide_common::Bundle;

mod bit_writer;
mod huffman;

use huffman::{BuiltHuffmanTable, HuffmanCode};

#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    Bitstream(jxl_bitstream::Error),
    Brotli(std::io::Error),
    InvalidData,
    ReconstructionWrite(std::io::Error),
    ReconstructionDataIncomplete,
    FrameDataIncomplete,
    FrameParse(jxl_frame::Error),
}

impl From<jxl_bitstream::Error> for Error {
    fn from(value: jxl_bitstream::Error) -> Self {
        Self::Bitstream(value)
    }
}

impl From<jxl_frame::Error> for Error {
    fn from(value: jxl_frame::Error) -> Self {
        Self::FrameParse(value)
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Bitstream(e) => write!(f, "failed to read reconstruction data: {e}"),
            Error::Brotli(e) => write!(f, "failed to decompress Brotli stream: {e}"),
            Error::InvalidData => write!(f, "invalid reconstruction data"),
            Error::ReconstructionWrite(e) => write!(f, "failed to write data: {e}"),
            Error::ReconstructionDataIncomplete => write!(f, "reconstruction data is incomplete"),
            Error::FrameDataIncomplete => write!(f, "JPEG XL frame data is incomplete"),
            Error::FrameParse(e) => write!(f, "error parsing JPEG XL frame: {e}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Bitstream(e) => Some(e),
            Error::Brotli(e) => Some(e),
            Error::ReconstructionWrite(e) => Some(e),
            Error::FrameParse(e) => Some(e),
            _ => None,
        }
    }
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

pub struct JpegBitstreamData {
    header: Box<JpegBitstreamHeader>,
    data_stream: Box<DecompressorWriter<Vec<u8>>>,
}

impl std::fmt::Debug for JpegBitstreamData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JpegBitstreamData").field("header", &self.header).finish_non_exhaustive()
    }
}

impl JpegBitstreamData {
    pub fn try_parse(data: &[u8]) -> Result<Option<Self>> {
        let mut bitstream = Bitstream::new(data);
        let header = match JpegBitstreamHeader::parse(&mut bitstream, ()) {
            Ok(header) => Box::new(header),
            Err(e) if e.unexpected_eof() => return Ok(None),
            Err(e) => return Err(e.into()),
        };
        bitstream.zero_pad_to_byte()?;

        let bytes_read = bitstream.num_read_bits() / 8;
        let compressed_data = &data[bytes_read..];
        let mut data_stream = Box::new(DecompressorWriter::new(Vec::new(), 4096));
        data_stream
            .write_all(compressed_data)
            .map_err(Error::Brotli)?;

        Ok(Some(Self {
            header,
            data_stream,
        }))
    }

    pub fn feed_bytes(&mut self, data: &[u8]) -> Result<()> {
        self.data_stream.write_all(data).map_err(Error::Brotli)
    }

    pub fn finalize(&mut self) -> Result<()> {
        self.data_stream.flush().map_err(Error::Brotli)?;

        let decompressed_len = self.data_stream.get_ref().len();
        if decompressed_len != self.header.expected_data_len() {
            return Err(Error::InvalidData);
        }

        Ok(())
    }

    pub fn is_complete(&mut self) -> Result<bool> {
        self.data_stream.flush().map_err(Error::Brotli)?;
        let decompressed_len = self.data_stream.get_ref().len();
        Ok(decompressed_len >= self.header.expected_data_len())
    }

    pub fn reconstruct<'jbrd, 'frame>(
        &'jbrd self,
        frame: &'frame Frame,
    ) -> Result<JpegBitstreamReconstructor<'jbrd, 'frame>> {
        let Self {
            ref header,
            ref data_stream,
        } = *self;
        JpegBitstreamReconstructor::new(
            header,
            data_stream.get_ref(),
            frame,
        )
    }

    pub fn header(&self) -> &JpegBitstreamHeader {
        &self.header
    }
}

#[derive(Debug)]
pub struct JpegBitstreamHeader {
    #[allow(unused)]
    is_gray: bool,
    markers: Vec<u8>,
    app_markers: Vec<AppMarker>,
    com_lengths: Vec<u32>,
    quant_tables: Vec<QuantTable>,
    components: Vec<Component>,
    huffman_codes: Vec<HuffmanCode>,
    scan_info: Vec<ScanInfo>,
    restart_interval: u32,
    scan_more_info: Vec<ScanMoreInfo>,
    intermarker_lengths: Vec<u32>,
    tail_data_length: u32,
    padding_bits: Option<Padding>,
}

impl Bundle for JpegBitstreamHeader {
    type Error = jxl_bitstream::Error;

    fn parse(bitstream: &mut Bitstream, _: ()) -> Result<Self, Self::Error> {
        let is_gray = bitstream.read_bool()?;

        let mut markers = Vec::new();
        let mut num_app_markers = 0usize;
        let mut num_com_markers = 0usize;
        let mut num_scans = 0usize;
        let mut num_intermarkers = 0usize;
        let mut has_dri = false;
        while markers.last() != Some(&0xd9) {
            let marker_bits = bitstream.read_bits(6)? as u8 + 0xc0;
            match marker_bits {
                0xe0..=0xef => num_app_markers += 1,
                0xfe => num_com_markers += 1,
                0xda => num_scans += 1,
                0xff => num_intermarkers += 1,
                0xdd => has_dri = true,
                _ => {}
            }
            markers.push(marker_bits);
        }
        eprintln!("{markers:#x?}");

        let app_markers = (0..num_app_markers)
            .map(|_| AppMarker::parse(bitstream, ()))
            .collect::<Result<_, _>>()?;
        let com_lengths = (0..num_com_markers)
            .map(|_| bitstream.read_bits(16).map(|x| x + 1))
            .collect::<Result<_, _>>()?;

        let num_quant_tables = bitstream.read_bits(2)? + 1;
        let quant_tables = (0..num_quant_tables)
            .map(|_| QuantTable::parse(bitstream, ()))
            .collect::<Result<_, _>>()?;

        let comp_type = bitstream.read_bits(2)?;
        let component_ids = match comp_type {
            0 => vec![1u8],
            1 => vec![1u8, 2, 3],
            2 => vec![b'R', b'G', b'B'],
            3 => {
                let num_comp = bitstream.read_bits(2)? as u8 + 1;
                (0..num_comp)
                    .map(|_| bitstream.read_bits(8).map(|x| x as u8))
                    .collect::<Result<_, _>>()?
            }
            _ => unreachable!(),
        };
        let components = component_ids
            .into_iter()
            .map(|id| -> Result<_, Self::Error> {
                let q_idx = bitstream.read_bits(2)? as u8;
                Ok(Component { id, q_idx })
            })
            .collect::<Result<_, _>>()?;

        let num_huff = bitstream.read_u32(4, 2 + U(3), 10 + U(4), 26 + U(6))?;
        let huffman_codes = (0..num_huff)
            .map(|_| HuffmanCode::parse(bitstream, ()))
            .collect::<Result<_, _>>()?;

        let scan_info = (0..num_scans)
            .map(|_| ScanInfo::parse(bitstream, ()))
            .collect::<Result<_, _>>()?;
        let restart_interval = if has_dri { bitstream.read_bits(16)? } else { 0 };
        let scan_more_info = (0..num_scans)
            .map(|_| ScanMoreInfo::parse(bitstream, ()))
            .collect::<Result<_, _>>()?;

        let intermarker_lengths = (0..num_intermarkers)
            .map(|_| bitstream.read_bits(16))
            .collect::<Result<_, _>>()?;

        let tail_data_length = bitstream.read_u32(0, 1 + U(8), 257 + U(16), 65793 + U(22))?;

        let has_padding = bitstream.read_bool()?;
        let padding_bits = has_padding
            .then(|| Padding::parse(bitstream, ()))
            .transpose()?;

        Ok(Self {
            is_gray,
            markers,
            app_markers,
            com_lengths,
            quant_tables,
            components,
            huffman_codes,
            scan_info,
            restart_interval,
            scan_more_info,
            intermarker_lengths,
            tail_data_length,
            padding_bits,
        })
    }
}

impl JpegBitstreamHeader {
    fn app_data_len(&self) -> usize {
        self.app_markers
            .iter()
            .filter_map(|marker| (marker.ty == 0).then_some(marker.length as usize))
            .sum::<usize>()
    }

    fn com_data_len(&self) -> usize {
        self.com_lengths.iter().map(|&x| x as usize).sum::<usize>()
    }

    fn intermarker_data_len(&self) -> usize {
        self.intermarker_lengths
            .iter()
            .map(|&x| x as usize)
            .sum::<usize>()
    }

    fn expected_data_len(&self) -> usize {
        self.app_data_len()
            + self.com_data_len()
            + self.intermarker_data_len()
            + self.tail_data_length as usize
    }
}

#[derive(Debug)]
struct AppMarker {
    ty: u32,
    length: u32,
}

impl Bundle for AppMarker {
    type Error = jxl_bitstream::Error;

    fn parse(bitstream: &mut Bitstream, _: ()) -> Result<Self, Self::Error> {
        Ok(Self {
            ty: bitstream.read_u32(0, 1, 2 + U(1), 4 + U(2))?,
            length: bitstream.read_bits(16)? + 1,
        })
    }
}

#[derive(Debug)]
struct QuantTable {
    precision: u8,
    index: u8,
    is_last: bool,
}

impl Bundle for QuantTable {
    type Error = jxl_bitstream::Error;

    fn parse(bitstream: &mut Bitstream, _: ()) -> Result<Self, Self::Error> {
        Ok(Self {
            precision: bitstream.read_bits(1)? as u8,
            index: bitstream.read_bits(2)? as u8,
            is_last: bitstream.read_bool()?,
        })
    }
}

#[derive(Debug)]
struct Component {
    id: u8,
    q_idx: u8,
}

#[derive(Debug)]
struct ScanInfo {
    ss: u8,
    se: u8,
    al: u8,
    ah: u8,
    component_info: Vec<ScanComponentInfo>,
    #[allow(unused)]
    last_needed_pass: u8,
}

impl Bundle for ScanInfo {
    type Error = jxl_bitstream::Error;

    fn parse(bitstream: &mut Bitstream, _: ()) -> Result<Self, Self::Error> {
        let num_comps = bitstream.read_bits(2)? as u8 + 1;
        let ss = bitstream.read_bits(6)? as u8;
        let se = bitstream.read_bits(6)? as u8;
        let al = bitstream.read_bits(4)? as u8;
        let ah = bitstream.read_bits(4)? as u8;
        let component_info = (0..num_comps)
            .map(|_| ScanComponentInfo::parse(bitstream, ()))
            .collect::<Result<_, _>>()?;
        let last_needed_pass = bitstream.read_u32(0, 1, 2, 3 + U(3))? as u8;
        Ok(Self {
            ss,
            se,
            ah,
            al,
            component_info,
            last_needed_pass,
        })
    }
}

impl ScanInfo {
    fn num_comps(&self) -> u8 {
        self.component_info.len() as u8
    }
}

#[derive(Debug)]
struct ScanComponentInfo {
    comp_idx: u8,
    ac_tbl_idx: u8,
    dc_tbl_idx: u8,
}

impl Bundle for ScanComponentInfo {
    type Error = jxl_bitstream::Error;

    fn parse(bitstream: &mut Bitstream, _: ()) -> Result<Self, Self::Error> {
        Ok(Self {
            comp_idx: bitstream.read_bits(2)? as u8,
            ac_tbl_idx: bitstream.read_bits(2)? as u8,
            dc_tbl_idx: bitstream.read_bits(2)? as u8,
        })
    }
}

#[derive(Debug)]
struct ScanMoreInfo {
    reset_points: HashSet<u32>,
    extra_zero_runs: HashMap<u32, u32>,
}

impl Bundle for ScanMoreInfo {
    type Error = jxl_bitstream::Error;

    fn parse(bitstream: &mut Bitstream, _: ()) -> Result<Self, Self::Error> {
        let num_reset_points = bitstream.read_u32(0, 1 + U(2), 4 + U(4), 20 + U(16))?;
        let reset_points = (0..num_reset_points)
            .map(|_| bitstream.read_u32(0, 1 + U(3), 9 + U(5), 41 + U(28)))
            .collect::<Result<_, _>>()?;

        let num_extra_zero_runs = bitstream.read_u32(0, 1 + U(2), 4 + U(4), 20 + U(16))?;
        let extra_zero_runs = (0..num_extra_zero_runs)
            .map(|_| -> Result<_, jxl_bitstream::Error> {
                let ExtraZeroRun { num_runs, run_length } = ExtraZeroRun::parse(bitstream, ())?;
                Ok((run_length, num_runs))
            })
            .collect::<Result<_, _>>()?;

        Ok(Self {
            reset_points,
            extra_zero_runs,
        })
    }
}

#[derive(Debug)]
struct ExtraZeroRun {
    num_runs: u32,
    run_length: u32,
}

impl Bundle for ExtraZeroRun {
    type Error = jxl_bitstream::Error;

    fn parse(bitstream: &mut Bitstream, _: ()) -> Result<Self, Self::Error> {
        Ok(Self {
            num_runs: bitstream.read_u32(1, 2 + U(2), 5 + U(4), 20 + U(8))?,
            run_length: bitstream.read_u32(0, 1 + U(3), 9 + U(5), 41 + U(28))?,
        })
    }
}

#[derive(Debug)]
struct Padding {
    bits: Vec<u8>,
}

impl Bundle for Padding {
    type Error = jxl_bitstream::Error;

    fn parse(bitstream: &mut Bitstream, _: ()) -> Result<Self, Self::Error> {
        let num_bits = bitstream.read_bits(24)?;
        let full_bytes = num_bits / 8;
        let extra_bits = num_bits % 8;
        let mut bits = Vec::with_capacity(full_bytes as usize + (extra_bits != 0) as usize);
        for _ in 0..full_bytes {
            bits.push(bitstream.read_bits(8)? as u8);
        }
        bits.push(bitstream.read_bits(extra_bits as usize)? as u8);

        Ok(Self { bits })
    }
}

pub struct JpegBitstreamReconstructor<'jbrd, 'frame> {
    state: ReconstructionState,
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
}

struct ParsedFrameData {
    hf_global: HfGlobal,
    lf_groups: Vec<LfGroup<i16>>,
    pass_groups: Vec<[AlignedGrid<i32>; 3]>,
}

impl<'jbrd, 'frame> JpegBitstreamReconstructor<'jbrd, 'frame> {
    fn new(header: &'jbrd JpegBitstreamHeader, data: &'jbrd [u8], frame: &'frame Frame) -> Result<Self> {
        let com_data_start = header.app_data_len();
        let intermarker_data_start = com_data_start + header.com_data_len();
        let tail_data_start = intermarker_data_start + header.intermarker_data_len();

        let frame_header = frame.header();
        let lf_global = frame.try_parse_lf_global::<i16>().ok_or(Error::FrameDataIncomplete)??;
        let hf_global = frame.try_parse_hf_global(Some(&lf_global)).ok_or(Error::FrameDataIncomplete)??;

        let lf_global_vardct = lf_global.vardct.as_ref();
        let global_ma_config = lf_global.gmodular.ma_config();
        let lf_groups = (0..frame.header().num_lf_groups())
            .map(|lf_group_idx| {
                frame
                    .try_parse_lf_group(lf_global_vardct, global_ma_config, None, lf_group_idx)
                    .ok_or(Error::FrameDataIncomplete)
                    .and_then(|r| r.map_err(Error::FrameParse))
            })
            .collect::<Result<Vec<_>>>()?;

        let upsampling_shifts: [_; 3] = std::array::from_fn(|idx| jxl_modular::ChannelShift::from_jpeg_upsampling(frame_header.jpeg_upsampling, idx));

        let num_groups = frame_header.num_groups();
        let mut pass_groups = Vec::with_capacity(num_groups as usize);
        for group_idx in 0..num_groups {
            let (w, h) = frame_header.group_size_for(group_idx);

            pass_groups.push([1, 0, 2].map(|idx| {
                let (w, h) = upsampling_shifts[idx].shift_size((w, h));
                let w = (w + 7) & !7;
                let h = (h + 7) & !7;
                AlignedGrid::<i32>::with_alloc_tracker(w as usize, h as usize, None).unwrap()
            }));
        }

        for pass_idx in 0..frame_header.passes.num_passes {
            for group_idx in 0..num_groups {
                let pass_group = frame
                    .pass_group_bitstream(pass_idx, group_idx)
                    .ok_or(Error::FrameDataIncomplete)
                    .and_then(|r| r.map_err(Error::FrameParse))?;
                if pass_group.partial {
                    return Err(Error::FrameDataIncomplete);
                }
                let mut bitstream = pass_group.bitstream;

                let pass_group = &pass_groups[group_idx as usize];
                let hf_coeff_output = [1, 0, 2].map(|idx| unsafe {
                    pass_group[idx].as_subgrid().as_atomic_i32()
                });
                let lf_group_idx = frame_header.lf_group_idx_from_group_idx(group_idx);
                let params = PassGroupParams {
                    frame_header,
                    lf_group: &lf_groups[lf_group_idx as usize],
                    pass_idx,
                    group_idx,
                    global_ma_config,
                    modular: None,
                    vardct: Some(PassGroupParamsVardct {
                        lf_vardct: lf_global_vardct.unwrap(),
                        hf_global: &hf_global,
                        hf_coeff_output: &hf_coeff_output,
                    }),
                    allow_partial: false,
                    tracker: None,
                    pool: &jxl_threadpool::JxlThreadPool::none(),
                };

                decode_pass_group(&mut bitstream, params).map_err(Error::FrameParse)?;
            }
        }

        Ok(Self {
            state: ReconstructionState::Init,
            parsed: ParsedFrameData {
                hf_global,
                lf_groups,
                pass_groups,
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
        })
    }
}

impl JpegBitstreamReconstructor<'_, '_> {
    pub fn write(&mut self, mut writer: impl Write) -> Result<ReconstructionStatus> {
        if self.state == ReconstructionState::Init {
            self.state = ReconstructionState::Writing;
            writer.write_all(&[0xff, 0xd8]).map_err(Error::ReconstructionWrite)?;
        }

        if self.state == ReconstructionState::Done {
            return Ok(ReconstructionStatus::Done);
        }

        while self.marker_ptr < self.header.markers.len() {
            let status = self.process_next(&mut writer)?;
            self.marker_ptr += 1;
            if status != ReconstructionStatus::Done {
                return Ok(status);
            }
        }

        self.state = ReconstructionState::Done;
        Ok(ReconstructionStatus::Done)
    }

    fn process_next(&mut self, mut writer: impl Write) -> Result<ReconstructionStatus> {
        let marker = self.header.markers[self.marker_ptr];
        Ok(match marker {
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
                writer.write_all(&header).map_err(Error::ReconstructionWrite)?;

                let mut jpeg_upsampling_reordered = self.frame.header().jpeg_upsampling;
                jpeg_upsampling_reordered.swap(0, 1);

                for (idx, comp) in self.header.components.iter().enumerate() {
                    let sampling_factor = jpeg_upsampling_reordered.get(idx).copied().unwrap_or(0);
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

                ReconstructionStatus::Done
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

                ReconstructionStatus::Done
            }

            // RSTn
            0xd0..=0xd7 => {
                writer
                    .write_all(&[0xff, marker])
                    .map_err(Error::ReconstructionWrite)?;
                ReconstructionStatus::Done
            }

            // EOI
            0xd9 => {
                writer
                    .write_all(&[0xff, 0xd9])
                    .map_err(Error::ReconstructionWrite)?;
                writer
                    .write_all(self.tail_data)
                    .map_err(Error::ReconstructionWrite)?;
                ReconstructionStatus::Done
            }

            // SOS
            0xda => {
                let idx = self.scan_info_ptr;
                let si = &self.header.scan_info[idx];
                let smi = &self.header.scan_more_info[idx];
                self.scan_info_ptr += 1;

                let num_comps = si.num_comps();
                let header_len_bytes = (6 + 2 * num_comps as u16).to_be_bytes();
                let header = [0xff, 0xda, header_len_bytes[0], header_len_bytes[1], num_comps];
                writer.write_all(&header).map_err(Error::ReconstructionWrite)?;

                let comps = &si.component_info;
                for c in comps {
                    let id = self.header.components[c.comp_idx as usize].id;
                    let table = (c.dc_tbl_idx << 4) | c.ac_tbl_idx;
                    writer.write_all(&[id, table]).map_err(Error::ReconstructionWrite)?;
                }

                writer.write_all(&[si.ss, si.se, (si.ah << 4) | si.al]).map_err(Error::ReconstructionWrite)?;

                let jpeg_upsampling = self.frame.header().jpeg_upsampling;
                let jpeg_upsampling_ycbcr = [1, 0, 2].map(|idx| jpeg_upsampling[idx]);
                let upsampling_shifts_ycbcr: [_; 3] = std::array::from_fn(|idx| jxl_modular::ChannelShift::from_jpeg_upsampling(jpeg_upsampling_ycbcr, idx));

                let (hsamples, vsamples) = if num_comps == 1 {
                    (vec![1], vec![1])
                } else {
                    let hsamples = comps
                        .iter()
                        .map(|c| [1u32, 2, 2, 1][jpeg_upsampling_ycbcr[c.comp_idx as usize] as usize])
                        .collect::<Vec<_>>();
                    let vsamples = comps
                        .iter()
                        .map(|c| [1u32, 2, 1, 2][jpeg_upsampling_ycbcr[c.comp_idx as usize] as usize])
                        .collect::<Vec<_>>();
                    (hsamples, vsamples)
                };
                let params = ScanParams {
                    si,
                    smi,
                    upsampling_shifts_ycbcr,
                    hsamples,
                    vsamples,
                };

                if !self.is_progressive {
                    self.process_sequential_scan(params, writer)?;
                } else if si.ah == 0 {
                    self.process_progressive_scan(params, writer)?;
                } else {
                    self.process_refinement_scan(params, writer)?;
                }

                ReconstructionStatus::Done
            }

            // DQT
            0xdb => {
                let hf_global = &self.parsed.hf_global;

                let last_idx = self.quant_ptr.iter().position(|qt| qt.is_last);
                let num_tables = last_idx.expect("is_last not found") + 1;
                let (qts, remainder) = self.quant_ptr.split_at(num_tables);
                self.quant_ptr = remainder;

                let encoded_len = 2 + 65 * num_tables + 64 * qts.iter().filter(|qt| qt.precision != 0).count();
                let mut header = [0xff, 0xdb, 0, 0];
                header[2..].copy_from_slice(&(encoded_len as u16).to_be_bytes());
                writer
                    .write_all(&header)
                    .map_err(Error::ReconstructionWrite)?;

                for qt in qts {
                    let channel = self.header.components.iter().position(|c| c.q_idx == qt.index);
                    let q = channel.and_then(|mut channel| {
                        if channel <= 1 {
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
                        for (&(y, x), q_val) in std::iter::zip(jxl_vardct::DCT8_NATURAL_ORDER, q_val) {
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

                    writer
                        .write_all(&buf)
                        .map_err(Error::ReconstructionWrite)?;
                }

                ReconstructionStatus::Done
            }

            // DRI
            0xdd => {
                let interval = (self.header.restart_interval as u16).to_be_bytes();
                let bytes = [0xff, 0xdd, 0, 4, interval[0], interval[1]];
                writer.write_all(&bytes).map_err(Error::ReconstructionWrite)?;
                if self.header.restart_interval != 0 {
                    self.restart_interval = Some(self.header.restart_interval);
                }
                ReconstructionStatus::Done
            }

            // APPn
            0xe0..=0xef => {
                let am = self.app_marker_ptr.next().ok_or(Error::InvalidData)?;
                let encoded_len = ((am.length - 1) as u16).to_be_bytes();
                match am.ty {
                    0 => {
                        writer.write_all(&[0xff]).map_err(Error::ReconstructionWrite)?;
                        let (app_data, next) = self.app_data.split_at(am.length as usize);
                        self.app_data = next;
                        writer.write_all(app_data).map_err(Error::ReconstructionWrite)?;
                        ReconstructionStatus::Done
                    }
                    1 => {
                        let header = [0xff, 0xe2, encoded_len[0], encoded_len[1]];
                        writer.write_all(&header).map_err(Error::ReconstructionWrite)?;
                        writer.write_all(b"ICC_PROFILE\0").map_err(Error::ReconstructionWrite)?;
                        let curr = self.next_icc_marker as u8 + 1;
                        let total = self.num_icc_markers as u8;
                        writer.write_all(&[curr, total]).map_err(Error::ReconstructionWrite)?;

                        let from = self.icc_marker_offset;
                        let len = am.length as usize - 17;
                        self.next_icc_marker += 1;
                        self.icc_marker_offset += len;
                        ReconstructionStatus::WriteIcc { from, len }
                    }
                    2 => {
                        let header = [0xff, 0xe0, encoded_len[0], encoded_len[1]];
                        writer.write_all(&header).map_err(Error::ReconstructionWrite)?;
                        writer.write_all(b"Exif\0\0").map_err(Error::ReconstructionWrite)?;
                        ReconstructionStatus::WriteExif
                    }
                    3 => {
                        let header = [0xff, 0xe0, encoded_len[0], encoded_len[1]];
                        writer.write_all(&header).map_err(Error::ReconstructionWrite)?;
                        writer.write_all(b"http://ns.adobe.com/xap/1.0/").map_err(Error::ReconstructionWrite)?;
                        ReconstructionStatus::WriteXml
                    }
                    _ => unreachable!(),
                }
            }

            // COM
            0xfe => {
                let length = *self.com_length.next().ok_or(Error::InvalidData)?;
                let (com_data, next) = self.com_data.split_at(length as usize);
                self.com_data = next;
                writer.write_all(&[0xff, 0xfe]).map_err(Error::ReconstructionWrite)?;
                writer.write_all(com_data).map_err(Error::ReconstructionWrite)?;
                ReconstructionStatus::Done
            }

            // Unrecognized
            0xff => {
                let length = *self.intermarker_length.next().ok_or(Error::InvalidData)?;
                let (data, next) = self.intermarker_data.split_at(length as usize);
                self.intermarker_data = next;
                writer.write_all(data).map_err(Error::ReconstructionWrite)?;
                ReconstructionStatus::Done
            }

            _ => return Err(Error::InvalidData),
        })
    }

    fn process_sequential_scan(&mut self, params: ScanParams, mut writer: impl Write) -> Result<()> {
        let ScanParams { si, smi, upsampling_shifts_ycbcr, hsamples, vsamples } = params;
        let comps = &si.component_info;
        let frame_header = self.frame.header();

        let max_hsample = hsamples.iter().copied().max().unwrap().trailing_zeros();
        let max_vsample = vsamples.iter().copied().max().unwrap().trailing_zeros();
        let w8 = (frame_header.width.div_ceil(8) + max_hsample) >> max_hsample;
        let h8 = (frame_header.height.div_ceil(8) + max_vsample) >> max_vsample;

        let group_dim = frame_header.group_dim();

        let mut state = ScanState::new(comps.len());
        let mut block_idx = 0u32;
        for y8 in 0..h8 {
            for x8 in 0..w8 {
                let mcu_idx = x8 + w8 * y8;

                let group_idx = frame_header.group_idx_from_coord(x8 << (3 + max_hsample), y8 << (3 + max_vsample)).unwrap();
                let lf_group_idx = frame_header.lf_group_idx_from_group_idx(group_idx);

                let hf_coeff = self.parsed.pass_groups[group_idx as usize].each_ref().map(|g| g.as_subgrid());
                let lf_quant = &self.parsed.lf_groups[lf_group_idx as usize].lf_coeff.as_ref().unwrap().lf_quant.image().unwrap().image_channels();

                for ((cidx, c), (&hs, &vs)) in comps.iter().enumerate().zip(std::iter::zip(&hsamples, &vsamples)) {
                    let dc_table = self
                        .dc_tables[c.dc_tbl_idx as usize]
                        .as_ref()
                        .ok_or(Error::InvalidData)?;
                    let ac_table = self
                        .ac_tables[c.ac_tbl_idx as usize]
                        .as_ref()
                        .ok_or(Error::InvalidData)?;

                    let idx = c.comp_idx as usize;
                    let lf_quant = &lf_quant[idx];
                    let hf_coeff = hf_coeff[idx];
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

                            let hf_coeff = hf_coeff.subgrid(x_ac_start..(x_ac_start + 8), y_ac_start..(y_ac_start + 8));

                            let coeff = *lf_quant.get(x_dc, y_dc).unwrap();
                            let diff = state.update_dc_pred(cidx, coeff);

                            let is_neg = diff < 0;
                            let bits = if is_neg { -diff } else { diff };
                            let bitlen = 16 - bits.leading_zeros();
                            let raw_bits = if is_neg { -bits - 1 } else { bits };

                            let (len, bits) = dc_table.lookup(bitlen as u8);
                            state.bit_writer.write_huffman(bits, len);
                            state.bit_writer.write_raw(raw_bits as u64, bitlen as u8);

                            let mut ac_coeffs: Vec<i16> = Vec::with_capacity(63);
                            for &(x, y) in &jxl_vardct::DCT8_NATURAL_ORDER[1..] {
                                let coeff = *hf_coeff.get(x as usize, y as usize) as i16;
                                ac_coeffs.push(coeff);
                            }

                            let mut remaining = &*ac_coeffs;
                            while let Some(mut nonzero_idx) = remaining.iter().position(|x| *x != 0) {
                                let coeff = remaining[nonzero_idx];
                                remaining = &remaining[nonzero_idx + 1..];

                                while nonzero_idx >= 16 {
                                    let (len, bits) = ac_table.lookup(0xf0);
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
                                let (len, bits) = ac_table.lookup(sym);
                                state.bit_writer.write_huffman(bits, len);
                                state.bit_writer.write_raw(raw_bits as u64, bitlen as u8);
                            }

                            let mut num_zeros = remaining.len() as i32;
                            if let Some(&ezr) = smi.extra_zero_runs.get(&block_idx) {
                                let (len, bits) = ac_table.lookup(0xf0);
                                for _ in 0..ezr {
                                    state.bit_writer.write_huffman(bits, len);
                                }

                                num_zeros -= ezr as i32 * 16;
                            }

                            if num_zeros > 0 {
                                let (len, bits) = ac_table.lookup(0);
                                state.bit_writer.write_huffman(bits, len);
                            }

                            block_idx += 1;
                        }
                    }
                }

                if let Some(restart_interval) = self.restart_interval {
                    if (mcu_idx + 1) % restart_interval == 0 {
                        state.restart(self.padding_bitstream.as_mut(), &mut writer)?;
                    }
                }
            }
        }

        state.flush_bit_writer(self.padding_bitstream.as_mut(), &mut writer)?;

        Ok(())
    }

    fn process_progressive_scan(&mut self, params: ScanParams, mut writer: impl Write) -> Result<()> {
        let ScanParams { si, smi, upsampling_shifts_ycbcr, hsamples, vsamples } = params;
        let comps = &si.component_info;
        let frame_header = self.frame.header();

        let max_hsample = hsamples.iter().copied().max().unwrap().trailing_zeros();
        let max_vsample = vsamples.iter().copied().max().unwrap().trailing_zeros();
        let w8 = (frame_header.width.div_ceil(8) + max_hsample) >> max_hsample;
        let h8 = (frame_header.height.div_ceil(8) + max_vsample) >> max_vsample;

        let ss = si.ss.max(1);
        let se = si.se + 1;
        let al = si.al;

        let group_dim = frame_header.group_dim();

        let mut state = ScanState::new(comps.len());
        let mut block_idx = 0u32;
        for y8 in 0..h8 {
            for x8 in 0..w8 {
                let mcu_idx = x8 + w8 * y8;

                let group_idx = frame_header.group_idx_from_coord(x8 << (3 + max_hsample), y8 << (3 + max_vsample)).unwrap();
                let lf_group_idx = frame_header.lf_group_idx_from_group_idx(group_idx);

                let hf_coeff = self.parsed.pass_groups[group_idx as usize].each_ref().map(|g| g.as_subgrid());
                let lf_quant = &self.parsed.lf_groups[lf_group_idx as usize].lf_coeff.as_ref().unwrap().lf_quant.image().unwrap().image_channels();

                let mut last_ac_table = None;
                for ((cidx, c), (&hs, &vs)) in comps.iter().enumerate().zip(std::iter::zip(&hsamples, &vsamples)) {
                    let dc_table = self
                        .dc_tables[c.dc_tbl_idx as usize]
                        .as_ref()
                        .ok_or(Error::InvalidData)?;
                    let ac_table = self
                        .ac_tables[c.ac_tbl_idx as usize]
                        .as_ref()
                        .ok_or(Error::InvalidData)?;
                    if last_ac_table.is_none() {
                        last_ac_table = Some(ac_table);
                    }

                    let idx = c.comp_idx as usize;
                    let lf_quant = &lf_quant[idx];
                    let hf_coeff = hf_coeff[idx];
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

                            let hf_coeff = hf_coeff.subgrid(x_ac_start..(x_ac_start + 8), y_ac_start..(y_ac_start + 8));

                            if si.ss == 0 {
                                let coeff = *lf_quant.get(x_dc, y_dc).unwrap();
                                let coeff = coeff >> al;
                                let diff = state.update_dc_pred(cidx, coeff);

                                let is_neg = diff < 0;
                                let bits = if is_neg { -diff } else { diff };
                                let bitlen = 16 - bits.leading_zeros();
                                let raw_bits = if is_neg { -bits - 1 } else { bits };

                                state.emit_eobrun(last_ac_table.unwrap());
                                let (len, bits) = dc_table.lookup(bitlen as u8);
                                state.bit_writer.write_huffman(bits, len);
                                state.bit_writer.write_raw(raw_bits as u64, bitlen as u8);

                                last_ac_table = Some(ac_table);
                            }

                            let mut ac_coeffs: Vec<i16> = Vec::with_capacity((se - ss) as usize);
                            for &(x, y) in &jxl_vardct::DCT8_NATURAL_ORDER[ss as usize..se as usize] {
                                let coeff = *hf_coeff.get(x as usize, y as usize) as i16;
                                let coeff = if coeff < 0 {
                                    -((-coeff) >> al)
                                } else {
                                    coeff >> al
                                };
                                ac_coeffs.push(coeff);
                            }

                            let mut remaining = &*ac_coeffs;
                            while let Some(mut nonzero_idx) = remaining.iter().position(|x| *x != 0) {
                                state.emit_eobrun(last_ac_table.unwrap());

                                let coeff = remaining[nonzero_idx];
                                remaining = &remaining[nonzero_idx + 1..];

                                while nonzero_idx >= 16 {
                                    let (len, bits) = ac_table.lookup(0xf0);
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
                                let (len, bits) = ac_table.lookup(sym);
                                state.bit_writer.write_huffman(bits, len);
                                state.bit_writer.write_raw(raw_bits as u64, bitlen as u8);

                                last_ac_table = Some(ac_table);
                            }

                            let mut num_zeros = remaining.len() as i32;
                            if let Some(&ezr) = smi.extra_zero_runs.get(&block_idx) {
                                state.emit_eobrun(last_ac_table.unwrap());
                                let (len, bits) = ac_table.lookup(0xf0);
                                for _ in 0..ezr {
                                    state.bit_writer.write_huffman(bits, len);
                                }

                                num_zeros -= ezr as i32 * 16;
                                last_ac_table = Some(ac_table);
                            }

                            if num_zeros > 0 {
                                state.eobrun += 1;
                                if state.eobrun >= 32767 {
                                    state.emit_eobrun(last_ac_table.unwrap());
                                }
                            }

                            block_idx += 1;
                            if smi.reset_points.contains(&block_idx) {
                                state.emit_eobrun(last_ac_table.unwrap());
                            }
                        }
                    }
                }

                if let Some(restart_interval) = self.restart_interval {
                    if (mcu_idx + 1) % restart_interval == 0 {
                        state.emit_eobrun(last_ac_table.unwrap());
                        state.restart(self.padding_bitstream.as_mut(), &mut writer)?;
                    }
                }
            }
        }

        state.flush_bit_writer(self.padding_bitstream.as_mut(), &mut writer)?;

        Ok(())
    }

    fn process_refinement_scan(&mut self, params: ScanParams, mut writer: impl Write) -> Result<()> {
        let ScanParams { si, smi, upsampling_shifts_ycbcr, hsamples, vsamples } = params;
        let comps = &si.component_info;
        let frame_header = self.frame.header();

        let max_hsample = hsamples.iter().copied().max().unwrap().trailing_zeros();
        let max_vsample = vsamples.iter().copied().max().unwrap().trailing_zeros();
        let w8 = (frame_header.width.div_ceil(8) + max_hsample) >> max_hsample;
        let h8 = (frame_header.height.div_ceil(8) + max_vsample) >> max_vsample;

        let ss = si.ss.max(1);
        let se = si.se + 1;
        let al = si.al;

        let group_dim = frame_header.group_dim();

        let mut state = ScanState::new(comps.len());
        let mut block_idx = 0u32;
        for y8 in 0..h8 {
            for x8 in 0..w8 {
                let mcu_idx = x8 + w8 * y8;

                let group_idx = frame_header.group_idx_from_coord(x8 << (3 + max_hsample), y8 << (3 + max_vsample)).unwrap();
                let lf_group_idx = frame_header.lf_group_idx_from_group_idx(group_idx);

                let hf_coeff = self.parsed.pass_groups[group_idx as usize].each_ref().map(|g| g.as_subgrid());
                let lf_quant = &self.parsed.lf_groups[lf_group_idx as usize].lf_coeff.as_ref().unwrap().lf_quant.image().unwrap().image_channels();

                let mut last_ac_table = None;
                for (c, (&hs, &vs)) in comps.iter().zip(std::iter::zip(&hsamples, &vsamples)) {
                    let ac_table = self
                        .ac_tables[c.ac_tbl_idx as usize]
                        .as_ref()
                        .ok_or(Error::InvalidData)?;
                    if last_ac_table.is_none() {
                        last_ac_table = Some(ac_table);
                    }

                    let idx = c.comp_idx as usize;
                    let lf_quant = &lf_quant[idx];
                    let hf_coeff = hf_coeff[idx];
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

                            let hf_coeff = hf_coeff.subgrid(x_ac_start..(x_ac_start + 8), y_ac_start..(y_ac_start + 8));

                            if si.ss == 0 {
                                state.emit_eobrun(last_ac_table.unwrap());
                                let coeff = *lf_quant.get(x_dc, y_dc).unwrap();
                                let coeff = coeff >> al;
                                state.bit_writer.write_raw(coeff as u64, 1);
                                last_ac_table = Some(ac_table);
                            }

                            let mut ac_coeffs = Vec::with_capacity((se - ss) as usize);
                            for &(x, y) in &jxl_vardct::DCT8_NATURAL_ORDER[ss as usize..se as usize] {
                                let coeff = *hf_coeff.get(x as usize, y as usize) as i16;
                                let coeff = if coeff < 0 {
                                    -((-coeff) >> al)
                                } else {
                                    coeff >> al
                                };
                                ac_coeffs.push(coeff);
                            }

                            let mut remaining = &*ac_coeffs;
                            while let Some(nonzero_idx) = remaining.iter().position(|x| *x == 1 || *x == -1) {
                                state.emit_eobrun(last_ac_table.unwrap());

                                let mut zero_runs = 0u8;
                                let mut refinement_bitlen = 0u8;
                                let mut refinement_bits = 0u64;
                                for &coeff in &remaining[..nonzero_idx] {
                                    if coeff == 0 {
                                        zero_runs += 1;
                                        if zero_runs == 16 {
                                            let (len, bits) = ac_table.lookup(0xf0);
                                            state.bit_writer.write_huffman(bits, len);
                                            state.bit_writer.write_raw(refinement_bits, refinement_bitlen);
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
                                let (len, bits) = ac_table.lookup(sym);
                                state.bit_writer.write_huffman(bits, len);
                                state.bit_writer.write_raw(bit, 1);
                                state.bit_writer.write_raw(refinement_bits, refinement_bitlen);
                                last_ac_table = Some(ac_table);
                            }

                            let mut remaining_zrl = smi.extra_zero_runs.get(&block_idx).copied().unwrap_or(0);
                            if remaining_zrl > 0 {
                                state.emit_eobrun(last_ac_table.unwrap());
                                last_ac_table = Some(ac_table);
                            }

                            let (zrl_len, zrl_bits) = ac_table.lookup(0xf0);
                            let mut zero_runs = 0u8;
                            let mut refinement_bitlen = 0u8;
                            let mut refinement_bits = 0u64;
                            for &coeff in remaining {
                                if coeff == 0 {
                                    zero_runs += 1;
                                    if remaining_zrl > 0 && zero_runs == 16 {
                                        state.bit_writer.write_huffman(zrl_bits, zrl_len);
                                        state.bit_writer.write_raw(refinement_bits, refinement_bitlen);
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
                                state.bit_writer.write_raw(refinement_bits, refinement_bitlen);
                                zero_runs = 0;
                                refinement_bitlen = 0;
                            }

                            if zero_runs > 0 || refinement_bitlen > 0 {
                                state.eobrun += 1;
                                state.buffer_refinement_bits(refinement_bits, refinement_bitlen);
                                if state.eobrun >= 32767 {
                                    state.emit_eobrun(last_ac_table.unwrap());
                                }
                            }

                            block_idx += 1;
                            if smi.reset_points.contains(&block_idx) {
                                state.emit_eobrun(last_ac_table.unwrap());
                            }
                        }
                    }
                }

                if let Some(restart_interval) = self.restart_interval {
                    if (mcu_idx + 1) % restart_interval == 0 {
                        state.emit_eobrun(last_ac_table.unwrap());
                        state.restart(self.padding_bitstream.as_mut(), &mut writer)?;
                    }
                }
            }
        }

        state.flush_bit_writer(self.padding_bitstream.as_mut(), &mut writer)?;

        Ok(())
    }
}

struct ScanParams<'jbrd> {
    si: &'jbrd ScanInfo,
    smi: &'jbrd ScanMoreInfo,
    upsampling_shifts_ycbcr: [ChannelShift; 3],
    hsamples: Vec<u32>,
    vsamples: Vec<u32>,
}

struct ScanState {
    bit_writer: bit_writer::BitWriter,
    dc_pred: Vec<i16>,
    eobrun: u32,
    refinement_bitlen: Vec<u8>,
    refinement_bits: Vec<u64>,
    rst_m: u8,
}

impl ScanState {
    fn new(num_comps: usize) -> Self {
        Self {
            bit_writer: bit_writer::BitWriter::new(),
            dc_pred: vec![0; num_comps],
            eobrun: 0,
            refinement_bitlen: Vec::new(),
            refinement_bits: Vec::new(),
            rst_m: 0,
        }
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

    fn emit_eobrun(&mut self, ac_table: &huffman::BuiltHuffmanTable) {
        if self.eobrun == 0 {
            return;
        }

        let eobn = 31 - self.eobrun.leading_zeros();
        let (len, bits) = ac_table.lookup((eobn as u8) << 4);
        self.bit_writer.write_huffman(bits, len);
        let mask = (1u32 << eobn) - 1;
        self.bit_writer.write_raw((self.eobrun & mask) as u64, eobn as u8);
        self.eobrun = 0;

        let refinement_bits = std::mem::take(&mut self.refinement_bits);
        let refinement_bitlen = std::mem::take(&mut self.refinement_bitlen);
        for (bits, bitlen) in std::iter::zip(refinement_bits, refinement_bitlen) {
            self.bit_writer.write_raw(bits, bitlen);
        }
    }

    fn flush_bit_writer(&mut self, padding_bitstream: Option<&mut Bitstream>, mut writer: impl Write) -> Result<()> {
        let mut bit_writer = std::mem::replace(&mut self.bit_writer, BitWriter::new());
        let padding_needed = bit_writer.padding_bits();
        if padding_needed != 0 {
            let bits = if let Some(padding_bitstream) = padding_bitstream {
                padding_bitstream.read_bits(padding_needed).map_err(|_| Error::InvalidData)?
            } else {
                (-1i32) as u32
            };
            bit_writer.write_raw(bits as u64, padding_needed as u8);
        }

        let bytes = bit_writer.finalize();
        writer.write_all(&bytes).map_err(Error::ReconstructionWrite)?;

        Ok(())
    }

    fn restart(&mut self, padding_bitstream: Option<&mut Bitstream>, mut writer: impl Write) -> Result<()> {
        assert_eq!(self.eobrun, 0);
        self.dc_pred.fill(0);
        self.flush_bit_writer(padding_bitstream, &mut writer)?;

        let rst = [0xff, 0xd0 + self.rst_m];
        writer.write_all(&rst).map_err(Error::ReconstructionWrite)?;
        self.rst_m = (self.rst_m + 1) % 8;

        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReconstructionState {
    Init,
    Writing,
    Done,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReconstructionStatus {
    Done,
    WriteIcc {
        from: usize,
        len: usize,
    },
    WriteExif,
    WriteXml,
}
