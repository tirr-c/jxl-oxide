use std::io::{Cursor, Write};

use brotli_decompressor::DecompressorWriter;
use jxl_bitstream::{Bitstream, U};
use jxl_frame::Frame;
use jxl_oxide_common::Bundle;

mod huffman;

use huffman::HuffmanCode;

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

    pub fn is_complete(&mut self) -> Result<bool> {
        self.data_stream.flush().map_err(Error::Brotli)?;
        let decompressed_len = self.data_stream.get_ref().len();
        Ok(decompressed_len >= self.header.expected_data_len())
    }

    pub fn reconstruct<'jbrd, 'frame>(
        &'jbrd mut self,
        frame: &'frame Frame,
    ) -> Result<JpegBitstreamReconstructor<'jbrd, 'frame>> {
        let Self {
            ref header,
            ref mut data_stream,
        } = *self;
        data_stream.close().map_err(Error::Brotli)?;
        Ok(JpegBitstreamReconstructor::new(
            header,
            data_stream.get_ref(),
            frame,
        ))
    }

    pub fn header(&self) -> &JpegBitstreamHeader {
        &self.header
    }
}

#[derive(Debug)]
pub struct JpegBitstreamHeader {
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
        dbg!(&app_markers);
        let com_lengths = (0..num_com_markers)
            .map(|_| bitstream.read_bits(16).map(|x| x + 1))
            .collect::<Result<_, _>>()?;
        dbg!(&com_lengths);

        let num_quant_tables = bitstream.read_bits(2)? + 1;
        let quant_tables = (0..num_quant_tables)
            .map(|_| QuantTable::parse(bitstream, ()))
            .collect::<Result<_, _>>()?;
        dbg!(&quant_tables);

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
        dbg!(&components);

        let num_huff = bitstream.read_u32(4, 2 + U(3), 10 + U(4), 26 + U(6))?;
        let huffman_codes = (0..num_huff)
            .map(|_| HuffmanCode::parse(bitstream, ()))
            .collect::<Result<_, _>>()?;
        dbg!(&huffman_codes);

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

    fn find_channel_id(&self, component_id: u8) -> Option<usize> {
        self.components.iter().position(|c| c.id == component_id)
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
    reset_points: Vec<u32>,
    extra_zero_runs: Vec<ExtraZeroRun>,
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
            .map(|_| ExtraZeroRun::parse(bitstream, ()))
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
    num_bits: u32,
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

        Ok(Self { num_bits, bits })
    }
}

pub struct JpegBitstreamReconstructor<'jbrd, 'frame> {
    header: &'jbrd JpegBitstreamHeader,
    frame: &'frame Frame,
    marker_ptr: usize,
    app_data: Cursor<&'jbrd [u8]>,
    com_data: Cursor<&'jbrd [u8]>,
    intermarker_data: Cursor<&'jbrd [u8]>,
    huffman_code_ptr: &'jbrd [HuffmanCode],
    quant_ptr: &'jbrd [QuantTable],
    last_quant_val: Option<Box<[u16; 64]>>,
    padding_bitstream: Option<Bitstream<'jbrd>>,
    scan_info_ptr: usize,
    tail_data: &'jbrd [u8],
}

impl<'jbrd, 'frame> JpegBitstreamReconstructor<'jbrd, 'frame> {
    fn new(header: &'jbrd JpegBitstreamHeader, data: &'jbrd [u8], frame: &'frame Frame) -> Self {
        let com_data_start = header.app_data_len();
        let intermarker_data_start = com_data_start + header.com_data_len();
        let tail_data_start = intermarker_data_start + header.intermarker_data_len();
        Self {
            header,
            frame,
            marker_ptr: 0,
            app_data: Cursor::new(&data[..com_data_start]),
            com_data: Cursor::new(&data[com_data_start..intermarker_data_start]),
            intermarker_data: Cursor::new(&data[intermarker_data_start..tail_data_start]),
            huffman_code_ptr: &header.huffman_codes,
            quant_ptr: &header.quant_tables,
            last_quant_val: None,
            padding_bitstream: header
                .padding_bits
                .as_ref()
                .map(|padding| Bitstream::new(&padding.bits)),
            scan_info_ptr: 0,
            tail_data: &data[tail_data_start..],
        }
    }
}

impl JpegBitstreamReconstructor<'_, '_> {
    pub fn write(&mut self, mut writer: impl Write) -> Result<ReconstructionStatus> {
        while self.marker_ptr < self.header.markers.len() {
            let status = self.process_next(&mut writer)?;
            self.marker_ptr += 1;
            if status != ReconstructionStatus::Done {
                return Ok(status);
            }
        }

        Ok(ReconstructionStatus::Done)
    }

    fn process_next(&mut self, mut writer: impl Write) -> Result<ReconstructionStatus> {
        let marker = self.header.markers[self.marker_ptr];
        Ok(match marker {
            // SOF
            0xc0 | 0xc1 | 0xc2 | 0xc9 | 0xca => {
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
                        .write_all(&hc.values)
                        .map_err(Error::ReconstructionWrite)?;
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
                todo!()
            }

            // DQT
            0xdb => {
                let lf_global = self.frame.try_parse_lf_global::<i16>().ok_or(Error::FrameDataIncomplete)??;
                let hf_global = self.frame.try_parse_hf_global(Some(&lf_global)).ok_or(Error::FrameDataIncomplete)??;

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
                    let q = channel.and_then(|channel| hf_global.dequant_matrices.jpeg_quant_values(channel));
                    if let Some(q) = q {
                        if self.last_quant_val.is_none() {
                            self.last_quant_val = Some(Box::new([0; 64]));
                        }
                        let q_val = self.last_quant_val.as_deref_mut().unwrap();

                        for (&(r, c), q_val) in std::iter::zip(jxl_vardct::DCT8_NATURAL_ORDER, q_val) {
                            *q_val = q[r as usize + 8 * c as usize] as u16
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
                todo!()
            }

            // APPn
            0xe0..=0xef => {
                todo!()
            }

            // COM
            0xfe => {
                todo!()
            }

            // Unrecognized
            0xff => {
                todo!()
            }

            _ => return Err(Error::InvalidData),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReconstructionStatus {
    Done,
    WriteIcc,
    WriteExif,
    WriteXml,
}