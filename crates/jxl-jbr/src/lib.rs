use std::io::{Cursor, Write};

use brotli_decompressor::DecompressorWriter;
use jxl_bitstream::{Bitstream, U};
use jxl_frame::Frame;
use jxl_oxide_common::Bundle;

#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    Bitstream(jxl_bitstream::Error),
    Brotli(std::io::Error),
    InvalidData,
    ReconstructionWrite(std::io::Error),
    ReconstructionDataIncomplete,
}

impl From<jxl_bitstream::Error> for Error {
    fn from(value: jxl_bitstream::Error) -> Self {
        Self::Bitstream(value)
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
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Bitstream(e) => Some(e),
            Error::Brotli(e) => Some(e),
            Error::ReconstructionWrite(e) => Some(e),
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
}

#[derive(Debug)]
struct JpegBitstreamHeader {
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
            let marker_bits = bitstream.read_bits(6)? as u8;
            match marker_bits {
                0xe0..=0xef => num_app_markers += 1,
                0xfe => num_com_markers += 1,
                0xda => num_scans += 1,
                0xff => num_intermarkers += 1,
                0xdd => has_dri = true,
                _ => {}
            }
            markers.push(0xc0 + marker_bits);
        }

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
struct HuffmanCode {
    is_ac: bool,
    is_last: bool,
    id_and_counts: [u8; 17],
    values: Vec<u8>,
}

impl HuffmanCode {
    fn encoded_len(&self) -> usize {
        1 + 16 + self.values.len()
    }
}

impl Bundle for HuffmanCode {
    type Error = jxl_bitstream::Error;

    fn parse(bitstream: &mut Bitstream, _: ()) -> Result<Self, Self::Error> {
        let is_ac = bitstream.read_bool()?;
        let id = bitstream.read_bits(2)? as u8;
        let is_last = bitstream.read_bool()?;

        let mut id_and_counts = [0u8; 17];
        id_and_counts[0] = id;
        let mut sum_counts = 0u32;
        for count in &mut id_and_counts[1..] {
            let x = bitstream.read_u32(0, 1, 2 + U(3), U(8))?;
            sum_counts += x;
            *count = x as u8;
        }
        let values = (0..sum_counts)
            .map(|_| {
                bitstream
                    .read_u32(U(2), 4 + U(2), 8 + U(4), 1 + U(8))
                    .map(|x| x as u8)
            })
            .collect::<Result<_, _>>()?;

        Ok(Self {
            is_ac,
            is_last,
            id_and_counts,
            values,
        })
    }
}

#[derive(Debug)]
struct ScanInfo {
    num_comps: u8,
    ss: u8,
    se: u8,
    ah: u8,
    al: u8,
    component_info: Vec<ScanComponentInfo>,
    last_needed_pass: u8,
}

impl Bundle for ScanInfo {
    type Error = jxl_bitstream::Error;

    fn parse(bitstream: &mut Bitstream, _: ()) -> Result<Self, Self::Error> {
        let num_comps = bitstream.read_bits(2)? as u8 + 1;
        let ss = bitstream.read_bits(6)? as u8;
        let se = bitstream.read_bits(6)? as u8;
        let ah = bitstream.read_bits(4)? as u8;
        let al = bitstream.read_bits(4)? as u8;
        let component_info = (0..num_comps)
            .map(|_| ScanComponentInfo::parse(bitstream, ()))
            .collect::<Result<_, _>>()?;
        let last_needed_pass = bitstream.read_u32(0, 1, 2, 3 + U(3))? as u8;
        Ok(Self {
            num_comps,
            ss,
            se,
            ah,
            al,
            component_info,
            last_needed_pass,
        })
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
                    writer
                        .write_all(&hc.id_and_counts)
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
                todo!()
            }

            // DQT
            0xdb => {
                todo!()
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
