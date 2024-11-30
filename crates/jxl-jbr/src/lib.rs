use std::collections::{HashMap, HashSet};
use std::io::Write;

use brotli_decompressor::DecompressorWriter;
use jxl_bitstream::{Bitstream, U};
use jxl_frame::Frame;
use jxl_oxide_common::Bundle;

use crate::huffman::HuffmanCode;

mod bit_writer;
mod error;
mod huffman;
mod reconstruct;

pub use error::Error;
pub use reconstruct::JpegBitstreamReconstructor;

use error::Result;

const HEADER_ICC: &[u8] = b"ICC_PROFILE\0";
const HEADER_EXIF: &[u8] = b"Exif\0\0";
const HEADER_XMP: &[u8] = b"http://ns.adobe.com/xap/1.0/\0";

pub struct JpegBitstreamData {
    header: Box<JpegBitstreamHeader>,
    data_stream: Box<DecompressorWriter<Vec<u8>>>,
}

impl std::fmt::Debug for JpegBitstreamData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JpegBitstreamData")
            .field("header", &self.header)
            .finish_non_exhaustive()
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
            tracing::error!(
                decompressed_len,
                expected = self.header.expected_data_len(),
                "Data section length of jbrd box doesn't match expected length"
            );
            return Err(Error::InvalidData);
        }

        Ok(())
    }

    pub fn is_complete(&mut self) -> Result<bool> {
        self.data_stream.flush().map_err(Error::Brotli)?;
        let decompressed_len = self.data_stream.get_ref().len();
        Ok(decompressed_len >= self.header.expected_data_len())
    }

    pub fn reconstruct<'jbrd, 'frame, 'meta>(
        &'jbrd self,
        frame: &'frame Frame,
        icc_profile: &'meta [u8],
        exif: &'meta [u8],
        xmp: &'meta [u8],
        pool: &jxl_threadpool::JxlThreadPool,
    ) -> Result<JpegBitstreamReconstructor<'jbrd, 'frame, 'meta>> {
        let Self {
            ref header,
            ref data_stream,
        } = *self;
        JpegBitstreamReconstructor::new(
            header,
            data_stream.get_ref(),
            frame,
            icc_profile,
            exif,
            xmp,
            pool,
        )
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

    pub fn expected_icc_len(&self) -> usize {
        self.app_markers
            .iter()
            .filter(|am| am.ty == 1)
            .map(|am| am.length as usize - 5 - HEADER_ICC.len())
            .sum::<usize>()
    }

    pub fn expected_exif_len(&self) -> usize {
        self.app_markers
            .iter()
            .find(|am| am.ty == 2)
            .map(|am| am.length as usize - 3 - HEADER_EXIF.len())
            .unwrap_or(0)
    }

    pub fn expected_xmp_len(&self) -> usize {
        self.app_markers
            .iter()
            .find(|am| am.ty == 3)
            .map(|am| am.length as usize - 3 - HEADER_XMP.len())
            .unwrap_or(0)
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
        let mut last_block_idx: Option<u32> = None;
        let reset_points = (0..num_reset_points)
            .map(|_| -> Result<_, Self::Error> {
                let diff = bitstream.read_u32(0, 1 + U(3), 9 + U(5), 41 + U(28))?;
                let block_idx = if let Some(last_block_idx) = last_block_idx {
                    last_block_idx.saturating_add(diff + 1)
                } else {
                    diff
                };
                if block_idx > (3 << 26) {
                    tracing::error!(value = block_idx, "reset_points too large");
                    return Err(jxl_bitstream::Error::ValidationFailed(
                        "reset_points too large",
                    ));
                }
                last_block_idx = Some(block_idx);
                Ok(block_idx)
            })
            .collect::<Result<_, _>>()?;

        let num_extra_zero_runs = bitstream.read_u32(0, 1 + U(2), 4 + U(4), 20 + U(16))?;
        let mut last_block_idx: Option<u32> = None;
        let extra_zero_runs = (0..num_extra_zero_runs)
            .map(|_| -> Result<_, jxl_bitstream::Error> {
                let ExtraZeroRun {
                    num_runs,
                    run_length,
                } = ExtraZeroRun::parse(bitstream, ())?;
                let block_idx = if let Some(last_block_idx) = last_block_idx {
                    last_block_idx.saturating_add(run_length + 1)
                } else {
                    run_length
                };
                if block_idx > (3 << 26) {
                    tracing::error!(block_idx, "extra_zero_runs.block_idx too large");
                    return Err(jxl_bitstream::Error::ValidationFailed(
                        "extra_zero_runs.block_idx too large",
                    ));
                }
                last_block_idx = Some(block_idx);
                Ok((block_idx, num_runs))
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
