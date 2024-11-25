use std::io::Write;

use jxl_oxide::{AuxBoxData, JxlImage};

use crate::commands::dump_jbrd::*;
use crate::{Error, Result};

pub fn handle_dump_jbrd(args: DumpJbrd) -> Result<()> {
    let _guard = tracing::trace_span!("Handle dump-jbrd subcommand").entered();

    let image = JxlImage::builder()
        .open(&args.input)
        .map_err(Error::ReadJxl)?;

    let Some(jbrd) = image.jbrd() else {
        println!("No reconstruction data available");
        return Ok(());
    };

    println!("{:#x?}", jbrd.header());

    let Some(output_path) = args.output_jpeg else {
        return Ok(());
    };

    let icc = image.original_icc();
    let aux_boxes = image.aux_boxes();
    let exif = aux_boxes.first_exif().map_err(Error::ReadJxl)?;
    let xmp = aux_boxes.first_xml();

    let frame = image.frame_by_keyframe(0).unwrap();
    let mut reconstructor = jbrd
        .reconstruct(frame)
        .map_err(|e| Error::Reconstruct(e.into()))?;

    let output = std::fs::File::create(output_path).map_err(Error::WriteImage)?;
    let mut output = std::io::BufWriter::new(output);
    loop {
        let status = reconstructor
            .write(&mut output)
            .map_err(|e| Error::Reconstruct(e.into()))?;
        match status {
            jxl_oxide::jpeg_bitstream::ReconstructionStatus::Done => break,
            jxl_oxide::jpeg_bitstream::ReconstructionStatus::WriteIcc { from, len } => {
                let icc = icc.unwrap();
                let chunk = &icc[from..][..len];
                output.write_all(chunk).map_err(Error::WriteImage)?;
            }
            jxl_oxide::jpeg_bitstream::ReconstructionStatus::WriteExif => {
                let AuxBoxData::Data(exif) = &exif else {
                    todo!();
                };
                output
                    .write_all(exif.payload())
                    .map_err(Error::WriteImage)?;
            }
            jxl_oxide::jpeg_bitstream::ReconstructionStatus::WriteXml => {
                let AuxBoxData::Data(xmp) = &xmp else {
                    todo!();
                };
                output.write_all(xmp).map_err(Error::WriteImage)?;
            }
        }
    }

    Ok(())
}
