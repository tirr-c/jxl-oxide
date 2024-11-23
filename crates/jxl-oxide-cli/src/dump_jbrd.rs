use std::io::Write;

use jxl_oxide::JxlImage;

use crate::{Error, Result};
use crate::commands::dump_jbrd::*;

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
    let frame = image.frame_by_keyframe(0).unwrap();
    let mut reconstructor = jbrd.reconstruct(frame).map_err(|e| Error::Reconstruct(e.into()))?;

    let mut output = std::fs::File::create(output_path).map_err(Error::WriteImage)?;
    loop {
        let status = reconstructor.write(&mut output).map_err(|e| Error::Reconstruct(e.into()))?;
        match status {
            jxl_oxide::jpeg_bitstream::ReconstructionStatus::Done => break,
            jxl_oxide::jpeg_bitstream::ReconstructionStatus::WriteIcc { from, len } => {
                let icc = icc.unwrap();
                let chunk = &icc[from..][..len];
                output.write_all(chunk).map_err(Error::WriteImage)?;
            },
            jxl_oxide::jpeg_bitstream::ReconstructionStatus::WriteExif => todo!(),
            jxl_oxide::jpeg_bitstream::ReconstructionStatus::WriteXml => todo!(),
        }
    }

    Ok(())
}
