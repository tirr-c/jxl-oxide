use jxl_oxide::{JpegReconstructionStatus, JxlImage};

use crate::commands::dump_jbrd::*;
use crate::{Error, Result};

pub fn handle_dump_jbrd(args: DumpJbrd) -> Result<()> {
    let _guard = tracing::trace_span!("Handle dump-jbrd subcommand").entered();

    let image = JxlImage::builder()
        .open(&args.input)
        .map_err(Error::ReadJxl)?;

    let status = image.jpeg_reconstruction_status();
    if status != JpegReconstructionStatus::Available {
        println!("No reconstruction data available: {status:?}");
        return Ok(());
    }

    let Some(output_path) = args.output_jpeg else {
        return Ok(());
    };

    let output = std::fs::File::create(output_path).map_err(Error::WriteImage)?;
    let output = std::io::BufWriter::new(output);
    image.reconstruct_jpeg(output).map_err(Error::Reconstruct)?;
    Ok(())
}
