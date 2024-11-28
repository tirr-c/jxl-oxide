use jxl_oxide::JxlImage;

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

    let aux_boxes = image.aux_boxes();
    let jbrd_header = jbrd.header();
    let expected_icc_len = jbrd_header.expected_icc_len();
    let expected_exif_len = jbrd_header.expected_exif_len();
    let expected_xmp_len = jbrd_header.expected_xmp_len();

    let icc = if expected_icc_len > 0 {
        image.original_icc().unwrap_or(&[])
    } else {
        &[]
    };

    let exif = if expected_exif_len > 0 {
        let b = aux_boxes.first_exif().map_err(Error::ReadJxl)?;
        b.map(|x| x.payload()).unwrap_or(&[])
    } else {
        &[]
    };

    let xmp = if expected_xmp_len > 0 {
        aux_boxes.first_xml().unwrap_or(&[])
    } else {
        &[]
    };

    let frame = image.frame_by_keyframe(0).unwrap();
    let reconstructor = jbrd
        .reconstruct(frame, icc, exif, xmp)
        .map_err(|e| Error::Reconstruct(e.into()))?;

    let output = std::fs::File::create(output_path).map_err(Error::WriteImage)?;
    let mut output = std::io::BufWriter::new(output);
    reconstructor
        .write(&mut output)
        .map_err(|e| Error::Reconstruct(e.into()))?;

    Ok(())
}
