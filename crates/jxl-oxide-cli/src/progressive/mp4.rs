use std::{fs::File, path::Path};

use jxl_oxide::{JxlImage, Render};

use crate::{output::VideoContext, Error, Result};

pub struct Context {
    inner: VideoContext<File>,
    font: std::ffi::CString,
    inited: bool,
    idx: usize,
}

impl Context {
    pub fn new(output_path: impl AsRef<Path>, font: &str) -> Result<Self> {
        let file = File::create(output_path).map_err(Error::WriteImage)?;
        let inner = VideoContext::new(file)?;
        let c_font = format!("{font}\0");
        Ok(Self {
            inner,
            font: std::ffi::CString::from_vec_with_nul(c_font.into_bytes()).unwrap(),
            inited: false,
            idx: 0,
        })
    }

    fn ensure_init(&mut self, image: &JxlImage) -> Result<()> {
        if self.inited {
            return Ok(());
        }

        let is_bt2100 = image.is_hdr();
        let width = image.width();
        let height = image.height();
        let pixel_format = image.pixel_format();
        let hdr_luminance = is_bt2100.then(|| {
            let tone_mapping = &image.image_header().metadata.tone_mapping;
            (tone_mapping.min_nits, tone_mapping.intensity_target)
        });
        self.inner
            .init_video(width, height, pixel_format, hdr_luminance, &self.font)?;

        self.inited = true;
        Ok(())
    }

    pub fn add_empty_frame(
        &mut self,
        image: &JxlImage,
        description: impl std::fmt::Display,
    ) -> Result<()> {
        self.ensure_init(image)?;
        self.inner.write_empty_frame(description)?;
        self.idx += 1;
        Ok(())
    }

    pub fn add_frame(
        &mut self,
        image: &JxlImage,
        render: &Render,
        description: impl std::fmt::Display,
    ) -> Result<()> {
        self.ensure_init(image)?;
        self.inner.write_frame(render, description)?;
        self.idx += 1;
        Ok(())
    }

    pub fn finalize(&mut self) -> Result<()> {
        self.inner.finalize()
    }
}
