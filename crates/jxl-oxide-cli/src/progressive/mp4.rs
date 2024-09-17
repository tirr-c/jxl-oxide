use std::{fs::File, path::Path};

use jxl_oxide::{JxlImage, Render};

use crate::{output::VideoContext, Error, Result};

pub struct Context {
    inner: VideoContext<File>,
    mastering_luminances: Option<(f32, f32)>,
    cll: Option<(f32, f32)>,
    font: std::ffi::CString,
    inited: bool,
    idx: usize,
}

impl Context {
    pub fn new(
        output_path: impl AsRef<Path>,
        mastering_luminances: Option<(f32, f32)>,
        cll: Option<(f32, f32)>,
        font: &str,
    ) -> Result<Self> {
        let file = File::create(output_path).map_err(Error::WriteImage)?;
        let inner = VideoContext::new(file)?;
        let c_font = format!("{font}\0");
        Ok(Self {
            inner,
            mastering_luminances,
            cll,
            font: std::ffi::CString::from_vec_with_nul(c_font.into_bytes()).unwrap(),
            inited: false,
            idx: 0,
        })
    }

    fn ensure_init(&mut self, image: &JxlImage) -> Result<()> {
        if self.inited {
            return Ok(());
        }

        let width = image.width();
        let height = image.height();
        let pixel_format = image.pixel_format();
        self.inner.init_video(
            width,
            height,
            pixel_format,
            self.mastering_luminances,
            self.cll,
            image.hdr_type(),
            &self.font,
        )?;

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
