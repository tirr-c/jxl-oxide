use std::path::{Path, PathBuf};

use jxl_oxide::{JxlImage, Render};

use crate::{Error, Result};

pub struct Context {
    output_dir: PathBuf,
    idx: usize,
}

impl Context {
    pub fn new(output_dir: impl AsRef<Path>) -> Result<Self> {
        let output_dir = output_dir.as_ref().to_owned();
        std::fs::create_dir_all(&output_dir).map_err(Error::WriteImage)?;
        Ok(Self { output_dir, idx: 0 })
    }

    pub fn add_empty_frame(
        &mut self,
        _image: &JxlImage,
        _description: impl std::fmt::Display,
    ) -> Result<()> {
        self.idx += 1;
        Ok(())
    }

    pub fn add_frame(
        &mut self,
        image: &JxlImage,
        render: &Render,
        _description: impl std::fmt::Display,
    ) -> Result<()> {
        let mut output_path = self.output_dir.clone();
        output_path.push(format!("frame{}.png", self.idx));
        let output = std::fs::File::create(output_path).map_err(Error::WriteImage)?;

        let width = image.width();
        let height = image.height();
        crate::output::write_png(
            output,
            image,
            std::slice::from_ref(render),
            image.pixel_format(),
            None,
            width,
            height,
        )
        .map_err(Error::WriteImage)?;

        self.idx += 1;
        Ok(())
    }
}
