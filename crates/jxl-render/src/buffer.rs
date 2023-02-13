use jxl_bitstream::header::ImageMetadata;

#[derive(Debug, Clone)]
pub struct FrameBuffer {
    width: u32,
    height: u32,
    stride: u32,
    buf: Vec<Vec<f32>>,
}

impl FrameBuffer {
    pub fn new(width: u32, height: u32, stride: u32, channels: u32) -> Self {
        Self {
            width,
            height,
            stride,
            buf: vec![vec![0.0f32; stride as usize * height as usize]; channels as usize],
        }
    }

    pub fn collected(buffers: Vec<FrameBuffer>) -> Self {
        if buffers.is_empty() {
            panic!();
        }

        let width = buffers[0].width;
        let height = buffers[0].height;
        let stride = buffers[0].stride;
        if !buffers.iter().all(|fb| fb.width == width && fb.height == height && fb.stride == stride) {
            panic!();
        }

        let buf = buffers.into_iter().flat_map(|fb| fb.buf).collect();
        Self {
            width,
            height,
            stride,
            buf,
        }
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn stride(&self) -> u32 {
        self.stride
    }

    pub fn channels(&self) -> u32 {
        self.buf.len() as u32
    }

    pub fn channel_buf(&self, channel: u32) -> &[f32] {
        &self.buf[channel as usize]
    }

    pub fn channel_buf_mut(&mut self, channel: u32) -> &mut [f32] {
        &mut self.buf[channel as usize]
    }

    pub fn channel_buffers(&self) -> Vec<&[f32]> {
        self.buf.iter().map(|b| &**b).collect()
    }

    pub fn channel_buffers_mut(&mut self) -> Vec<&mut [f32]> {
        self.buf.iter_mut().map(|b| &mut **b).collect()
    }
}

impl FrameBuffer {
    pub fn yxb_to_rgb(&mut self, metadata: &ImageMetadata) {
        crate::color::perform_inverse_xyb(self, metadata)
    }

    pub fn rgba_be_interleaved<F, E>(&self, mut f: F) -> std::result::Result<(), E>
    where
        F: FnMut(&[u8]) -> std::result::Result<(), E>,
    {
        let mut buf = vec![0u8; self.width as usize * self.buf.len()];
        let channels = self.buf.len();

        for y in 0..self.height as usize {
            for x in 0..self.width as usize {
                for c in 0..channels {
                    let s = self.buf[c][y * self.stride as usize + x].clamp(0.0, 1.0);
                    buf[x * channels + c] = (s * 255.0) as u8;
                }
            }
            f(&buf)?;
        }

        Ok(())
    }
}
