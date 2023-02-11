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

    pub fn channel_buf(&self, channel: u32) -> &[f32] {
        &self.buf[channel as usize]
    }

    pub fn channel_buf_mut(&mut self, channel: u32) -> &mut [f32] {
        &mut self.buf[channel as usize]
    }
}
