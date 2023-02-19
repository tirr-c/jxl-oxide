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
    pub fn yxb_to_srgb_linear(&mut self, metadata: &ImageMetadata) {
        crate::color::perform_inverse_xyb(self, metadata)
    }

    pub fn ycbcr_to_rgb(&mut self) {
        crate::color::perform_inverse_ycbcr(self)
    }

    pub fn ycbcr_upsample(&mut self, jpeg_upsampling: [u32; 3]) {
        fn interpolate(left: f32, center: f32, right: f32) -> (f32, f32) {
            (0.25 * left + 0.75 * center, 0.75 * center + 0.25 * right)
        }

        let shifts_ycbcr = [1, 0, 2].map(|idx| {
            jxl_modular::ChannelShift::from_jpeg_upsampling(jpeg_upsampling, idx)
        });

        let width = self.width;
        let height = self.height;
        let stride = self.stride;
        for (buf, shift) in self.buf[..3].iter_mut().zip(shifts_ycbcr) {
            let h_upsampled = shift.hshift() == 0;
            let v_upsampled = shift.vshift() == 0;

            if !h_upsampled {
                let orig_width = width;
                let width = (width + 1) / 2;
                let height = if v_upsampled { height } else { (height + 1) / 2 };

                for y in 0..height {
                    let y = if v_upsampled { y } else { y * 2 };
                    let idx_base = (y * stride) as usize;
                    let mut prev_sample = buf[idx_base];
                    for x in 0..width {
                        let x = x as usize;
                        let curr_sample = buf[idx_base + x * 2];
                        let right_x = if x == width as usize - 1 { x } else { x + 1 };

                        let (me, next) = interpolate(
                            prev_sample,
                            curr_sample,
                            buf[idx_base + right_x * 2],
                        );
                        buf[idx_base + x * 2] = me;
                        if x * 2 + 1 < orig_width as usize {
                            buf[idx_base + x * 2 + 1] = next;
                        }

                        prev_sample = curr_sample;
                    }
                }
            }

            // image is horizontally upsampled here
            if !v_upsampled {
                let orig_height = height;
                let height = (height + 1) / 2;

                let mut prev_row = buf[..width as usize].to_vec();
                for y in 0..height {
                    let idx_base = (y * 2 * stride) as usize;
                    let bottom_base = if y == height - 1 { idx_base } else { idx_base + stride as usize * 2 };
                    for x in 0..width {
                        let x = x as usize;
                        let curr_sample = buf[idx_base + x];

                        let (me, next) = interpolate(
                            prev_row[x],
                            curr_sample,
                            buf[bottom_base + x],
                        );
                        buf[idx_base + x] = me;
                        if y * 2 + 1 < orig_height {
                            buf[idx_base + stride as usize + x] = next;
                        }

                        prev_row[x] = curr_sample;
                    }
                }
            }
        }
    }

    pub fn srgb_linear_to_standard(&mut self) {
        for buf in &mut self.buf {
            for s in buf {
                *s = if *s <= 0.0031308f32 {
                    12.92 * *s
                } else {
                    1.055 * s.powf(1.0 / 2.4) - 0.055
                };
            }
        }
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
