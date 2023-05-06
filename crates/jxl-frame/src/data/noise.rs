use std::num::Wrapping;

use jxl_grid::{PaddedGrid, SimpleGrid};

use crate::FrameHeader;

#[derive(Debug)]
pub struct NoiseParameters {
    pub lut: [f32; 8],
}

impl<Ctx> jxl_bitstream::Bundle<Ctx> for NoiseParameters {
    type Error = crate::Error;

    fn parse<R: std::io::Read>(
        bitstream: &mut jxl_bitstream::Bitstream<R>,
        _: Ctx,
    ) -> crate::Result<Self> {
        let mut lut = [0.0f32; 8];
        for slot in &mut lut {
            *slot = bitstream.read_bits(10)? as f32 / (1 << 10) as f32;
        }

        Ok(Self { lut })
    }
}

#[inline]
fn rng_seed0(visible_frames: usize, invisible_frames: usize) -> u64 {
    ((visible_frames as u64) << 32) + invisible_frames as u64
}

#[inline]
fn rng_seed1(x0: usize, y0: usize) -> u64 {
    ((x0 as u64) << 32) + y0 as u64
}

pub fn init_noise(
    visible_frames: usize,
    invisible_frames: usize,
    header: &FrameHeader,
) -> [SimpleGrid<f32>; 3] {
    let width = header.width as usize;
    let height = header.height as usize;
    let seed0 = rng_seed0(visible_frames, invisible_frames);

    // NOTE: It may be necessary to multiply group_dim by `upsampling`
    let group_dim = header.group_dim() as usize;

    // Padding for 5x5 kernel convolution step
    const PADDING: usize = 2;

    let mut noise_buffer: [PaddedGrid<f32>; 3] = [
        PaddedGrid::new(width, height, PADDING),
        PaddedGrid::new(width, height, PADDING),
        PaddedGrid::new(width, height, PADDING),
    ];

    let groups_per_row = (width + group_dim - 1) / group_dim;
    let groups_num = groups_per_row * ((height + group_dim - 1) / group_dim);

    for group_idx in 0..groups_num {
        let group_x = group_idx % groups_per_row;
        let group_y = group_idx / groups_per_row;
        let x0 = group_x * group_dim;
        let y0 = group_y * group_dim;
        init_noise_group(seed0, &mut noise_buffer, x0, y0, width, height, group_dim);
    }

    for channel in &mut noise_buffer {
        channel.mirror_edges_padding();
    }

    let mut convolved: [SimpleGrid<f32>; 3] = [
        SimpleGrid::new(width, height),
        SimpleGrid::new(width, height),
        SimpleGrid::new(width, height),
    ];

    let mut laplacian = [[0.16f32; 5]; 5];
    laplacian[2][2] = -3.84;

    // Each channel is convolved by the 5×5 kernel
    for (cchannel, nchannel) in convolved.iter_mut().zip(noise_buffer) {
        let cbuffer = cchannel.buf_mut();
        (0..height).for_each(|y| {
            (0..width).for_each(|x| {
                (0..5).for_each(|iy| {
                    (0..5).for_each(|ix| {
                        let cy = (y + iy) as i32 - 2;
                        let cx = (x + ix) as i32 - 2;
                        cbuffer[y * width + x] +=
                            nchannel.get_unchecked(cx, cy) * laplacian[iy][ix];
                    });
                });
            });
        });
    }

    convolved
}

fn init_noise_group(
    seed0: u64,
    noise_buffer: &mut [PaddedGrid<f32>; 3],
    // Group start coordinates
    x0: usize,
    y0: usize,
    // Frame size
    width: usize,
    height: usize,
    group_dim: usize,
) {
    let xsize = usize::min(group_dim, width.wrapping_sub(x0));
    let ysize = usize::min(group_dim, height.wrapping_sub(y0));

    let seed1 = rng_seed1(x0, y0);
    let mut rng = XorShift128Plus::new(seed0, seed1);

    for channel in noise_buffer {
        for y in 0..ysize {
            for x in (0..xsize).step_by(N * 2) {
                let bits = rng.get_u32_bits();

                #[allow(clippy::needless_range_loop)]
                for i in 0..(N * 2) {
                    if x + i >= xsize {
                        break;
                    }
                    let random = f32::from_bits(bits[i] >> 9 | 0x3F_80_00_00);
                    *channel.get_unchecked_mut_usize(x0 + x + i, y0 + y) = random;
                }
            }
        }
    }
}

const N: usize = 8;
struct XorShift128Plus {
    s0: [Wrapping<u64>; N],
    s1: [Wrapping<u64>; N],
    pub batch: [u64; N],
}

impl XorShift128Plus {
    fn new(seed0: u64, seed1: u64) -> Self {
        let seed0 = Wrapping(seed0);
        let seed1 = Wrapping(seed1);
        let mut s0 = [Wrapping(0u64); N];
        let mut s1 = [Wrapping(0u64); N];
        s0[0] = split_mix_64(seed0 + Wrapping(0x9E3779B97F4A7C15));
        s1[0] = split_mix_64(seed1 + Wrapping(0x9E3779B97F4A7C15));
        for i in 1..N {
            s0[i] = split_mix_64(s0[i - 1]);
            s1[i] = split_mix_64(s1[i - 1]);
        }
        Self {
            s0,
            s1,
            batch: [0u64; N],
        }
    }

    pub fn get_u32_bits(&mut self) -> [u32; N * 2] {
        let mut bits = [0; N * 2];
        self.fill_batch();
        (0..N).for_each(|i| {
            let l = self.batch[i];
            bits[i * 2] = (l & 0xFF_FF_FF_FF) as u32;
            bits[i * 2 + 1] = (l >> 32) as u32;
        });
        bits
    }

    fn fill_batch(&mut self) {
        (0..N).for_each(|i| {
            let mut s1 = self.s0[i];
            let s0 = self.s1[i];
            self.batch[i] = (s1 + s0).0;
            self.s0[i] = s0;
            s1 ^= s1 << 23;
            self.s1[i] = s1 ^ (s0 ^ (s1 >> 18) ^ (s0 >> 5));
        });
    }
}

#[inline]
fn split_mix_64(z: Wrapping<u64>) -> Wrapping<u64> {
    let z = (z ^ (z >> 30)) * Wrapping(0xBF58476D1CE4E5B9);
    let z = (z ^ (z >> 27)) * Wrapping(0x94D049BB133111EB);
    z ^ (z >> 31)
}
