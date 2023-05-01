#![allow(unused_variables, unused_mut, dead_code)]

use std::num::Wrapping;

use jxl_grid::SimpleGrid;

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
pub fn rng_seed0(visible_frames: u64, invisible_frames: u64) -> u64 {
    (visible_frames << 32) + invisible_frames
}

#[inline]
pub fn rng_seed1(x0: usize, y0: usize) -> u64 {
    ((x0 as u64) << 32) + y0 as u64
}

pub fn init_noise(
    visible_frames: u64,
    invisible_frames: u64,
    header: &FrameHeader,
) -> [SimpleGrid<f32>; 3] {
    let width = header.width as usize;
    let height = header.height as usize;
    let group_dim = header.group_dim() as usize;
    let seed0 = rng_seed0(visible_frames, invisible_frames);

    let mut noise_buffer: [SimpleGrid<f32>; 3] = [
        SimpleGrid::new(width, height),
        SimpleGrid::new(width, height),
        SimpleGrid::new(width, height),
    ];

    let groups_per_row = (width + group_dim - 1) / group_dim;
    let groups_num = groups_per_row * ((height + group_dim - 1) / group_dim);

    for group_idx in 0..groups_num {
        // TODO: It may be necessary to multiply by `upsamping`
        let group_x = group_idx % groups_per_row;
        let group_y = group_idx / groups_per_row;
        // TODO: additional groups for upsampling may be needed
        let x0 = group_x * group_dim;
        let y0 = group_y * group_dim;
        init_noise_group(seed0, &mut noise_buffer, x0, y0, width, height, group_dim);
    }

    let mut laplacian = [[0f32; 5]; 5];
    (0..5).for_each(|y| {
        (0..5).for_each(|x| laplacian[y][x] = if x == 2 && y == 2 { -3.84 } else { 0.16 });
    });

    let mut convolved: [SimpleGrid<f32>; 3] = [
        SimpleGrid::new(width, height),
        SimpleGrid::new(width, height),
        SimpleGrid::new(width, height),
    ];

    for c in 0..3 {
        let cbuffer = convolved[c].buf_mut();
        let nbuffer = noise_buffer[c].buf();
        (0..height).for_each(|y| {
            (0..width).for_each(|x| {
                (0..5).for_each(|iy| {
                    (0..5).for_each(|ix| {
                        let cy = mirror((y + iy) as i32 - 2, height as i32);
                        let cx = mirror((x + ix) as i32 - 2, width as i32);
                        cbuffer[y * width + x] += nbuffer[cy * width + cx] * laplacian[iy][ix];
                    });
                });
            });
        });
    }

    convolved
}

fn mirror(val: i32, size: i32) -> usize {
    if val < 0 {
        return mirror(-val - 1, size);
    }
    if val >= size {
        return mirror(2 * size - val - 1, size);
    }
    val as usize
}

pub fn init_noise_group(
    seed0: u64,
    noise_buffer: &mut [SimpleGrid<f32>; 3],
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

    dbg!((width, height, xsize, ysize));

    (0..3).for_each(|c| {
        let channel = &mut noise_buffer[c];
        let buffer = channel.buf_mut();

        for y in 0..ysize {
            for x in (0..xsize).step_by(N * 2) {
                let bits = rng.get_u32_bits();
                for i in 0..(N * 2) {
                    if x + i >= xsize {
                        break;
                    }
                    let random = u32bits_to_f32(bits[i] >> 9 | 0x3F_80_00_00);
                    buffer[(y0 + y) * width + x0 + x + i] = random;
                }
            }
        }
    });
}

const N: usize = 8;
struct XorShift128Plus {
    s0: [Wrapping<u64>; N],
    s1: [Wrapping<u64>; N],
    pub batch: [u64; N],
    pub batch_pos: usize,
}

impl XorShift128Plus {
    fn new(seed0: u64, seed1: u64) -> Self {
        let seed0 = Wrapping(seed0);
        let seed1 = Wrapping(seed1);
        let mut s0 = [Wrapping(0u64); N];
        let mut s1 = [Wrapping(0u64); N];
        s0[0] = split_mix_64(seed0 + Wrapping(0x9E3779B97F4A7C15));
        s1[0] = split_mix_64(seed1 + Wrapping(0x9E3779B97F4A7C15));
        for i in 1..8 {
            s0[i] = split_mix_64(s0[i - 1]);
            s1[i] = split_mix_64(s1[i - 1]);
        }
        Self {
            s0,
            s1,
            batch: [0u64; N],
            batch_pos: 0,
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
fn reinterpret_cast_u64_to_u32(val: u64) -> [u32; 2] {
    let lo = (val & 0xFF_FF_FF_FF) as u32;
    let hi = (val >> 32) as u32;
    [lo, hi]
}

#[inline]
fn u32bits_to_f32(val: u32) -> f32 {
    f32::from_le_bytes(u32::to_le_bytes(val))
}

fn split_mix_64(z: Wrapping<u64>) -> Wrapping<u64> {
    let z = (z ^ (z >> 30)) * Wrapping(0xBF58476D1CE4E5B9);
    let z = (z ^ (z >> 27)) * Wrapping(0x94D049BB133111EB);
    z ^ (z >> 31)
}
