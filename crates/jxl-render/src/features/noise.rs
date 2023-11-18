use std::num::Wrapping;

use jxl_frame::{data::NoiseParameters, FrameHeader};
use jxl_grid::{PaddedGrid, SimpleGrid};

use crate::region::ImageWithRegion;

pub fn render_noise(
    header: &FrameHeader,
    visible_frames_num: usize,
    invisible_frames_num: usize,
    base_correlations_xb: Option<(f32, f32)>,
    grid: &mut ImageWithRegion,
    params: &NoiseParameters,
) -> crate::Result<()> {
    let region = grid.region();
    let [grid_r, grid_g, grid_b, ..] = grid.buffer_mut() else {
        panic!()
    };
    assert!(region.left >= 0 && region.top >= 0);

    let left = region.left as usize;
    let top = region.top as usize;
    let width = region.width as usize;
    let height = region.height as usize;
    let (corr_x, corr_b) = base_correlations_xb.unwrap_or((0.0, 1.0));

    // TODO: Initialize smaller noise grid.
    let noise_buffer = init_noise(visible_frames_num, invisible_frames_num, header);

    for fy in 0..height {
        let y = fy + top;
        for fx in 0..width {
            let x = fx + left;

            let grid_x = grid_r.get(fx, fy).unwrap();
            let grid_y = grid_g.get(fx, fy).unwrap();
            let noise_r = noise_buffer[0].get(x, y).unwrap();
            let noise_g = noise_buffer[1].get(x, y).unwrap();
            let noise_b = noise_buffer[2].get(x, y).unwrap();

            let in_g = grid_y - grid_x;
            let in_r = grid_x + grid_y;
            let in_scaled_r = f32::max(0.0, in_r * 3.0);
            let in_scaled_g = f32::max(0.0, in_g * 3.0);

            let (in_int_r, in_frac_r) = if in_scaled_r >= 7.0 {
                (6, 1.0)
            } else {
                (
                    in_scaled_r.floor() as usize,
                    in_scaled_r - in_scaled_r.floor(),
                )
            };
            let (in_int_g, in_frac_g) = if in_scaled_g >= 7.0 {
                (6, 1.0)
            } else {
                (
                    in_scaled_g.floor() as usize,
                    in_scaled_g - in_scaled_g.floor(),
                )
            };

            let lut = params.lut;
            let sr = (lut[in_int_r + 1] - lut[in_int_r]) * in_frac_r + lut[in_int_r];
            let sg = (lut[in_int_g + 1] - lut[in_int_g]) * in_frac_g + lut[in_int_g];
            let nr = 0.22 * sr * (0.0078125 * noise_r + 0.9921875 * noise_b);
            let ng = 0.22 * sg * (0.0078125 * noise_g + 0.9921875 * noise_b);
            *grid_r.get_mut(fx, fy).unwrap() += corr_x * (nr + ng) + nr - ng;
            *grid_g.get_mut(fx, fy).unwrap() += nr + ng;
            *grid_b.get_mut(fx, fy).unwrap() += corr_b * (nr + ng);
        }
    }

    Ok(())
}

fn init_noise(
    visible_frames: usize,
    invisible_frames: usize,
    header: &FrameHeader,
) -> [SimpleGrid<f32>; 3] {
    let seed0 = rng_seed0(visible_frames, invisible_frames);

    // We use header.width and header.height because
    // these are the dimensions after upsampling (the "actual" frame size),
    // and noise synthesis is done after upsampling.
    let width = header.width as usize;
    let height = header.height as usize;

    let group_dim = header.group_dim() as usize;
    let groups_per_row = (width + group_dim - 1) / group_dim;
    let groups_num = groups_per_row * ((height + group_dim - 1) / group_dim);

    // Padding for 5x5 kernel convolution step
    const PADDING: usize = 2;

    let mut noise_buffer: [PaddedGrid<f32>; 3] = [
        PaddedGrid::new(width, height, PADDING),
        PaddedGrid::new(width, height, PADDING),
        PaddedGrid::new(width, height, PADDING),
    ];

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
        let noise_width = nchannel.width() + nchannel.padding() * 2;
        let nbuffer = nchannel.buf_padded();
        (0..height).for_each(|y| {
            (0..width).for_each(|x| {
                (0..5).for_each(|iy| {
                    (0..5).for_each(|ix| {
                        let cy = y + iy;
                        let cx = x + ix;
                        cbuffer[y * width + x] +=
                            nbuffer[cy * noise_width + cx] * laplacian[iy][ix];
                    });
                });
            });
        });
    }

    convolved
}

#[inline]
/// Seed for [`XorShift128Plus`] from the number of ‘visible’ frames decoded so far
/// and the number of ‘invisible’ frames since the previous visible frame.
fn rng_seed0(visible_frames: usize, invisible_frames: usize) -> u64 {
    ((visible_frames as u64) << 32) + invisible_frames as u64
}

#[inline]
/// Seed for [`XorShift128Plus`] from the coordinates of the top-left pixel of the
/// group within the frame.
fn rng_seed1(x0: usize, y0: usize) -> u64 {
    ((x0 as u64) << 32) + y0 as u64
}

/// Initializes `noise_buffer` for group
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
    let seed1 = rng_seed1(x0, y0);
    let mut rng = XorShift128Plus::new(seed0, seed1);

    let xsize = usize::min(group_dim, width.wrapping_sub(x0));
    let ysize = usize::min(group_dim, height.wrapping_sub(y0));

    for channel in noise_buffer {
        let buf_padding = channel.padding();
        let buf_width = channel.width() + channel.padding() * 2;
        let buf = channel.buf_padded_mut();
        for y in 0..ysize {
            let y = y0 + y + buf_padding;
            for x in (0..xsize).step_by(N * 2) {
                let bits = rng.get_u32_bits();
                let iters = (xsize - x).min(N * 2);

                #[allow(clippy::needless_range_loop)]
                for i in 0..iters {
                    let x = x0 + x + i + buf_padding;
                    let random = f32::from_bits(bits[i] >> 9 | 0x3F_80_00_00);
                    buf[y * buf_width + x] = random;
                }
            }
        }
    }
}

const N: usize = 8;

/// Shift-register pseudo-random number generator
struct XorShift128Plus {
    s0: [Wrapping<u64>; N],
    s1: [Wrapping<u64>; N],
    pub batch: [u64; N],
}

impl XorShift128Plus {
    /// Initialize a new XorShift128+ PRNG.
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

    /// Returns N * 2 [`u32`] pseudorandom numbers
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
/// Pseudo-random number generator used to calculate initial state of [`XorShift128Plus`]
fn split_mix_64(z: Wrapping<u64>) -> Wrapping<u64> {
    let z = (z ^ (z >> 30)) * Wrapping(0xBF58476D1CE4E5B9);
    let z = (z ^ (z >> 27)) * Wrapping(0x94D049BB133111EB);
    z ^ (z >> 31)
}
