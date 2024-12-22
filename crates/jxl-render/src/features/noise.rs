use std::num::Wrapping;

use jxl_frame::{data::NoiseParameters, FrameHeader};
use jxl_grid::{AlignedGrid, AllocTracker};
use jxl_threadpool::JxlThreadPool;

use crate::{ImageWithRegion, Region, Result};

// Padding for 5x5 kernel convolution step
const PADDING: usize = 2;

pub fn render_noise(
    header: &FrameHeader,
    visible_frames_num: usize,
    invisible_frames_num: usize,
    base_correlations_xb: Option<(f32, f32)>,
    grid: &mut ImageWithRegion,
    params: &NoiseParameters,
    pool: &JxlThreadPool,
) -> Result<()> {
    let (region, shift) = grid.regions_and_shifts()[0];
    let tracker = grid.alloc_tracker().cloned();
    let [grid_x, grid_y, grid_b] = grid.as_color_floats_mut();

    let full_frame_region = Region::with_size(header.width, header.height);
    let actual_region = region
        .intersection(full_frame_region)
        .downsample_with_shift(shift);

    let left = actual_region.left as usize;
    let top = actual_region.top as usize;
    let width = actual_region.width as usize;
    let height = actual_region.height as usize;
    let (corr_x, corr_b) = base_correlations_xb.unwrap_or((0.0, 1.0));

    let noise_buffer = init_noise(
        visible_frames_num,
        invisible_frames_num,
        header,
        tracker.as_ref(),
        pool,
    )?;

    let mut lut = [0f32; 9];
    lut[..8].copy_from_slice(&params.lut);
    lut[8] = params.lut[7];
    for fy in 0..height {
        let y = fy + top;
        let row_x = grid_x.get_row_mut(fy).unwrap();
        let row_y = grid_y.get_row_mut(fy).unwrap();
        let row_b = grid_b.get_row_mut(fy).unwrap();
        let row_noise_x = noise_buffer[0].get_row(y).unwrap();
        let row_noise_y = noise_buffer[1].get_row(y).unwrap();
        let row_noise_b = noise_buffer[2].get_row(y).unwrap();

        for fx in 0..width {
            let x = fx + left;

            let grid_x = row_x[fx];
            let grid_y = row_y[fx];
            let noise_x = row_noise_x[x];
            let noise_y = row_noise_y[x];
            let noise_b = row_noise_b[x];

            let in_x = grid_x + grid_y;
            let in_y = grid_y - grid_x;
            let in_scaled_x = f32::max(0.0, in_x * 3.0);
            let in_scaled_y = f32::max(0.0, in_y * 3.0);

            let in_x_int = (in_scaled_x as usize).min(7);
            let in_x_frac = in_scaled_x - in_x_int as f32;
            let in_y_int = (in_scaled_y as usize).min(7);
            let in_y_frac = in_scaled_y - in_y_int as f32;

            let sx = (lut[in_x_int + 1] - lut[in_x_int]) * in_x_frac + lut[in_x_int];
            let sy = (lut[in_y_int + 1] - lut[in_y_int]) * in_y_frac + lut[in_y_int];
            let nx = 0.22 * sx * (0.0078125 * noise_x + 0.9921875 * noise_b);
            let ny = 0.22 * sy * (0.0078125 * noise_y + 0.9921875 * noise_b);
            row_x[fx] += corr_x * (nx + ny) + nx - ny;
            row_y[fx] += nx + ny;
            row_b[fx] += corr_b * (nx + ny);
        }
    }

    Ok(())
}

fn init_noise(
    visible_frames: usize,
    invisible_frames: usize,
    header: &FrameHeader,
    tracker: Option<&AllocTracker>,
    pool: &JxlThreadPool,
) -> Result<[AlignedGrid<f32>; 3]> {
    let seed0 = rng_seed0(visible_frames, invisible_frames);

    // We use header.width and header.height because
    // these are the dimensions after upsampling (the "actual" frame size),
    // and noise synthesis is done after upsampling.
    let width = header.width as usize;
    let height = header.height as usize;

    let group_dim = header.group_dim() as usize;
    let groups_per_row = width.div_ceil(group_dim);
    let num_groups = groups_per_row * height.div_ceil(group_dim);

    let mut noise_groups = Vec::with_capacity(num_groups);
    for group_idx in 0..num_groups {
        let group_x = group_idx % groups_per_row;
        let group_y = group_idx / groups_per_row;
        let x0 = group_x * group_dim;
        let y0 = group_y * group_dim;
        let seed1 = rng_seed1(x0, y0);

        let group_width = group_dim.min(width - x0);
        let group_height = group_dim.min(height - y0);
        let mut noise_buffer = [
            AlignedGrid::with_alloc_tracker(group_width, group_height, tracker)?,
            AlignedGrid::with_alloc_tracker(group_width, group_height, tracker)?,
            AlignedGrid::with_alloc_tracker(group_width, group_height, tracker)?,
        ];

        init_noise_group(seed0, seed1, &mut noise_buffer);
        noise_groups.push(noise_buffer);
    }

    let mut convolved: [AlignedGrid<f32>; 3] = [
        AlignedGrid::with_alloc_tracker(width, height, tracker)?,
        AlignedGrid::with_alloc_tracker(width, height, tracker)?,
        AlignedGrid::with_alloc_tracker(width, height, tracker)?,
    ];

    // Each channel is convolved by the 5×5 kernel
    let mut jobs = Vec::with_capacity(num_groups * 3);
    for (channel_idx, out) in convolved.iter_mut().enumerate() {
        for (group_idx, out_subgrid) in out
            .as_subgrid_mut()
            .into_groups(group_dim, group_dim)
            .into_iter()
            .enumerate()
        {
            let group_x = group_idx % groups_per_row;
            let group_y = group_idx / groups_per_row;

            // `adjacent_groups[4] == this`
            let adjacent_groups: [_; 9] = std::array::from_fn(|idx| {
                let offset_x = (idx % 3) as isize - 1;
                let offset_y = (idx / 3) as isize - 1;
                if let (Some(x), Some(y)) = (
                    group_x.checked_add_signed(offset_x),
                    group_y.checked_add_signed(offset_y),
                ) {
                    let group_idx = y * groups_per_row + x;
                    if x < groups_per_row {
                        noise_groups.get(group_idx).map(|group| &group[channel_idx])
                    } else {
                        None
                    }
                } else {
                    None
                }
            });

            jobs.push((out_subgrid, adjacent_groups));
        }
    }

    let result = std::sync::Mutex::new(Ok(()));
    pool.for_each_vec(jobs, |job| {
        let (out_subgrid, adjacent_groups) = job;
        let r = convolve_fill(out_subgrid, adjacent_groups, tracker);
        if r.is_err() {
            *result.lock().unwrap() = r;
        }
    });
    result.into_inner().unwrap()?;

    Ok(convolved)
}

/// Seed for [`XorShift128Plus`] from the number of ‘visible’ frames decoded so far
/// and the number of ‘invisible’ frames since the previous visible frame.
#[inline]
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
fn init_noise_group(seed0: u64, seed1: u64, noise_buffer: &mut [AlignedGrid<f32>; 3]) {
    let mut rng = XorShift128Plus::new(seed0, seed1);

    for channel in noise_buffer {
        let width = channel.width();
        let height = channel.height();
        let buf = channel.buf_mut();
        for y in 0..height {
            for x in (0..width).step_by(N * 2) {
                let idx_base = y * width + x;
                let bits = rng.get_u32_bits();
                let iters = (width - x).min(N * 2);
                let buf_section = &mut buf[idx_base..][..iters];
                for (out, bits) in std::iter::zip(buf_section, bits) {
                    *out = f32::from_bits(bits >> 9 | 0x3f800000);
                }
            }
        }
    }
}

#[inline(never)]
fn convolve_fill(
    mut out: jxl_grid::MutableSubgrid<'_, f32>,
    adjacent_groups: [Option<&AlignedGrid<f32>>; 9],
    tracker: Option<&AllocTracker>,
) -> Result<()> {
    let this = adjacent_groups[4].unwrap();
    let width = out.width();
    let height = out.height();
    assert_eq!(this.width(), width);
    assert_eq!(this.height(), height);

    let mut rows = AlignedGrid::with_alloc_tracker(width + PADDING * 2, 1 + PADDING * 2, tracker)?;
    if let Some(c) = adjacent_groups[1] {
        let l = adjacent_groups[0];
        let r = adjacent_groups[2];
        for offset_y in -2..0 {
            let out = rows
                .get_row_mut(2usize.wrapping_add_signed(offset_y))
                .unwrap();
            let c = c.get_row(c.height().wrapping_add_signed(offset_y)).unwrap();
            let l = l.map(|l| l.get_row(l.height().wrapping_add_signed(offset_y)).unwrap());
            let r = r.map(|r| r.get_row(r.height().wrapping_add_signed(offset_y)).unwrap());
            fill_padded_row(out, c, l, r);
        }
    } else if height >= 2 {
        let c = this;
        let l = adjacent_groups[3];
        let r = adjacent_groups[5];
        for offset_y in -2..0 {
            let y = (-(offset_y + 1)) as usize;
            let out = rows
                .get_row_mut(2usize.wrapping_add_signed(offset_y))
                .unwrap();
            let c = c.get_row(y).unwrap();
            let l = l.map(|l| l.get_row(y).unwrap());
            let r = r.map(|r| r.get_row(y).unwrap());
            fill_padded_row(out, c, l, r);
        }
    } else {
        let c = this;
        let l = adjacent_groups[3];
        let r = adjacent_groups[5];

        let c = c.get_row(0).unwrap();
        let l = l.map(|l| l.get_row(0).unwrap());
        let r = r.map(|r| r.get_row(0).unwrap());
        for y in 0..2 {
            let out = rows.get_row_mut(y).unwrap();
            fill_padded_row(out, c, l, r);
        }
    }

    for y in 0..2 {
        let out = rows.get_row_mut(2 + y).unwrap();
        fill_once(out, y, adjacent_groups);
    }

    let input_width = rows.width();
    for y in 0..height {
        let fill_y = y + 2;
        fill_once(rows.get_row_mut(4).unwrap(), fill_y, adjacent_groups);

        let input_buf = rows.buf();
        let out_buf = out.get_row_mut(y);
        for (x, out) in out_buf.iter_mut().enumerate() {
            let mut sum = 0f32;
            for dy in 0..5 {
                let input_row = &input_buf[dy * input_width..][..input_width];
                for dx in 0..5 {
                    sum += input_row[x + dx] * 0.16;
                }
            }
            *out = sum - input_buf[2 * input_width + x + 2] * 4.0;
        }

        rows.buf_mut().copy_within(input_width.., 0);
    }

    Ok(())
}

#[inline(never)]
fn fill_once(out: &mut [f32], fill_y: usize, adjacent_groups: [Option<&AlignedGrid<f32>>; 9]) {
    let this = adjacent_groups[4].unwrap();
    let height = this.height();

    let (source_y, c, l, r) = if let Some(fill_y) = fill_y.checked_sub(height) {
        (
            fill_y,
            adjacent_groups[7],
            adjacent_groups[6],
            adjacent_groups[8],
        )
    } else {
        (
            fill_y,
            adjacent_groups[4],
            adjacent_groups[3],
            adjacent_groups[5],
        )
    };

    let (source_y, c, l, r) = if let Some(c) = c {
        (source_y, c, l, r)
    } else if let Some(y) = (height - 1).checked_sub(source_y) {
        let c = this;
        let l = adjacent_groups[3];
        let r = adjacent_groups[5];
        (y, c, l, r)
    } else {
        let dy = source_y - height + 1;
        if let Some(c) = adjacent_groups[1] {
            let l = adjacent_groups[0];
            let r = adjacent_groups[2];
            (c.height() - dy, c, l, r)
        } else {
            let c = this;
            let l = adjacent_groups[3];
            let r = adjacent_groups[5];
            (0, c, l, r)
        }
    };
    let c = c.get_row(source_y).unwrap();
    let l = l.map(|l| l.get_row(source_y).unwrap());
    let r = r.map(|r| r.get_row(source_y).unwrap());

    fill_padded_row(out, c, l, r);
}

fn fill_padded_row(out: &mut [f32], this: &[f32], left: Option<&[f32]>, right: Option<&[f32]>) {
    assert_eq!(out.len(), this.len() + PADDING * 2);

    if let Some(left) = left {
        out[0] = left[left.len() - 2];
        out[1] = left[left.len() - 1];
    } else if this.len() >= PADDING {
        out[0] = this[1];
        out[1] = this[0];
    } else {
        out[0] = this[0];
        out[1] = this[0];
    }

    out[2..][..this.len()].copy_from_slice(this);

    if let Some(right) = right {
        if right.len() >= PADDING {
            out[out.len() - 2] = right[0];
            out[out.len() - 1] = right[1];
        } else {
            out[out.len() - 2] = right[0];
            out[out.len() - 1] = right[0];
        }
    } else {
        out[out.len() - 2] = out[out.len() - 3];
        out[out.len() - 1] = out[out.len() - 4];
    }
}

const N: usize = 8;

/// Shift-register pseudo-random number generator
struct XorShift128Plus {
    s0: [Wrapping<u64>; N],
    s1: [Wrapping<u64>; N],
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
        Self { s0, s1 }
    }

    /// Returns N * 2 [`u32`] pseudorandom numbers
    pub fn get_u32_bits(&mut self) -> [u32; N * 2] {
        let batch = self.fill_batch();
        if 1u64.to_le() == 1u64 {
            bytemuck::cast(batch)
        } else {
            bytemuck::cast(batch.map(|x| x.rotate_left(32)))
        }
    }

    fn fill_batch(&mut self) -> [u64; N] {
        std::array::from_fn(|i| {
            let mut s1 = self.s0[i];
            let s0 = self.s1[i];
            let ret = (s1 + s0).0;
            self.s0[i] = s0;
            s1 ^= s1 << 23;
            self.s1[i] = s1 ^ (s0 ^ (s1 >> 18) ^ (s0 >> 5));
            ret
        })
    }
}

/// Pseudo-random number generator used to calculate initial state of [`XorShift128Plus`]
#[inline]
fn split_mix_64(z: Wrapping<u64>) -> Wrapping<u64> {
    let z = (z ^ (z >> 30)) * Wrapping(0xBF58476D1CE4E5B9);
    let z = (z ^ (z >> 27)) * Wrapping(0x94D049BB133111EB);
    z ^ (z >> 31)
}
