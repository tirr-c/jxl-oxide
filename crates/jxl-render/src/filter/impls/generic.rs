#![allow(dead_code)]

use jxl_frame::filter::EpfParams;
use jxl_grid::SimpleGrid;
use jxl_threadpool::JxlThreadPool;

use crate::{Region, Result};

pub(crate) mod epf;
use epf::weight;

pub(crate) struct EpfRow<'buf, 'epf> {
    pub(crate) input_rows: [[&'buf [f32]; 7]; 3],
    pub(crate) merged_input_rows: Option<[&'buf [f32]; 3]>,
    pub(crate) output_rows: [&'buf mut [f32]; 3],
    pub(crate) width: usize,
    pub(crate) x: usize,
    pub(crate) y: usize,
    pub(crate) sigma_row: &'buf [f32],
    pub(crate) epf_params: &'epf EpfParams,
    pub(crate) skip_inner: bool,
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn epf_common<'buf>(
    input: &'buf [SimpleGrid<f32>; 3],
    output: &'buf mut [SimpleGrid<f32>; 3],
    sigma_grid: &'buf SimpleGrid<f32>,
    region: Region,
    epf_params: &EpfParams,
    pool: &JxlThreadPool,
    handle_row_simd: Option<for<'a, 'b> unsafe fn(EpfRow<'a, 'b>)>,
    handle_row_generic: for<'a, 'b> fn(EpfRow<'a, 'b>),
) {
    struct EpfJob<'buf> {
        base_y: usize,
        output0: &'buf mut [f32],
        output1: &'buf mut [f32],
        output2: &'buf mut [f32],
    }

    let width = region.width as usize;
    let height = region.height as usize;
    assert!(region.left >= 0);
    assert!(region.top >= 0);
    let left = region.left as usize;
    let top = region.top as usize;

    let input_buf = [input[0].buf(), input[1].buf(), input[2].buf()];
    let sigma_buf = sigma_grid.buf();
    let (output0, output1, output2) = {
        let [a, b, c] = output;
        (a.buf_mut(), b.buf_mut(), c.buf_mut())
    };

    let expected_sigma_height = (height + top + 7) / 8 - top / 8;
    assert_eq!(sigma_buf.len(), width * expected_sigma_height);
    assert_eq!(input_buf[0].len(), width * height);
    assert_eq!(input_buf[1].len(), width * height);
    assert_eq!(input_buf[2].len(), width * height);
    assert_eq!(output0.len(), width * height);
    assert_eq!(output1.len(), width * height);
    assert_eq!(output2.len(), width * height);

    if height <= 6 {
        let it = output0.chunks_exact_mut(width).zip(output1.chunks_exact_mut(width)).zip(output2.chunks_exact_mut(width));
        for (y, ((output0, output1), output2)) in it.enumerate() {
            let input_rows: [[_; 7]; 3] = std::array::from_fn(|c| {
                std::array::from_fn(|idx| {
                    let y = mirror((y + idx) as isize - 3, height);
                    &input_buf[c][y * width..][..width]
                })
            });

            let image_y = y + top;
            let sigma_y = (image_y / 8) - (top / 8);
            let sigma_row = &sigma_buf[sigma_y * width..][..width];

            let output_rows = [output0, output1, output2];
            let row = EpfRow {
                input_rows,
                merged_input_rows: None,
                output_rows,
                width,
                x: left,
                y: image_y,
                sigma_row,
                epf_params,
                skip_inner: false,
            };
            handle_row_generic(row);
        }

        return;
    }

    let (output0_head, output0, output0_tail) = {
        let (output_head, tmp) = output0.split_at_mut(3 * width);
        let (output, output_tail) = tmp.split_at_mut(tmp.len() - 3 * width);
        (output_head, output, output_tail)
    };
    let (output1_head, output1, output1_tail) = {
        let (output_head, tmp) = output1.split_at_mut(3 * width);
        let (output, output_tail) = tmp.split_at_mut(tmp.len() - 3 * width);
        (output_head, output, output_tail)
    };
    let (output2_head, output2, output2_tail) = {
        let (output_head, tmp) = output2.split_at_mut(3 * width);
        let (output, output_tail) = tmp.split_at_mut(tmp.len() - 3 * width);
        (output_head, output, output_tail)
    };

    let mut jobs = vec![
        EpfJob {
            base_y: 0,
            output0: output0_head,
            output1: output1_head,
            output2: output2_head,
        },
        EpfJob {
            base_y: height - 3,
            output0: output0_tail,
            output1: output1_tail,
            output2: output2_tail,
        },
    ];
    let output0_it = output0.chunks_mut(8 * width);
    let output1_it = output1.chunks_mut(8 * width);
    let output2_it = output2.chunks_mut(8 * width);
    jobs.extend(output0_it
        .zip(output1_it)
        .zip(output2_it)
        .enumerate()
        .map(|(y8, ((output0, output1), output2))| EpfJob {
            base_y: y8 * 8 + 3,
            output0,
            output1,
            output2,
        }));

    pool.for_each_vec(
        jobs,
        |EpfJob {
             base_y,
             output0,
             output1,
             output2,
         }| {
            let it = output0.chunks_exact_mut(width).zip(output1.chunks_exact_mut(width)).zip(output2.chunks_exact_mut(width));
            for (dy, ((output0, output1), output2)) in it.enumerate() {
                let y = base_y + dy;
                let input_rows: [[_; 7]; 3] = std::array::from_fn(|c| {
                    std::array::from_fn(|idx| {
                        let y = mirror((y + idx) as isize - 3, height);
                        &input_buf[c][y * width..][..width]
                    })
                });
                let merged_input_rows: Option<[_; 3]> = if y >= 3 && y < height - 3 {
                    Some(std::array::from_fn(|c| {
                        &input_buf[c][(y - 3) * width..][..7 * width]
                    }))
                } else {
                    None
                };

                let image_y = y + top;
                let sigma_y = (image_y / 8) - (top / 8);
                let sigma_row = &sigma_buf[sigma_y * width..][..width];

                let mut skip_inner = false;
                if merged_input_rows.is_some() {
                    if let Some(handle_row_simd) = handle_row_simd {
                        skip_inner = true;
                        let output_rows = [&mut *output0, &mut *output1, &mut *output2];
                        let row = EpfRow {
                            input_rows,
                            merged_input_rows,
                            output_rows,
                            width,
                            x: left,
                            y: image_y,
                            sigma_row,
                            epf_params,
                            skip_inner,
                        };
                        handle_row_simd(row);
                    }
                }

                let output_rows = [output0, output1, output2];
                let row = EpfRow {
                    input_rows,
                    output_rows,
                    merged_input_rows,
                    width,
                    x: left,
                    y: image_y,
                    sigma_row,
                    epf_params,
                    skip_inner,
                };
                handle_row_generic(row);
            }
        },
    );
}

pub(crate) const fn epf_kernel<const STEP: usize>() -> (&'static [(isize, isize)], &'static [(isize, isize)]) {
    const EPF_KERNEL_DIST_1: [(isize, isize); 4] = [(0, -1), (-1, 0), (1, 0), (0, 1)];
    #[rustfmt::skip]
    const EPF_KERNEL_DIST_2: [(isize, isize); 12] = [
        (0, -2), (-1, -1), (0, -1), (1, -1),
        (-2, 0), (-1, 0), (1, 0), (2, 0),
        (-1, 1), (0, 1), (1, 1), (0, 2),
    ];
    const EPF_KERNEL_SIZE_0: [(isize, isize); 1] = [(0, 0)];
    const EPF_KERNEL_SIZE_1: [(isize, isize); 5] = [(0, -1), (-1, 0), (0, 0), (1, 0), (0, 1)];

    if STEP == 0 {
        (&EPF_KERNEL_DIST_2, &EPF_KERNEL_SIZE_1)
    } else if STEP == 1 {
        (&EPF_KERNEL_DIST_1, &EPF_KERNEL_SIZE_1)
    } else if STEP == 2 {
        (&EPF_KERNEL_DIST_1, &EPF_KERNEL_SIZE_0)
    } else {
        panic!()
    }
}

fn mirror(mut offset: isize, len: usize) -> usize {
    loop {
        if offset < 0 {
            offset = -(offset + 1);
        } else if (offset as usize) >= len {
            offset = (-(offset + 1)).wrapping_add_unsigned(len * 2);
        } else {
            return offset as usize;
        }
    }
}

pub(crate) fn epf_row<const STEP: usize>(epf_row: EpfRow<'_, '_>) {
    let EpfRow {
        input_rows,
        output_rows,
        width,
        x,
        y,
        sigma_row,
        epf_params,
        skip_inner,
        ..
    } = epf_row;
    let (kernel_offsets, dist_offsets) = epf_kernel::<STEP>();

    let step_multiplier = if STEP == 0 {
        epf_params.sigma.pass0_sigma_scale
    } else if STEP == 2 {
        epf_params.sigma.pass2_sigma_scale
    } else {
        1.0
    };
    let border_sad_mul = epf_params.sigma.border_sad_mul;
    let channel_scale = epf_params.channel_scale;

    let is_y_border = (y + 1) & 0b110 == 0;
    let sm = if is_y_border {
        [step_multiplier * border_sad_mul; 8]
    } else {
        let neg_x = 8 - (x & 7);
        let mut sm = [step_multiplier; 8];
        sm[neg_x & 7] *= border_sad_mul;
        sm[(neg_x + 7) & 7] *= border_sad_mul;
        sm
    };

    let padding = 3 - STEP;
    let (left_edge_width, right_edge_width) = if width < padding * 2 {
        let left_edge_width = width.saturating_sub(padding);
        (left_edge_width, width - left_edge_width)
    } else {
        let right_edge_width = ((width - padding * 2) & 7) + padding;
        (padding, right_edge_width)
    };
    let right_edge_start = width - right_edge_width;

    for dx in 0..left_edge_width {
        let sm_idx = dx & 7;
        let sigma_val = sigma_row[dx];
        if sigma_val < 0.3 {
            for c in 0..3 {
                output_rows[c][dx] = input_rows[c][3][dx];
            }
            continue;
        }

        let mut sum_weights = 1.0f32;
        let mut sum_channels: [f32; 3] = std::array::from_fn(|c| input_rows[c][3][dx]);

        for &(kx, ky) in kernel_offsets {
            let kernel_dy = 3 + ky;
            let kernel_dx = dx as isize + kx;
            let mut dist = 0f32;
            for c in 0..3 {
                let scale = channel_scale[c];
                for &(ix, iy) in dist_offsets {
                    let kernel_dy = (kernel_dy + iy) as usize;
                    let kernel_dx = mirror(kernel_dx + ix, width);
                    let base_dy = (3 + iy) as usize;
                    let base_dx = mirror(dx as isize + ix, width);

                    dist = scale.mul_add((input_rows[c][kernel_dy][kernel_dx] - input_rows[c][base_dy][base_dx]).abs(), dist);
                }
            }

            let weight = weight(
                dist,
                sigma_val,
                sm[sm_idx],
            );
            sum_weights += weight;

            let kernel_dy = kernel_dy as usize;
            let kernel_dx = mirror(kernel_dx, width);
            for (c, sum) in sum_channels.iter_mut().enumerate() {
                *sum = weight.mul_add(input_rows[c][kernel_dy][kernel_dx], *sum);
            }
        }

        for (c, sum) in sum_channels.into_iter().enumerate() {
            output_rows[c][dx] = sum / sum_weights;
        }
    }

    for dx in left_edge_width..width.saturating_sub(padding) {
        if skip_inner && dx < right_edge_start {
            continue;
        }

        let sm_idx = dx & 7;
        let sigma_val = sigma_row[dx];
        if sigma_val < 0.3 {
            for c in 0..3 {
                output_rows[c][dx] = input_rows[c][3][dx];
            }
            continue;
        }

        let mut sum_weights = 1.0f32;
        let mut sum_channels: [f32; 3] = std::array::from_fn(|c| input_rows[c][3][dx]);

        for &(kx, ky) in kernel_offsets {
            let kernel_dy = 3 + ky;
            let kernel_dx = dx as isize + kx;
            let mut dist = 0f32;
            for c in 0..3 {
                let scale = channel_scale[c];
                for &(ix, iy) in dist_offsets {
                    let kernel_dy = (kernel_dy + iy) as usize;
                    let kernel_dx = (kernel_dx + ix) as usize;
                    let base_dy = (3 + iy) as usize;
                    let base_dx = (dx as isize + ix) as usize;

                    dist = scale.mul_add((input_rows[c][kernel_dy][kernel_dx] - input_rows[c][base_dy][base_dx]).abs(), dist);
                }
            }

            let weight = weight(
                dist,
                sigma_val,
                sm[sm_idx],
            );
            sum_weights += weight;

            let kernel_dy = kernel_dy as usize;
            let kernel_dx = kernel_dx as usize;
            for (c, sum) in sum_channels.iter_mut().enumerate() {
                *sum = weight.mul_add(input_rows[c][kernel_dy][kernel_dx], *sum);
            }
        }

        for (c, sum) in sum_channels.into_iter().enumerate() {
            output_rows[c][dx] = sum / sum_weights;
        }
    }

    for dx in width.saturating_sub(padding)..width {
        let sm_idx = dx & 7;
        let sigma_val = sigma_row[dx];
        if sigma_val < 0.3 {
            for c in 0..3 {
                output_rows[c][dx] = input_rows[c][3][dx];
            }
            continue;
        }

        let mut sum_weights = 1.0f32;
        let mut sum_channels: [f32; 3] = std::array::from_fn(|c| input_rows[c][3][dx]);

        for &(kx, ky) in kernel_offsets {
            let kernel_dy = 3 + ky;
            let kernel_dx = dx as isize + kx;
            let mut dist = 0f32;
            for c in 0..3 {
                let scale = channel_scale[c];
                for &(ix, iy) in dist_offsets {
                    let kernel_dy = (kernel_dy + iy) as usize;
                    let kernel_dx = mirror(kernel_dx + ix, width);
                    let base_dy = (3 + iy) as usize;
                    let base_dx = mirror(dx as isize + ix, width);

                    dist = scale.mul_add((input_rows[c][kernel_dy][kernel_dx] - input_rows[c][base_dy][base_dx]).abs(), dist);
                }
            }

            let weight = weight(
                dist,
                sigma_val,
                sm[sm_idx],
            );
            sum_weights += weight;

            let kernel_dy = kernel_dy as usize;
            let kernel_dx = mirror(kernel_dx, width);
            for (c, sum) in sum_channels.iter_mut().enumerate() {
                *sum = weight.mul_add(input_rows[c][kernel_dy][kernel_dx], *sum);
            }
        }

        for (c, sum) in sum_channels.into_iter().enumerate() {
            output_rows[c][dx] = sum / sum_weights;
        }
    }
}

pub fn epf_step0(
    input: &[SimpleGrid<f32>; 3],
    output: &mut [SimpleGrid<f32>; 3],
    sigma_grid: &SimpleGrid<f32>,
    region: Region,
    epf_params: &EpfParams,
    pool: &JxlThreadPool,
) {
    unsafe {
        epf_common(
            input,
            output,
            sigma_grid,
            region,
            epf_params,
            pool,
            None,
            epf_row::<0>,
        )
    }
}

pub fn epf_step1(
    input: &[SimpleGrid<f32>; 3],
    output: &mut [SimpleGrid<f32>; 3],
    sigma_grid: &SimpleGrid<f32>,
    region: Region,
    epf_params: &EpfParams,
    pool: &JxlThreadPool,
) {
    unsafe {
        epf_common(
            input,
            output,
            sigma_grid,
            region,
            epf_params,
            pool,
            None,
            epf_row::<1>,
        )
    }
}

pub fn epf_step2(
    input: &[SimpleGrid<f32>; 3],
    output: &mut [SimpleGrid<f32>; 3],
    sigma_grid: &SimpleGrid<f32>,
    region: Region,
    epf_params: &EpfParams,
    pool: &JxlThreadPool,
) {
    unsafe {
        epf_common(
            input,
            output,
            sigma_grid,
            region,
            epf_params,
            pool,
            None,
            epf_row::<2>,
        )
    }
}

pub fn apply_gabor_like(fb: [&mut SimpleGrid<f32>; 3], weights_xyb: [[f32; 2]; 3]) -> Result<()> {
    for (fb, [weight1, weight2]) in fb.into_iter().zip(weights_xyb) {
        run_gabor_inner(fb, weight1, weight2)?;
    }
    Ok(())
}

#[inline(always)]
pub fn run_gabor_inner(
    fb: &mut jxl_grid::SimpleGrid<f32>,
    weight1: f32,
    weight2: f32,
) -> Result<()> {
    let global_weight = (1.0 + weight1 * 4.0 + weight2 * 4.0).recip();

    let tracker = fb.tracker();
    let width = fb.width();
    let height = fb.height();
    if width * height <= 1 {
        return Ok(());
    }

    let mut input = SimpleGrid::with_alloc_tracker(width, height + 2, tracker.as_ref())?;
    let input = input.buf_mut();
    input[width..][..width * height].copy_from_slice(fb.buf());
    input[..width].copy_from_slice(&fb.buf()[..width]);
    input[width * (height + 1)..][..width]
        .copy_from_slice(&fb.buf()[width * (height - 1)..][..width]);

    let input = &*input;
    let output = fb.buf_mut();

    if width == 1 {
        for idx in 0..height {
            output[idx] = (input[idx + 1]
                + (input[idx] + input[idx + 1] + input[idx + 1] + input[idx + 2]) * weight1
                + (input[idx] + input[idx + 2]) * weight2 * 2.0)
                * global_weight;
        }
        return Ok(());
    }

    let len = width * height - 2;
    let center = &input[width + 1..][..len];
    let sides = [
        &input[1..][..len],
        &input[width..][..len],
        &input[width + 2..][..len],
        &input[width * 2 + 1..][..len],
    ];
    let diags = [
        &input[..len],
        &input[2..][..len],
        &input[width * 2..][..len],
        &input[width * 2 + 2..][..len],
    ];

    for (idx, out) in output[1..][..len].iter_mut().enumerate() {
        *out = (center[idx]
            + (sides[0][idx] + sides[1][idx] + sides[2][idx] + sides[3][idx]) * weight1
            + (diags[0][idx] + diags[1][idx] + diags[2][idx] + diags[3][idx]) * weight2)
            * global_weight;
    }

    // left side
    let center = &input[width..];
    let sides = [
        input,
        &input[width..],
        &input[width + 1..],
        &input[width * 2..],
    ];
    let diags = [
        input,
        &input[1..],
        &input[width * 2..],
        &input[width * 2 + 1..],
    ];
    for idx in 0..height {
        let offset = idx * width;
        output[offset] = (center[offset]
            + (sides[0][offset] + sides[1][offset] + sides[2][offset] + sides[3][offset])
                * weight1
            + (diags[0][offset] + diags[1][offset] + diags[2][offset] + diags[3][offset])
                * weight2)
            * global_weight;
    }

    // right side
    let center = &input[width * 2 - 1..];
    let sides = [
        &input[width - 1..],
        &input[width * 2 - 2..],
        &input[width * 2 - 1..],
        &input[width * 3 - 1..],
    ];
    let diags = [
        &input[width - 2..],
        &input[width - 1..],
        &input[width * 3 - 2..],
        &input[width * 3 - 1..],
    ];
    for idx in 0..height {
        let offset = idx * width;
        output[width - 1 + offset] = (center[offset]
            + (sides[0][offset] + sides[1][offset] + sides[2][offset] + sides[3][offset])
                * weight1
            + (diags[0][offset] + diags[1][offset] + diags[2][offset] + diags[3][offset])
                * weight2)
            * global_weight;
    }
    Ok(())
}
