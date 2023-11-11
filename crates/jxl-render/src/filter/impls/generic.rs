#![allow(dead_code)]

use jxl_grid::SimpleGrid;

pub(crate) struct EpfRow<'buf> {
    pub(crate) input_buf: [&'buf [f32]; 3],
    pub(crate) output_buf_rows: [&'buf mut [f32]; 3],
    pub(crate) width: usize,
    pub(crate) y8: usize,
    pub(crate) dy: usize,
    pub(crate) sigma_grid: &'buf SimpleGrid<f32>,
    pub(crate) channel_scale: [f32; 3],
    pub(crate) border_sad_mul: f32,
    pub(crate) step_multiplier: f32,
}

pub(crate) unsafe fn epf_common<'buf>(
    input: &'buf [SimpleGrid<f32>; 3],
    output: &'buf mut [SimpleGrid<f32>; 3],
    sigma_grid: &'buf SimpleGrid<f32>,
    channel_scale: [f32; 3],
    border_sad_mul: f32,
    step_multiplier: f32,
    handle_row: for<'a> unsafe fn(EpfRow<'a>),
) {
    let width = input[0].width();
    let height = input[0].height();
    assert!(width % 8 == 0);

    let input_buf = [input[0].buf(), input[1].buf(), input[2].buf()];
    let (output0, output1, output2) = {
        let [a, b, c] = output;
        (a.buf_mut(), b.buf_mut(), c.buf_mut())
    };
    assert_eq!(input_buf[0].len(), width * height);
    assert_eq!(input_buf[1].len(), width * height);
    assert_eq!(input_buf[2].len(), width * height);
    assert_eq!(output0.len(), width * height);
    assert_eq!(output1.len(), width * height);
    assert_eq!(output2.len(), width * height);

    let height = height - 6;
    let output0_it = output0[3 * width..][..(height * width)].chunks_mut(8 * width);
    let output1_it = output1[3 * width..][..(height * width)].chunks_mut(8 * width);
    let output2_it = output2[3 * width..][..(height * width)].chunks_mut(8 * width);

    rayon_core::scope(|scope| {
        for (y8, ((output0, output1), output2)) in output0_it.zip(output1_it).zip(output2_it).enumerate() {
            scope.spawn(move |_| {
                for dy in 0..(height - y8 * 8).min(8) {
                    let output_buf_rows = [
                        &mut output0[dy * width..][..width],
                        &mut output1[dy * width..][..width],
                        &mut output2[dy * width..][..width],
                    ];
                    let row = EpfRow {
                        input_buf,
                        output_buf_rows,
                        width,
                        y8,
                        dy,
                        sigma_grid,
                        channel_scale,
                        border_sad_mul,
                        step_multiplier,
                    };

                    unsafe { handle_row(row); }
                }
            });
        }
    });
}

pub fn epf_step0(
    input: &[SimpleGrid<f32>; 3],
    output: &mut [SimpleGrid<f32>; 3],
    sigma_grid: &SimpleGrid<f32>,
    channel_scale: [f32; 3],
    border_sad_mul: f32,
    step_multiplier: f32,
) {
    // SAFETY: row handler is safe.
    unsafe {
        epf_common(
            input,
            output,
            sigma_grid,
            channel_scale,
            border_sad_mul,
            step_multiplier,
            epf_row_step0,
        );
    }
}

pub fn epf_step1(
    input: &[SimpleGrid<f32>; 3],
    output: &mut [SimpleGrid<f32>; 3],
    sigma_grid: &SimpleGrid<f32>,
    channel_scale: [f32; 3],
    border_sad_mul: f32,
    step_multiplier: f32,
) {
    // SAFETY: row handler is safe.
    unsafe {
        epf_common(
            input,
            output,
            sigma_grid,
            channel_scale,
            border_sad_mul,
            step_multiplier,
            epf_row_step1,
        );
    }
}

pub fn epf_step2(
    input: &[SimpleGrid<f32>; 3],
    output: &mut [SimpleGrid<f32>; 3],
    sigma_grid: &SimpleGrid<f32>,
    channel_scale: [f32; 3],
    border_sad_mul: f32,
    step_multiplier: f32,
) {
    // SAFETY: row handler is safe.
    unsafe {
        epf_common(
            input,
            output,
            sigma_grid,
            channel_scale,
            border_sad_mul,
            step_multiplier,
            epf_row_step2,
        );
    }
}

#[inline]
fn weight(scaled_distance: f32, sigma: f32, step_multiplier: f32) -> f32 {
    let inv_sigma = step_multiplier * 6.6 * (1.0 - std::f32::consts::FRAC_1_SQRT_2) / sigma;
    (1.0 - scaled_distance * inv_sigma).max(0.0)
}

macro_rules! define_epf {
    { $($v:vis fn $name:ident ($width:ident, $kernel_offsets:expr, $dist_offsets:expr $(,)?); )* } => {
        $(
            $v fn $name(epf_row: EpfRow<'_>) {
                let EpfRow {
                    input_buf,
                    mut output_buf_rows,
                    width,
                    y8,
                    dy,
                    sigma_grid,
                    channel_scale,
                    border_sad_mul,
                    step_multiplier,
                } = epf_row;

                let y = y8 * 8 + dy;
                let $width = width as isize;

                let is_y_border = dy == 0 || dy == 7;
                let sm = if is_y_border {
                    [border_sad_mul * step_multiplier; 8]
                } else {
                    [
                        border_sad_mul * step_multiplier,
                        step_multiplier,
                        step_multiplier,
                        step_multiplier,
                        step_multiplier,
                        step_multiplier,
                        step_multiplier,
                        border_sad_mul * step_multiplier,
                    ]
                };

                for x8 in 1..width / 8 {
                    let base_x = x8 * 8;
                    let base_idx = (y + 3) * width + base_x;

                    let Some(&sigma_val) = sigma_grid.get(x8 - 1, y8) else { break; };
                    if sigma_val < 0.3 {
                        for (input_ch, output_ch) in input_buf.iter().zip(output_buf_rows.iter_mut()) {
                            output_ch[base_idx..][..8].copy_from_slice(&input_ch[base_x..][..8]);
                        }
                        continue;
                    }

                    for (dx, sm) in sm.into_iter().enumerate() {
                        let base_idx = base_idx + dx;
                        let base_x = base_x + dx;

                        let mut sum_weights = 1.0f32;
                        let mut sum_channels = [0.0f32; 3];
                        for (sum, ch) in sum_channels.iter_mut().zip(input_buf) {
                            *sum = ch[base_idx];
                        }

                        for offset in $kernel_offsets {
                            let kernel_idx = base_idx.wrapping_add_signed(offset);
                            let mut dist = 0.0f32;
                            for (ch, scale) in input_buf.iter().zip(channel_scale) {
                                for offset in $dist_offsets {
                                    let base_idx = base_idx.wrapping_add_signed(offset);
                                    let kernel_idx = kernel_idx.wrapping_add_signed(offset);
                                    dist = scale.mul_add((ch[base_idx] - ch[kernel_idx]).abs(), dist);
                                }
                            }

                            let weight = weight(
                                dist,
                                sigma_val,
                                sm,
                            );
                            sum_weights += weight;

                            for (sum, ch) in sum_channels.iter_mut().zip(input_buf) {
                                *sum = weight.mul_add(ch[kernel_idx], *sum);
                            }
                        }

                        for (sum, ch) in sum_channels.into_iter().zip(output_buf_rows.iter_mut()) {
                            ch[base_x] = sum / sum_weights;
                        }
                    }
                }
            }
        )*
    };
}

define_epf! {
    pub(crate) fn epf_row_step0(
        width,
        [
            -2 * width,
            -1 - width, -width, 1 - width,
            -2, -1, 1, 2,
            width - 1, width, width + 1,
            2 * width,
        ],
        [-width, -1, 0, 1, width],
    );
    pub(crate) fn epf_row_step1(
        width,
        [-width, -1, 1, width],
        [-width, -1, 0, 1, width],
    );
    pub(crate) fn epf_row_step2(
        width,
        [-width, -1, 1, width],
        [0isize],
    );
}

pub fn apply_gabor_like(fb: [&mut SimpleGrid<f32>; 3], weights_xyb: [[f32; 2]; 3]) {
    for (fb, [weight1, weight2]) in fb.into_iter().zip(weights_xyb) {
        run_gabor_inner(fb, weight1, weight2);
    }
}

#[inline(always)]
pub fn run_gabor_inner(fb: &mut jxl_grid::SimpleGrid<f32>, weight1: f32, weight2: f32) {
    let global_weight = (1.0 + weight1 * 4.0 + weight2 * 4.0).recip();

    let width = fb.width();
    let height = fb.height();
    if width * height <= 1 {
        return;
    }

    let mut input = vec![0f32; width * (height + 2)];
    input[width..][..width * height].copy_from_slice(fb.buf());
    input[..width].copy_from_slice(&fb.buf()[..width]);
    input[width * (height + 1)..][..width].copy_from_slice(&fb.buf()[width * (height - 1)..][..width]);

    let input = &*input;
    let output = fb.buf_mut();

    if width == 1 {
        for idx in 0..height {
            output[idx] = (
                input[idx + 1] +
                (input[idx] + input[idx + 1] + input[idx + 1] + input[idx + 2]) * weight1 +
                (input[idx] + input[idx + 2]) * weight2 * 2.0
            ) * global_weight;
        }
        return;
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
        *out = (
            center[idx] +
            (sides[0][idx] + sides[1][idx] + sides[2][idx] + sides[3][idx]) * weight1 +
            (diags[0][idx] + diags[1][idx] + diags[2][idx] + diags[3][idx]) * weight2
        ) * global_weight;
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
        output[offset] = (
            center[offset] +
            (sides[0][offset] + sides[1][offset] + sides[2][offset] + sides[3][offset]) * weight1 +
            (diags[0][offset] + diags[1][offset] + diags[2][offset] + diags[3][offset]) * weight2
        ) * global_weight;
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
        output[width - 1 + offset] = (
            center[offset] +
            (sides[0][offset] + sides[1][offset] + sides[2][offset] + sides[3][offset]) * weight1 +
            (diags[0][offset] + diags[1][offset] + diags[2][offset] + diags[3][offset]) * weight2
        ) * global_weight;
    }
}
