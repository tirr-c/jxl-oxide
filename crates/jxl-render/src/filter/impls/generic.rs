#![allow(dead_code)]

use jxl_frame::{filter::EpfParams, FrameHeader};
use jxl_grid::SimpleGrid;
use jxl_threadpool::JxlThreadPool;

use crate::{filter::epf::run_epf_rows, Region, Result};

pub(crate) mod epf;

pub fn epf<const STEP: usize>(
    input: &[SimpleGrid<f32>; 3],
    output: &mut [SimpleGrid<f32>; 3],
    frame_header: &FrameHeader,
    sigma_grid_map: &[Option<&SimpleGrid<f32>>],
    region: Region,
    epf_params: &EpfParams,
    pool: &JxlThreadPool,
) {
    unsafe {
        run_epf_rows(
            input,
            output,
            frame_header,
            sigma_grid_map,
            region,
            epf_params,
            pool,
            None,
            epf::epf_row::<STEP>,
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
