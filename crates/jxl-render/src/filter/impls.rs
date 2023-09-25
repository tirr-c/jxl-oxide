#[cfg(not(target_arch = "x86_64"))]
mod generic;
#[cfg(target_arch = "x86_64")]
mod x86_64;

#[cfg(not(target_arch = "x86_64"))]
pub use generic::*;

#[cfg(target_arch = "x86_64")]
pub use x86_64::*;

#[inline(always)]
fn run_gabor_inner(fb: &mut jxl_grid::SimpleGrid<f32>, weight1: f32, weight2: f32) {
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
