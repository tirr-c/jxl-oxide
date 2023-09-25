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
    let input = fb.clone();

    let input = input.buf();
    let output = fb.buf_mut();

    let len = width * (height - 2) - 2;
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

    for (idx, out) in output[width + 1..][..len].iter_mut().enumerate() {
        *out = (
            center[idx] +
            (sides[0][idx] + sides[1][idx] + sides[2][idx] + sides[3][idx]) * weight1 +
            (diags[0][idx] + diags[1][idx] + diags[2][idx] + diags[3][idx]) * weight2
        ) * global_weight;
    }

    // top side
    let len = width - 2;
    let center = &input[1..][..len];
    let sides = [
        &input[1..][..len],
        &input[..len],
        &input[2..][..len],
        &input[width + 1..][..len],
    ];
    let diags = [
        &input[..len],
        &input[2..][..len],
        &input[width..][..len],
        &input[width + 2..][..len],
    ];

    for (idx, out) in output[1..][..len].iter_mut().enumerate() {
        *out = (
            center[idx] +
            (sides[0][idx] + sides[1][idx] + sides[2][idx] + sides[3][idx]) * weight1 +
            (diags[0][idx] + diags[1][idx] + diags[2][idx] + diags[3][idx]) * weight2
        ) * global_weight;
    }

    // bottom side
    let len = width - 2;
    let base = width * (height - 1);
    let center = &input[base + 1..][..len];
    let sides = [
        &input[base - width + 1..][..len],
        &input[base..][..len],
        &input[base + 2..][..len],
        &input[base + 1..][..len],
    ];
    let diags = [
        &input[base - width..][..len],
        &input[base - width + 2..][..len],
        &input[base..][..len],
        &input[base + 2..][..len],
    ];

    for (idx, out) in output[base + 1..][..len].iter_mut().enumerate() {
        *out = (
            center[idx] +
            (sides[0][idx] + sides[1][idx] + sides[2][idx] + sides[3][idx]) * weight1 +
            (diags[0][idx] + diags[1][idx] + diags[2][idx] + diags[3][idx]) * weight2
        ) * global_weight;
    }

    // left side
    let len = height - 2;
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
    for idx in 0..len {
        output[width + idx * width] = (
            center[idx * width] +
            (sides[0][idx * width] + sides[1][idx * width] + sides[2][idx * width] + sides[3][idx * width]) * weight1 +
            (diags[0][idx * width] + diags[1][idx * width] + diags[2][idx * width] + diags[3][idx * width]) * weight2
        ) * global_weight;
    }

    // right side
    let len = height - 2;
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
    for idx in 0..len {
        output[width * 2 - 1 + idx * width] = (
            center[idx * width] +
            (sides[0][idx * width] + sides[1][idx * width] + sides[2][idx * width] + sides[3][idx * width]) * weight1 +
            (diags[0][idx * width] + diags[1][idx * width] + diags[2][idx * width] + diags[3][idx * width]) * weight2
        ) * global_weight;
    }

    // corners
    output[0] = (
        input[0] +
        (input[0] + input[0] + input[1] + input[width]) * weight1 +
        (input[0] + input[1] + input[width] + input[width + 1]) * weight2
    ) * global_weight;
    output[width - 1] = (
        input[width - 1] +
        (input[width - 1] + input[width - 1] + input[width - 2] + input[width * 2 - 1]) * weight1 +
        (input[width - 1] + input[width - 2] + input[width * 2 - 2] + input[width * 2 - 1]) * weight2
    ) * global_weight;
    output[base] = (
        input[base] +
        (input[base] + input[base] + input[base - width] + input[base + 1]) * weight1 +
        (input[base] + input[base - width] + input[base - width + 1] + input[base + 1]) * weight2
    ) * global_weight;
    let last = width * height - 1;
    output[last] = (
        input[last] +
        (input[last] + input[last] + input[last - width] + input[last - 1]) * weight1 +
        (input[last] + input[last - width] + input[last - width - 1] + input[last - 1]) * weight2
    ) * global_weight;
}
