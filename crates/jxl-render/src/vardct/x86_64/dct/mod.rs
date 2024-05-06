use std::arch::x86_64::*;

use jxl_grid::{MutableSubgrid, SimdVector};

use super::super::dct_common::{self, DctDirection};

const LANE_SIZE: usize = 4;
type Lane = __m128;

#[inline(always)]
pub(crate) fn transpose_lane(lanes: &mut [Lane]) {
    let [row0, row1, row2, row3] = lanes else {
        panic!()
    };
    unsafe {
        _MM_TRANSPOSE4_PS(row0, row1, row2, row3);
    }
}

#[inline(always)]
pub(crate) fn dct_2d_x86_64_sse2(io: &mut MutableSubgrid<'_>, direction: DctDirection) {
    if io.width() % LANE_SIZE != 0 || io.height() % LANE_SIZE != 0 {
        return super::generic::dct_2d(io, direction);
    }

    let Some(mut io) = io.as_vectored() else {
        tracing::trace!("Input buffer is not aligned");
        return super::generic::dct_2d(io, direction);
    };

    if io.width() == 2 && io.height() == 8 {
        unsafe {
            return dct8x8(&mut io, direction);
        }
    }

    dct_2d_lane(&mut io, direction);
}

fn dct_2d_lane(io: &mut MutableSubgrid<'_, Lane>, direction: DctDirection) {
    let scratch_size = io.height().max(io.width() * LANE_SIZE) * 2;
    unsafe {
        let mut scratch_lanes = vec![_mm_setzero_ps(); scratch_size];
        column_dct_lane(io, &mut scratch_lanes, direction);
        row_dct_lane(io, &mut scratch_lanes, direction);
    }
}

#[inline]
unsafe fn dct4_vec_forward(v: Lane) -> Lane {
    const SEC0: f32 = 0.5411961;
    const SEC1: f32 = 1.306563;

    let v_rev = unsafe { _mm_shuffle_ps::<0b00011011>(v, v) };
    let mul = Lane::set([1.0, 1.0, -1.0, -1.0]);
    let addsub = v.muladd(mul, v_rev);

    let a = unsafe { _mm_shuffle_ps::<0b10011100>(addsub, addsub) };
    let mul_a = Lane::set([
        0.25,
        (std::f32::consts::FRAC_1_SQRT_2 / 2.0 + 0.25) * SEC0,
        -0.25,
        -0.25 * SEC1,
    ]);
    let b = unsafe { _mm_shuffle_ps::<0b11001001>(addsub, addsub) };
    let mul_b = Lane::set([
        0.25,
        (std::f32::consts::FRAC_1_SQRT_2 / 2.0 - 0.25) * SEC1,
        0.25,
        0.25 * SEC0,
    ]);
    a.muladd(mul_a, b.mul(mul_b))
}

#[inline]
pub(crate) unsafe fn dct4_vec_inverse(v: Lane) -> Lane {
    const SEC0: f32 = 0.5411961;
    const SEC1: f32 = 1.306563;

    let v_flip = unsafe { _mm_shuffle_ps::<0b01001110>(v, v) };
    let mul_a = Lane::set([1.0, (std::f32::consts::SQRT_2 + 1.0) * SEC0, -1.0, -SEC1]);
    let mul_b = Lane::set([1.0, SEC0, 1.0, (std::f32::consts::SQRT_2 - 1.0) * SEC1]);
    let tmp = v.muladd(mul_a, v_flip.mul(mul_b));

    let tmp_a = unsafe { _mm_shuffle_ps::<0b00101000>(tmp, tmp) };
    let tmp_b = unsafe { _mm_shuffle_ps::<0b01111101>(tmp, tmp) };
    let mul = Lane::set([1.0, 1.0, -1.0, -1.0]);
    tmp_b.muladd(mul, tmp_a)
}

#[inline]
unsafe fn dct8_vec_forward(vl: Lane, vr: Lane) -> (Lane, Lane) {
    #[allow(clippy::excessive_precision)]
    let sec_vec = Lane::set([
        0.2548977895520796,
        0.30067244346752264,
        0.4499881115682078,
        1.2814577238707527,
    ]);
    let vr_rev = unsafe { _mm_shuffle_ps::<0b00011011>(vr, vr) };
    let input0 = vl.add(vr_rev).mul(Lane::splat_f32(0.5));
    let input1 = vl.sub(vr_rev).mul(sec_vec);
    let output0 = dct4_vec_forward(input0);
    let output1 = dct4_vec_forward(input1);
    let output1_shifted =
        unsafe { _mm_castsi128_ps(_mm_srli_si128::<4>(_mm_castps_si128(output1))) };
    let output1_mul = Lane::set([std::f32::consts::SQRT_2, 1.0, 1.0, 1.0]);
    let output1 = output1.muladd(output1_mul, output1_shifted);
    (unsafe { _mm_unpacklo_ps(output0, output1) }, unsafe {
        _mm_unpackhi_ps(output0, output1)
    })
}

#[inline]
pub(crate) unsafe fn dct8_vec_inverse(vl: Lane, vr: Lane) -> (Lane, Lane) {
    #[allow(clippy::excessive_precision)]
    let sec_vec = Lane::set([
        0.5097955791041592,
        0.6013448869350453,
        0.8999762231364156,
        2.5629154477415055,
    ]);
    let input0 = unsafe { _mm_shuffle_ps::<0b10001000>(vl, vr) };
    let input1 = unsafe { _mm_shuffle_ps::<0b11011101>(vl, vr) };
    let input1_shifted = unsafe { _mm_castsi128_ps(_mm_slli_si128::<4>(_mm_castps_si128(input1))) };
    let input1_mul = Lane::set([std::f32::consts::SQRT_2, 1.0, 1.0, 1.0]);
    let input1 = input1.muladd(input1_mul, input1_shifted);
    let output0 = dct4_vec_inverse(input0);
    let output1 = dct4_vec_inverse(input1);
    let output1 = output1.mul(sec_vec);
    let sub = output0.sub(output1);
    (output0.add(output1), unsafe {
        _mm_shuffle_ps::<0b00011011>(sub, sub)
    })
}

unsafe fn dct8x8(io: &mut MutableSubgrid<'_, Lane>, direction: DctDirection) {
    let (mut col0, mut col1) = io.split_horizontal(1);

    if direction == DctDirection::Forward {
        dct8_forward(&mut col0);
        dct8_forward(&mut col1);
        for y in 0..8 {
            let row = io.get_row_mut(y);
            let (vl, vr) = dct8_vec_forward(row[0], row[1]);
            row[0] = vl;
            row[1] = vr;
        }
    } else {
        dct8_inverse(&mut col0);
        dct8_inverse(&mut col1);
        for y in 0..8 {
            let row = io.get_row_mut(y);
            let (vl, vr) = dct8_vec_inverse(row[0], row[1]);
            row[0] = vl;
            row[1] = vr;
        }
    }
}

unsafe fn column_dct_lane(
    io: &mut MutableSubgrid<'_, Lane>,
    scratch: &mut [Lane],
    direction: DctDirection,
) {
    let width = io.width();
    let height = io.height();
    let (io_lanes, scratch_lanes) = scratch[..height * 2].split_at_mut(height);
    for x in 0..width {
        for (y, input) in io_lanes.iter_mut().enumerate() {
            *input = io.get(x, y);
        }
        dct(io_lanes, scratch_lanes, direction);
        for (y, output) in io_lanes.chunks_exact_mut(LANE_SIZE).enumerate() {
            transpose_lane(output);
            for (dy, output) in output.iter_mut().enumerate() {
                *io.get_mut(x, y * LANE_SIZE + dy) = *output;
            }
        }
    }
}

unsafe fn row_dct_lane(
    io: &mut MutableSubgrid<'_, Lane>,
    scratch: &mut [Lane],
    direction: DctDirection,
) {
    let width = io.width() * LANE_SIZE;
    let height = io.height();
    let (io_lanes, scratch_lanes) = scratch[..width * 2].split_at_mut(width);
    for y in (0..height).step_by(LANE_SIZE) {
        for (x, input) in io_lanes.chunks_exact_mut(LANE_SIZE).enumerate() {
            for (dy, input) in input.iter_mut().enumerate() {
                *input = io.get(x, y + dy);
            }
        }
        dct(io_lanes, scratch_lanes, direction);
        for (x, output) in io_lanes.chunks_exact_mut(LANE_SIZE).enumerate() {
            transpose_lane(output);
            for (dy, output) in output.iter_mut().enumerate() {
                *io.get_mut(x, y + dy) = *output;
            }
        }
    }
}

#[inline]
unsafe fn dct4_forward(input: [Lane; 4]) -> [Lane; 4] {
    let sec0 = Lane::splat_f32(0.5411961 / 4.0);
    let sec1 = Lane::splat_f32(1.306563 / 4.0);
    let quarter = Lane::splat_f32(0.25);
    let sqrt2 = Lane::splat_f32(std::f32::consts::SQRT_2);

    let sum03 = input[0].add(input[3]);
    let sum12 = input[1].add(input[2]);
    let tmp0 = input[0].sub(input[3]).mul(sec0);
    let tmp1 = input[1].sub(input[2]).mul(sec1);
    let out0 = tmp0.add(tmp1);
    let out1 = tmp0.sub(tmp1);

    [
        sum03.add(sum12).mul(quarter),
        out0.muladd(sqrt2, out1),
        sum03.sub(sum12).mul(quarter),
        out1,
    ]
}

#[inline]
pub(crate) unsafe fn dct4_inverse(input: [Lane; 4]) -> [Lane; 4] {
    let sec0 = Lane::splat_f32(0.5411961);
    let sec1 = Lane::splat_f32(1.306563);
    let sqrt2 = Lane::splat_f32(std::f32::consts::SQRT_2);

    let tmp0 = input[1].mul(sqrt2);
    let tmp1 = input[1].add(input[3]);
    let out0 = tmp0.add(tmp1).mul(sec0);
    let out1 = tmp0.sub(tmp1).mul(sec1);
    let sum02 = input[0].add(input[2]);
    let sub02 = input[0].sub(input[2]);

    [
        sum02.add(out0),
        sub02.add(out1),
        sub02.sub(out1),
        sum02.sub(out0),
    ]
}

#[inline]
unsafe fn dct8_forward(io: &mut MutableSubgrid<'_, Lane>) {
    assert!(io.height() == 8);
    let half = Lane::splat_f32(0.5);
    let sqrt2 = Lane::splat_f32(std::f32::consts::SQRT_2);
    let sec = dct_common::sec_half_small(8);

    let input0 = [
        io.get(0, 0).add(io.get(0, 7)).mul(half),
        io.get(0, 1).add(io.get(0, 6)).mul(half),
        io.get(0, 2).add(io.get(0, 5)).mul(half),
        io.get(0, 3).add(io.get(0, 4)).mul(half),
    ];
    let input1 = [
        io.get(0, 0)
            .sub(io.get(0, 7))
            .mul(Lane::splat_f32(sec[0] / 2.0)),
        io.get(0, 1)
            .sub(io.get(0, 6))
            .mul(Lane::splat_f32(sec[1] / 2.0)),
        io.get(0, 2)
            .sub(io.get(0, 5))
            .mul(Lane::splat_f32(sec[2] / 2.0)),
        io.get(0, 3)
            .sub(io.get(0, 4))
            .mul(Lane::splat_f32(sec[3] / 2.0)),
    ];
    let output0 = dct4_forward(input0);
    for (idx, v) in output0.into_iter().enumerate() {
        *io.get_mut(0, idx * 2) = v;
    }
    let mut output1 = dct4_forward(input1);
    output1[0] = output1[0].mul(sqrt2);
    for idx in 0..3 {
        *io.get_mut(0, idx * 2 + 1) = output1[idx].add(output1[idx + 1]);
    }
    *io.get_mut(0, 7) = output1[3];
}

#[inline]
unsafe fn dct8_inverse(io: &mut MutableSubgrid<'_, Lane>) {
    assert!(io.height() == 8);
    let sqrt2 = Lane::splat_f32(std::f32::consts::SQRT_2);
    let sec = dct_common::sec_half_small(8);

    let input0 = [io.get(0, 0), io.get(0, 2), io.get(0, 4), io.get(0, 6)];
    let input1 = [
        io.get(0, 1).mul(sqrt2),
        io.get(0, 3).add(io.get(0, 1)),
        io.get(0, 5).add(io.get(0, 3)),
        io.get(0, 7).add(io.get(0, 5)),
    ];
    let output0 = dct4_inverse(input0);
    let output1 = dct4_inverse(input1);
    for (idx, &sec) in sec.iter().enumerate() {
        let r = output1[idx].mul(Lane::splat_f32(sec));
        *io.get_mut(0, idx) = output0[idx].add(r);
        *io.get_mut(0, 7 - idx) = output0[idx].sub(r);
    }
}

unsafe fn dct(io: &mut [Lane], scratch: &mut [Lane], direction: DctDirection) {
    let n = io.len();
    assert!(scratch.len() == n);

    if n == 0 {
        return;
    }
    if n == 1 {
        return;
    }

    let half = Lane::splat_f32(0.5);
    if n == 2 {
        let tmp0 = io[0].add(io[1]);
        let tmp1 = io[0].sub(io[1]);
        if direction == DctDirection::Forward {
            io[0] = tmp0.mul(half);
            io[1] = tmp1.mul(half);
        } else {
            io[0] = tmp0;
            io[1] = tmp1;
        }
        return;
    }

    if n == 4 {
        if direction == DctDirection::Forward {
            io.copy_from_slice(&dct4_forward([io[0], io[1], io[2], io[3]]));
        } else {
            io.copy_from_slice(&dct4_inverse([io[0], io[1], io[2], io[3]]));
        }
        return;
    }

    let sqrt2 = Lane::splat_f32(std::f32::consts::SQRT_2);
    if n == 8 {
        if direction == DctDirection::Forward {
            dct8_forward(&mut MutableSubgrid::from_buf(io, 1, 8, 1));
        } else {
            dct8_inverse(&mut MutableSubgrid::from_buf(io, 1, 8, 1));
        }
        return;
    }

    assert!(n.is_power_of_two());

    if direction == DctDirection::Forward {
        let (input0, input1) = scratch.split_at_mut(n / 2);
        for (idx, &sec) in dct_common::sec_half(n).iter().enumerate() {
            input0[idx] = io[idx].add(io[n - idx - 1]).mul(half);
            input1[idx] = io[idx].sub(io[n - idx - 1]).mul(Lane::splat_f32(sec / 2.0));
        }
        let (output0, output1) = io.split_at_mut(n / 2);
        dct(input0, output0, DctDirection::Forward);
        dct(input1, output1, DctDirection::Forward);
        for (idx, v) in input0.iter().enumerate() {
            io[idx * 2] = *v;
        }
        input1[0] = input1[0].mul(sqrt2);
        for idx in 0..(n / 2 - 1) {
            io[idx * 2 + 1] = input1[idx].add(input1[idx + 1]);
        }
        io[n - 1] = input1[n / 2 - 1];
    } else {
        let (input0, input1) = scratch.split_at_mut(n / 2);
        for idx in 1..(n / 2) {
            let idx = n / 2 - idx;
            input0[idx] = io[idx * 2];
            input1[idx] = io[idx * 2 + 1].add(io[idx * 2 - 1]);
        }
        input0[0] = io[0];
        input1[0] = io[1].mul(sqrt2);
        let (output0, output1) = io.split_at_mut(n / 2);
        dct(input0, output0, DctDirection::Inverse);
        dct(input1, output1, DctDirection::Inverse);
        for (idx, &sec) in dct_common::sec_half(n).iter().enumerate() {
            let r = input1[idx].mul(Lane::splat_f32(sec));
            output0[idx] = input0[idx].add(r);
            output1[n / 2 - idx - 1] = input0[idx].sub(r);
        }
    }
}
