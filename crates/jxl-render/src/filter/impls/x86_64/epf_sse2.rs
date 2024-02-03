use std::arch::x86_64::*;

use jxl_grid::SimdVector;

use crate::filter::impls::generic::EpfRow;

type Vector = __m128;

#[inline]
unsafe fn weight_sse2(scaled_distance: Vector, sigma: f32, step_multiplier: Vector) -> Vector {
    let neg_inv_sigma = Vector::splat_f32(6.6 * (std::f32::consts::FRAC_1_SQRT_2 - 1.0) / sigma)
        .mul(step_multiplier);
    let result = scaled_distance.muladd(neg_inv_sigma, Vector::splat_f32(1.0));
    _mm_max_ps(result, Vector::zero())
}

pub(crate) unsafe fn epf_row_x86_64_sse2<const STEP: usize>(epf_row: EpfRow<'_, '_>) {
    let EpfRow {
        input_rows,
        output_rows,
        width,
        x,
        y,
        sigma_row,
        epf_params,
        ..
    } = epf_row;
    let (kernel_offsets, dist_offsets) = super::super::generic::epf_kernel::<STEP>();

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
    todo!()
}
