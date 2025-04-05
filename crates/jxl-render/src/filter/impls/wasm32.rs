use jxl_frame::{FrameHeader, filter::EpfParams};
use jxl_grid::{AlignedGrid, MutableSubgrid};
use jxl_threadpool::JxlThreadPool;

use crate::Region;
use crate::filter::epf::run_epf_rows;

mod epf;

pub fn epf<const STEP: usize>(
    input: &mut [MutableSubgrid<f32>; 3],
    output: &mut [MutableSubgrid<f32>; 3],
    color_padded_region: Region,
    frame_header: &FrameHeader,
    sigma_grid_map: &[Option<&AlignedGrid<f32>>],
    epf_params: &EpfParams,
    pool: &JxlThreadPool,
) {
    unsafe {
        run_epf_rows(
            input,
            output,
            color_padded_region,
            frame_header,
            sigma_grid_map,
            epf_params,
            pool,
            Some(epf::epf_row_wasm32_simd128::<STEP>),
            super::generic::epf::epf_row::<STEP>,
        )
    }
}

pub use super::generic::apply_gabor_like;
