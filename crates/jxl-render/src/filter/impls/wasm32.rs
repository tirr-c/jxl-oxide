use jxl_frame::{filter::EpfParams, FrameHeader};
use jxl_grid::SimpleGrid;
use jxl_threadpool::JxlThreadPool;

use crate::filter::epf::run_epf_rows;
use crate::Region;

mod epf;

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
            Some(epf::epf_row_wasm32_simd128::<STEP>),
            super::generic::epf::epf_row::<STEP>,
        )
    }
}

pub use super::generic::apply_gabor_like;
