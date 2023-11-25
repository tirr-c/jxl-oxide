use jxl_grid::SimpleGrid;

pub fn apply_gabor_like(
    fb: [&mut SimpleGrid<f32>; 3],
    weights_xyb: [[f32; 2]; 3],
) -> crate::Result<()> {
    tracing::debug!("Running gaborish");
    super::impls::apply_gabor_like(fb, weights_xyb)
}
