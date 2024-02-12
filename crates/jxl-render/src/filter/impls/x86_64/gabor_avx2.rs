use crate::filter::gabor::GaborRow;

#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
pub(super) unsafe fn run_gabor_row_x86_64_avx2(row: GaborRow) {
    super::super::generic::gabor::run_gabor_row_generic(row)
}
