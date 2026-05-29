#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::float32x4_t;

use crate::AlignedGrid;
use crate::MutableSubgrid;
use crate::SharedSubgrid;

#[test]
fn shared_subgrid_slices() {
    let grid = AlignedGrid::<u32>::with_alloc_tracker(128, 128, None).unwrap();
    let shared = grid.as_subgrid();

    let (top, bottom) = shared.split_vertical(64);
    assert_eq!(top.width(), 128);
    assert_eq!(top.height(), 64);
    assert_eq!(bottom.width(), 128);
    assert_eq!(bottom.height(), 64);

    let (tl, tr) = top.split_horizontal(64);
    assert_eq!(tl.width(), 64);
    assert_eq!(tl.height(), 64);
    assert_eq!(tr.width(), 64);
    assert_eq!(tr.height(), 64);

    let (tr, empty) = tr.split_vertical(64);
    assert_eq!(tr.height(), 64);
    assert_eq!(empty.height(), 0);
    assert!(empty.try_get_ref(0, 0).is_none());

    let (tr, empty) = tr.split_horizontal(64);
    assert_eq!(tr.width(), 64);
    assert_eq!(empty.width(), 0);
    assert!(empty.try_get_ref(0, 0).is_none());
}

#[test]
fn mutable_subgrid_slices() {
    let mut grid = AlignedGrid::<u32>::with_alloc_tracker(128, 128, None).unwrap();
    let mut mutable = grid.as_subgrid_mut();

    let (mut top, bottom) = mutable.split_vertical(64);
    assert_eq!(top.width(), 128);
    assert_eq!(top.height(), 64);
    assert_eq!(bottom.width(), 128);
    assert_eq!(bottom.height(), 64);

    let (tl, mut tr) = top.split_horizontal(64);
    assert_eq!(tl.width(), 64);
    assert_eq!(tl.height(), 64);
    assert_eq!(tr.width(), 64);
    assert_eq!(tr.height(), 64);

    let (mut tr, mut empty) = tr.split_vertical(64);
    assert_eq!(tr.height(), 64);
    assert_eq!(empty.height(), 0);
    assert!(empty.try_get_mut(0, 0).is_none());

    let (mut tr, mut empty) = tr.split_horizontal(64);
    assert_eq!(tr.width(), 64);
    assert_eq!(empty.width(), 0);
    assert!(empty.try_get_mut(0, 0).is_none());

    *tr.get_mut(0, 0) = 42;
    assert_eq!(grid.get(64, 0), 42);
}

#[test]
fn mutable_subgrid_split_merge() {
    let mut grid = AlignedGrid::<u32>::with_alloc_tracker(128, 128, None).unwrap();
    let mut mutable = grid.as_subgrid_mut();

    let bottom = mutable.split_vertical_in_place(64);
    let mut top = mutable;
    assert_eq!(top.height(), 64);
    assert_eq!(bottom.height(), 64);

    let mut tr = top.split_horizontal_in_place(64);
    let mut tl = top;
    assert_eq!(tl.width(), 64);
    assert_eq!(tr.width(), 64);

    let empty0 = tr.split_vertical_in_place(64);
    let empty1 = tr.split_horizontal_in_place(64);
    assert_eq!(empty0.height(), 0);
    assert_eq!(empty1.width(), 0);

    tr.merge_horizontal_in_place(empty1);
    tr.merge_vertical_in_place(empty0);
    tl.merge_horizontal_in_place(tr);
    tl.merge_vertical_in_place(bottom);
    assert_eq!(tl.width(), 128);
    assert_eq!(tl.height(), 128);
}

#[test]
#[should_panic]
fn mutable_subgrid_from_buf_with_width_exceeds_stride() {
    let mut buf = [0; 2];
    let _ = MutableSubgrid::<u32>::from_buf(&mut buf, 2, 2, 1);
}

#[test]
#[should_panic]
fn mutable_subgrid_from_buf_with_buffer_smaller_than_required_area() {
    let mut buf = [1, 2];
    let _ = MutableSubgrid::from_buf(&mut buf, 1, 3, 1);
}

#[test]
fn mutable_subgrid_get_row() {
    let mut buf = [1, 2, 3, 4, 5, 6];
    let sub = MutableSubgrid::<u32>::from_buf(&mut buf, 2, 2, 3);
    let row0 = sub.get_row(0);
    assert_eq!(row0, &[1, 2]);
    let row1 = sub.get_row(1);
    assert_eq!(row1, &[4, 5]);
}

#[test]
fn mutable_subgrid_get_row_mut() {
    let mut buf = [1, 2, 3, 4, 5];
    let mut sub = MutableSubgrid::<u32>::from_buf(&mut buf, 2, 2, 3);
    let row = sub.get_row_mut(1);
    row[0] = 7;
    row[1] = 8;
    assert_eq!(buf[3], 7);
    assert_eq!(buf[4], 8);
}

#[test]
fn mutable_subgrid_swap() {
    let mut buf = [1, 2, 3, 4];
    let mut sub = MutableSubgrid::<u32>::from_buf(&mut buf, 2, 2, 2);
    sub.swap((0, 0), (1, 1));
    assert_eq!(buf, [4, 2, 3, 1]);
}

#[test]
fn mutable_subgrid_borrow_mut() {
    let mut buf = [1, 2, 3, 4];
    let mut grid = MutableSubgrid::from_buf(&mut buf, 4, 1, 4);
    let m2 = grid.borrow_mut();
    assert_eq!(m2.width(), 4);
    assert_eq!(m2.height(), 1);
}

#[test]
fn mutable_subgrid_as_shared() {
    let mut buf = [1, 2, 3, 4];
    let grid = MutableSubgrid::from_buf(&mut buf, 4, 1, 4);
    let s2 = grid.as_shared();
    assert_eq!(s2.width(), 4);
    assert_eq!(s2.height(), 1);
}

#[test]
#[should_panic]
fn mutable_subgrid_split_horizontal_with_index_exceeds_width() {
    let mut buf = [0, 1, 2];
    let mut grid = MutableSubgrid::from_buf(&mut buf, 3, 1, 3);
    let _ = grid.split_horizontal(4);
}

#[test]
#[should_panic]
fn mutable_subgrid_split_split_horizontal_in_place_with_index_exceeds_width() {
    let mut buf = [0, 1, 2];
    let mut grid = MutableSubgrid::from_buf(&mut buf, 3, 1, 3);
    let _ = grid.split_horizontal_in_place(4);
}

#[test]
#[should_panic]
fn mutable_subgrid_split_split_vertical_with_index_exceeds_height() {
    let mut buf = [0];
    let mut grid = MutableSubgrid::from_buf(&mut buf, 1, 1, 1);
    let _ = grid.split_vertical(2);
}

#[test]
#[should_panic]
fn mutable_subgrid_split_split_vertical_in_place_with_index_exceeds_height() {
    let mut buf = [0];
    let mut grid = MutableSubgrid::from_buf(&mut buf, 1, 1, 1);
    let _ = grid.split_vertical_in_place(2);
}

#[test]
fn mutable_subgrid_into_groups() {
    let mut buf = [1, 2, 3, 4];
    let grid1 = MutableSubgrid::from_buf(&mut buf, 2, 2, 2);
    let group1 = grid1.into_groups(1, 1);
    assert_eq!(group1.len(), 4);
    for g in group1 {
        assert_eq!(g.width(), 1);
        assert_eq!(g.height(), 1);
    }
    let grid2 = MutableSubgrid::from_buf(&mut buf, 2, 2, 2);
    let group2 = grid2.into_groups(3, 3);
    assert_eq!(group2.len(), 1);
    let g = &group2[0];
    assert_eq!(g.width(), 2);
    assert_eq!(g.height(), 2);
}

#[test]
fn mutable_subgrid_into_groups_with_fix_count() {
    let mut buf = [0, 1, 2, 3, 4, 5, 6, 7, 8];
    let grid = MutableSubgrid::from_buf(&mut buf, 3, 3, 3);
    let groups = grid.into_groups_with_fixed_count(2, 2, 2, 2);
    assert_eq!(groups.len(), 4);
    assert_eq!(groups[0].width(), 2);
    assert_eq!(groups[0].height(), 2);
    assert_eq!(groups[1].width(), 1);
    assert_eq!(groups[1].height(), 2);
    assert_eq!(groups[2].width(), 2);
    assert_eq!(groups[2].height(), 1);
    assert_eq!(groups[3].width(), 1);
    assert_eq!(groups[3].height(), 1);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn mutable_subgrid_as_vectored() {
    if std::arch::is_aarch64_feature_detected!("neon") {
        let len = 8usize;
        let align = std::mem::align_of::<float32x4_t>();

        // Create aligned buffer
        let mut buf = vec![1.0f32; len + align - 1];
        let extra = buf.as_ptr() as usize & (align - 1);
        let offset = (align - extra) / std::mem::size_of::<f32>();
        let data = &mut buf[offset..][..len];

        let mut msg = MutableSubgrid::from_buf(data, 4, 2, 4);
        let opt = msg.as_vectored::<float32x4_t>();
        assert!(opt.is_some());
        let msv = opt.unwrap();
        assert_eq!(msv.width(), 1);
        assert_eq!(msv.height(), msg.height());
    }
}

#[test]
#[should_panic]
fn shared_subgrid_from_buf_with_zero_width() {
    let _ = SharedSubgrid::from_buf(&[1, 2, 3], 0, 1, 1);
}

#[test]
#[should_panic]
fn shared_subgrid_from_buf_with_zero_height() {
    let _ = SharedSubgrid::from_buf(&[1, 2, 3], 1, 0, 1);
}

#[test]
#[should_panic]
fn shared_subgrid_from_buf_with_buffer_smaller_than_required_area() {
    let _ = SharedSubgrid::from_buf(&[1, 2], 1, 3, 1);
}

#[test]
#[should_panic]
fn shared_subgrid_split_horizontal_with_index_exceeds_width() {
    let buf = [1, 2, 3];
    let grid = SharedSubgrid::from_buf(&buf, 3, 1, 3);
    grid.split_horizontal(4);
}

#[test]
#[should_panic]
fn shared_subgrid_split_vertical_with_index_exceeds_height() {
    let buf = [1];
    let grid = SharedSubgrid::from_buf(&buf, 1, 1, 1);
    grid.split_vertical(2);
}

#[test]
fn shared_subgrid_get_row() {
    let buf = [10, 11, 12, 13];
    let grid = SharedSubgrid::from_buf(&buf, 2, 2, 2);
    assert_eq!(grid.get_row(0), &[10, 11]);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn shared_subgrid_as_vectored() {
    if std::arch::is_aarch64_feature_detected!("neon") {
        let len = 8usize;
        let align = std::mem::align_of::<float32x4_t>();

        // Create aligned buffer
        let buf = vec![1.0f32; len + align - 1];
        let extra = buf.as_ptr() as usize & (align - 1);
        let offset = (align - extra) / std::mem::size_of::<f32>();
        let data = &buf[offset..][..len];

        let ssg = SharedSubgrid::from_buf(data, 4, 2, 4);
        let opt = ssg.as_vectored::<float32x4_t>();
        assert!(opt.is_some());
        let ssv = opt.unwrap();
        assert_eq!(ssv.width(), ssg.width() / 4);
        assert_eq!(ssv.height(), ssg.height());
    }
}

mod miri_ub {
    use super::*;

    // These intentionally exercise currently unsound safe APIs

    // `MutableSubgrid::from_buf` accepts `width == 0` with `height > 0` over an empty slice.
    // The grid has no elements, but `get_row(1)` still treats the second row as in-bounds and
    // computes `dangling.add(stride)` before creating a zero-length slice; `get_row_mut` uses the
    // same calculation. Offsetting the dangling pointer from the empty backing slice is UB and is
    // reported by Miri under:
    // cargo +nightly miri test -p jxl-grid --release zero_width_mutable_subgrid_row_offsets_empty_slice_pointer
    #[test]
    fn zero_width_mutable_subgrid_row_offsets_empty_slice_pointer() {
        let mut buf = [];
        let grid = MutableSubgrid::<u8>::from_buf(&mut buf, 0, 2, 1);
        let row = grid.get_row(1);
        std::hint::black_box(row);
    }

    // Splitting a strided grid at its bottom edge creates an empty bottom subgrid.
    // The implementation still computes `base + height * stride`, which is past the allocation
    // when the final row has padding.
    #[test]
    fn mutable_split_vertical_at_bottom_with_padded_stride() {
        let mut buf = [0u8; 3];
        let mut grid = MutableSubgrid::from_buf(&mut buf, 1, 2, 2);
        let split = grid.split_vertical(2);
        std::hint::black_box(split);
    }

    // Same bottom-edge pointer arithmetic issue as above, but through the in-place split API.
    #[test]
    fn mutable_split_vertical_in_place_at_bottom_with_padded_stride() {
        let mut buf = [0u8; 3];
        let mut grid = MutableSubgrid::from_buf(&mut buf, 1, 2, 2);
        let bottom = grid.split_vertical_in_place(2);
        std::hint::black_box(bottom);
    }

    // The shared subgrid split has the same bottom-edge empty-subgrid bug as the mutable split.
    #[test]
    fn shared_split_vertical_at_bottom_with_padded_stride() {
        let buf = [0u8; 3];
        let grid = SharedSubgrid::from_buf(&buf, 1, 2, 2);
        let split = grid.split_vertical(2);
        std::hint::black_box(split);
    }

    // Creating an empty mutable subgrid at `height..height` uses the same invalid bottom-edge
    // pointer calculation when stride includes padding.
    #[test]
    fn mutable_empty_subgrid_at_bottom_with_padded_stride() {
        let mut buf = [0u8; 3];
        let grid = MutableSubgrid::from_buf(&mut buf, 1, 2, 2);
        let empty = grid.subgrid(.., 2..2);
        std::hint::black_box(empty);
    }

    // Same empty bottom subgrid issue through `SharedSubgrid::subgrid`.
    #[test]
    fn shared_empty_subgrid_at_bottom_with_padded_stride() {
        let buf = [0u8; 3];
        let grid = SharedSubgrid::from_buf(&buf, 1, 2, 2);
        let empty = grid.subgrid(.., 2..2);
        std::hint::black_box(empty);
    }

    // Fixed-count grouping can ask for groups beyond the original height. Those out-of-bounds
    // groups are zero-sized, but their base pointer is still computed past the allocation.
    #[test]
    fn fixed_count_groups_can_create_oob_empty_bottom_group() {
        let mut buf = [0u8; 3];
        let grid = MutableSubgrid::from_buf(&mut buf, 1, 2, 2);
        let groups = grid.into_groups_with_fixed_count(1, 1, 1, 3);
        std::hint::black_box(groups);
    }

    // `MutableSubgrid::from_buf` must reject area-size overflow instead of accepting a grid that
    // can later produce invalid pointers.
    #[test]
    #[should_panic]
    fn mutable_from_buf_area_check_overflows() {
        let mut buf = [0u8; 1];
        MutableSubgrid::from_buf(&mut buf, 1, 2, usize::MAX);
    }

    // Same area-size overflow check as above, through `SharedSubgrid::from_buf`.
    #[test]
    #[should_panic]
    fn shared_from_buf_area_check_overflows() {
        let buf = [0u8; 1];
        SharedSubgrid::from_buf(&buf, 1, 2, usize::MAX);
    }

    // `AlignedGrid::with_alloc_tracker` must reject dimension-product overflow instead of
    // accepting a grid that can later produce invalid pointers.
    #[test]
    #[should_panic]
    fn aligned_grid_dimension_product_overflows() {
        let width = usize::MAX / 2 + 1;
        AlignedGrid::<u8>::with_alloc_tracker(width, 2, None).unwrap();
    }

    // `gy * group_height` must not wrap a later group back onto the first row.
    #[test]
    fn fixed_count_group_height_overflow_aliases_mutable_groups() {
        let mut buf = [0u8; 1];
        let grid = MutableSubgrid::from_buf(&mut buf, 1, 1, 1);
        let mut groups = grid.into_groups_with_fixed_count(1, usize::MAX / 2 + 1, 1, 3);
        assert_eq!(groups.len(), 3);
        assert_eq!((groups[0].width(), groups[0].height()), (1, 1));
        assert_eq!((groups[1].width(), groups[1].height()), (1, 0));
        assert_eq!((groups[2].width(), groups[2].height()), (1, 0));

        let (first, rest) = groups.split_at_mut(1);
        let (_second, third) = rest.split_at_mut(1);
        let first = &mut first[0];
        let third = &mut third[0];

        let a = first.get_mut(0, 0);
        assert!(third.try_get_mut(0, 0).is_none());
        *a = 1;
    }

    // `gx * group_width` must not wrap a later group back onto the first column.
    #[test]
    fn fixed_count_group_width_overflow_aliases_mutable_groups() {
        let mut buf = [0u8; 1];
        let grid = MutableSubgrid::from_buf(&mut buf, 1, 1, 1);
        let mut groups = grid.into_groups_with_fixed_count(usize::MAX / 2 + 1, 1, 3, 1);
        assert_eq!(groups.len(), 3);
        assert_eq!((groups[0].width(), groups[0].height()), (1, 1));
        assert_eq!((groups[1].width(), groups[1].height()), (0, 1));
        assert_eq!((groups[2].width(), groups[2].height()), (0, 1));

        let (first, rest) = groups.split_at_mut(1);
        let (_second, third) = rest.split_at_mut(1);
        let first = &mut first[0];
        let third = &mut third[0];

        let a = first.get_mut(0, 0);
        assert!(third.try_get_mut(0, 0).is_none());
        *a = 1;
    }

    // The requested fixed group count must fit in a `Vec` length.
    #[test]
    #[should_panic(expected = "subgrid group count overflows usize")]
    fn fixed_count_group_count_overflow_panics() {
        let mut buf = [];
        let grid = MutableSubgrid::<u8>::from_buf(&mut buf, 0, 0, 0);
        grid.into_groups_with_fixed_count(1, 1, usize::MAX, 2);
    }
}
