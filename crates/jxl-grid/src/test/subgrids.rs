#[cfg(all(target_arch = "aarch64"))]
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

#[cfg(all(target_arch = "aarch64"))]
#[test]
fn mutable_subgrid_as_vectored() {
    if std::arch::is_aarch64_feature_detected!("neon") {
        let mut data = vec![1.0; 8];
        let mut msg = MutableSubgrid::from_buf(&mut data[..], 4, 2, 4);
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

#[cfg(all(target_arch = "aarch64"))]
#[test]
fn shared_subgrid_as_vectored() {
    if std::arch::is_aarch64_feature_detected!("neon") {
        let buf: Vec<f32> = vec![1.0; 8];
        let ssg = SharedSubgrid::from_buf(&buf, 4, 2, 4);
        let opt = ssg.as_vectored::<float32x4_t>();
        assert!(opt.is_some());
        let ssv = opt.unwrap();
        assert_eq!(ssv.width(), ssg.width() / 4);
        assert_eq!(ssv.height(), ssg.height());
    }
}
