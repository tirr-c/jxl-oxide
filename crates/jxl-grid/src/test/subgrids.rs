use crate::AlignedGrid;

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
