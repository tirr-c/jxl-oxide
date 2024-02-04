pub const fn epf_kernel_offsets<const STEP: usize>() -> &'static [(isize, isize)] {
    const EPF_KERNEL_1: [(isize, isize); 4] = [(0, -1), (0, 1), (-1, 0), (1, 0)];
    #[rustfmt::skip]
    const EPF_KERNEL_2: [(isize, isize); 12] = [
        (0, -2), (-1, -1), (0, -1), (1, -1),
        (-2, 0), (-1, 0), (1, 0), (2, 0),
        (-1, 1), (0, 1), (1, 1), (0, 2),
    ];

    if STEP == 0 {
        &EPF_KERNEL_2
    } else if STEP == 1 || STEP == 2 {
        &EPF_KERNEL_1
    } else {
        panic!()
    }
}

pub const fn epf_dist_offsets<const STEP: usize>() -> &'static [(isize, isize)] {
    if STEP == 0 {
        &[(0, -1), (1, 0), (0, 0), (-1, 0), (0, 1)]
    } else if STEP == 1 {
        &[(0, -1), (0, 0), (0, 1), (-1, 0), (1, 0)]
    } else if STEP == 2 {
        &[(0, 0)]
    } else {
        panic!()
    }
}
