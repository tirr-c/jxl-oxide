const MAT_BRADFORD: [f32; 9] = [
    0.8951, 0.2664, -0.1614, -0.7502, 1.7135, 0.0367, 0.0389, -0.0685, 1.0296,
];

const MAT_BRADFORD_INV: [f32; 9] = [
    0.9869929, -0.1470543, 0.1599627, 0.4323053, 0.5183603, 0.0492912, -0.0085287, 0.0400428,
    0.9684867,
];

#[inline]
pub fn matmul3(a: &[f32; 9], b: &[f32; 9]) -> [f32; 9] {
    let bcol0 = [b[0], b[3], b[6]];
    let bcol1 = [b[1], b[4], b[7]];
    let bcol2 = [b[2], b[5], b[8]];

    [
        a[0] * bcol0[0] + a[1] * bcol0[1] + a[2] * bcol0[2],
        a[0] * bcol1[0] + a[1] * bcol1[1] + a[2] * bcol1[2],
        a[0] * bcol2[0] + a[1] * bcol2[1] + a[2] * bcol2[2],
        a[3] * bcol0[0] + a[4] * bcol0[1] + a[5] * bcol0[2],
        a[3] * bcol1[0] + a[4] * bcol1[1] + a[5] * bcol1[2],
        a[3] * bcol2[0] + a[4] * bcol2[1] + a[5] * bcol2[2],
        a[6] * bcol0[0] + a[7] * bcol0[1] + a[8] * bcol0[2],
        a[6] * bcol1[0] + a[7] * bcol1[1] + a[8] * bcol1[2],
        a[6] * bcol2[0] + a[7] * bcol2[1] + a[8] * bcol2[2],
    ]
}

#[inline]
pub fn matmul3vec(a: &[f32; 9], b: &[f32; 3]) -> [f32; 3] {
    [
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2],
        a[3] * b[0] + a[4] * b[1] + a[5] * b[2],
        a[6] * b[0] + a[7] * b[1] + a[8] * b[2],
    ]
}

#[inline]
fn matinv(mat: &[f32; 9]) -> [f32; 9] {
    let det = mat[0] * (mat[4] * mat[8] - mat[5] * mat[7])
        + mat[1] * (mat[5] * mat[6] - mat[3] * mat[8])
        + mat[2] * (mat[3] * mat[7] - mat[4] * mat[6]);
    [
        (mat[4] * mat[8] - mat[5] * mat[7]) / det,
        (mat[7] * mat[2] - mat[8] * mat[1]) / det,
        (mat[1] * mat[5] - mat[2] * mat[4]) / det,
        (mat[5] * mat[6] - mat[3] * mat[8]) / det,
        (mat[8] * mat[0] - mat[6] * mat[2]) / det,
        (mat[2] * mat[3] - mat[0] * mat[5]) / det,
        (mat[3] * mat[7] - mat[4] * mat[6]) / det,
        (mat[6] * mat[1] - mat[7] * mat[0]) / det,
        (mat[0] * mat[4] - mat[1] * mat[3]) / det,
    ]
}

#[inline]
pub fn illuminant_to_xyz([x, y]: [f32; 2]) -> [f32; 3] {
    [x / y, 1.0, (1.0 - x) / y - 1.0]
}

pub fn adapt_mat(from_illuminant: [f32; 2], to_illuminant: [f32; 2]) -> [f32; 9] {
    let from_w = illuminant_to_xyz(from_illuminant);
    let to_w = illuminant_to_xyz(to_illuminant);

    let from_lms = matmul3vec(&MAT_BRADFORD, &from_w);
    let to_lms = matmul3vec(&MAT_BRADFORD, &to_w);

    let mul = [
        to_lms[0] / from_lms[0],
        to_lms[1] / from_lms[1],
        to_lms[2] / from_lms[2],
    ];
    let multiplied = std::array::from_fn(|idx| MAT_BRADFORD[idx] * mul[idx / 3]);
    matmul3(&MAT_BRADFORD_INV, &multiplied)
}

pub fn primaries_to_xyz_mat(primaries: [[f32; 2]; 3], wp: [f32; 2]) -> [f32; 9] {
    let mut primaries = [
        primaries[0][0],
        primaries[1][0],
        primaries[2][0],
        primaries[0][1],
        primaries[1][1],
        primaries[2][1],
        (1.0 - primaries[0][0] - primaries[0][1]),
        (1.0 - primaries[1][0] - primaries[1][1]),
        (1.0 - primaries[2][0] - primaries[2][1]),
    ];
    let primaries_inv = matinv(&primaries);

    let w_xyz = illuminant_to_xyz(wp);
    let mul = matmul3vec(&primaries_inv, &w_xyz);

    for (idx, p) in primaries.iter_mut().enumerate() {
        *p *= mul[idx % 3];
    }
    primaries
}

pub fn xyz_to_primaries_mat(primaries: [[f32; 2]; 3], wp: [f32; 2]) -> [f32; 9] {
    let primaries = [
        primaries[0][0],
        primaries[1][0],
        primaries[2][0],
        primaries[0][1],
        primaries[1][1],
        primaries[2][1],
        (1.0 - primaries[0][0] - primaries[0][1]),
        (1.0 - primaries[1][0] - primaries[1][1]),
        (1.0 - primaries[2][0] - primaries[2][1]),
    ];
    let mut primaries_inv = matinv(&primaries);

    let w_xyz = illuminant_to_xyz(wp);
    let mul = matmul3vec(&primaries_inv, &w_xyz);

    for (idx, p) in primaries_inv.iter_mut().enumerate() {
        *p /= mul[idx / 3];
    }
    primaries_inv
}
