mod consts;
mod impls;

fn reorder<T: Copy>(input: &[T], output: &mut [T]) {
    let n = input.len();
    assert!(n.is_power_of_two());
    assert!(output.len() >= n);
    let bits = n.trailing_zeros();
    let shift_bits = usize::BITS - bits;

    for (idx, i) in input.iter().enumerate() {
        let target = idx.reverse_bits() >> shift_bits;
        output[target] = *i;
    }
}

fn small_reorder<const N: usize, T: Copy>(input: &[T], output: &mut [T]) {
    const fn compute_idx_map<const N: usize>() -> [usize; N] {
        let bits = N.trailing_zeros();
        let shift_bits = usize::BITS - bits;

        let mut out = [0; N];
        let mut idx = 0;
        while idx < N {
            out[idx] = idx.reverse_bits() >> shift_bits;
            idx += 1;
        }
        out
    }

    const IDX_MAP: [&[usize]; 6] = [
        &[0],
        &[0, 1],
        &compute_idx_map::<4>(),
        &compute_idx_map::<8>(),
        &compute_idx_map::<16>(),
        &compute_idx_map::<32>(),
    ];

    assert!(N.is_power_of_two());
    assert!(input.len() >= N);
    assert!(output.len() >= N);
    let bits = N.trailing_zeros() as usize;
    assert!(bits < IDX_MAP.len());

    let map = IDX_MAP[bits];
    for (idx, i) in input.iter().enumerate() {
        output[map[idx]] = *i;
    }
}

pub use impls::dct_2d;
pub use impls::idct_2d;
