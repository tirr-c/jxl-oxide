mod consts;
mod generic;

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

#[cfg(
    not(target_arch = "x86_64")
)]
pub use generic::*;

#[cfg(target_arch = "x86_64")]
mod x86_64;
#[cfg(target_arch = "x86_64")]
pub use x86_64::*;
