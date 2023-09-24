/// Trait representing a SIMD vector.
pub trait SimdVector: Copy {
    /// The number of `f32` lanes in a single SIMD vector.
    const SIZE: usize;

    /// Return whether this vector type is supported by current CPU.
    fn available() -> bool;

    /// Initialize a SIMD vector with zeroes.
    fn zero() -> Self;
    /// Initialize a SIMD vector with given floats.
    fn set<const N: usize>(val: [f32; N]) -> Self;
    /// Initialize a SIMD vector filled with given float.
    fn splat_f32(val: f32) -> Self;
    /// Load a SIMD vector from memory.
    ///
    /// The pointer doesn't need to be aligned.
    ///
    /// # Safety
    /// The given pointer must be valid.
    unsafe fn load(ptr: *const f32) -> Self;
    /// Load a SIMD vector from memory with aligned pointer.
    ///
    /// # Safety
    /// The given pointer must be valid and properly aligned.
    unsafe fn load_aligned(ptr: *const f32) -> Self;

    /// Extract a single element from the SIMD vector.
    fn extract_f32<const N: i32>(self) -> f32;
    /// Store the SIMD vector to memory.
    ///
    /// The pointer doesn't need to be aligned.
    ///
    /// # Safety
    /// The given pointer must be valid.
    unsafe fn store(self, ptr: *mut f32);
    /// Store the SIMD vector to memory with aligned pointer.
    ///
    /// # Safety
    /// The given pointer must be valid and properly aligned.
    unsafe fn store_aligned(self, ptr: *mut f32);

    fn add(self, lhs: Self) -> Self;
    fn sub(self, lhs: Self) -> Self;
    fn mul(self, lhs: Self) -> Self;
    fn div(self, lhs: Self) -> Self;
    fn abs(self) -> Self;

    fn muladd(self, mul: Self, add: Self) -> Self;
    fn mulsub(self, mul: Self, sub: Self) -> Self;
}

#[cfg(target_arch = "x86_64")]
impl SimdVector for std::arch::x86_64::__m128 {
    const SIZE: usize = 4;

    #[inline]
    fn available() -> bool {
        // x86_64 always supports 128-bit vector (SSE2).
        true
    }

    #[inline]
    fn zero() -> Self {
        unsafe { std::arch::x86_64::_mm_setzero_ps() }
    }

    #[inline]
    fn set<const N: usize>(val: [f32; N]) -> Self {
        assert_eq!(N, Self::SIZE);
        unsafe { std::arch::x86_64::_mm_set_ps(val[3], val[2], val[1], val[0]) }
    }

    #[inline]
    fn splat_f32(val: f32) -> Self {
        unsafe { std::arch::x86_64::_mm_set1_ps(val) }
    }

    #[inline]
    unsafe fn load(ptr: *const f32) -> Self {
        std::arch::x86_64::_mm_loadu_ps(ptr)
    }

    #[inline]
    unsafe fn load_aligned(ptr: *const f32) -> Self {
        std::arch::x86_64::_mm_load_ps(ptr)
    }

    #[inline]
    fn extract_f32<const N: i32>(self) -> f32 {
        assert!((N as usize) < Self::SIZE);
        let bits = unsafe { std::arch::x86_64::_mm_extract_ps::<N>(self) };
        f32::from_bits(bits as u32)
    }

    #[inline]
    unsafe fn store(self, ptr: *mut f32) {
        std::arch::x86_64::_mm_storeu_ps(ptr, self);
    }

    #[inline]
    unsafe fn store_aligned(self, ptr: *mut f32) {
        std::arch::x86_64::_mm_store_ps(ptr, self);
    }

    #[inline]
    fn add(self, lhs: Self) -> Self {
        unsafe { std::arch::x86_64::_mm_add_ps(self, lhs) }
    }

    #[inline]
    fn sub(self, lhs: Self) -> Self {
        unsafe { std::arch::x86_64::_mm_sub_ps(self, lhs) }
    }

    #[inline]
    fn mul(self, lhs: Self) -> Self {
        unsafe { std::arch::x86_64::_mm_mul_ps(self, lhs) }
    }

    #[inline]
    fn div(self, lhs: Self) -> Self {
        unsafe { std::arch::x86_64::_mm_div_ps(self, lhs) }
    }

    #[inline]
    fn abs(self) -> Self {
        unsafe {
            std::arch::x86_64::_mm_andnot_ps(
                Self::splat_f32(f32::from_bits(0x80000000)),
                self,
            )
        }
    }

    #[inline]
    #[cfg(target_feature = "fma")]
    fn muladd(self, mul: Self, add: Self) -> Self {
        unsafe { std::arch::x86_64::_mm_fmadd_ps(self, mul, add) }
    }

    #[inline]
    #[cfg(target_feature = "fma")]
    fn mulsub(self, mul: Self, sub: Self) -> Self {
        unsafe { std::arch::x86_64::_mm_fmadd_ps(self, mul, sub) }
    }

    #[inline]
    #[cfg(not(target_feature = "fma"))]
    fn muladd(self, mul: Self, add: Self) -> Self {
        self.mul(mul).add(add)
    }

    #[inline]
    #[cfg(not(target_feature = "fma"))]
    fn mulsub(self, mul: Self, sub: Self) -> Self {
        self.mul(mul).sub(sub)
    }
}
