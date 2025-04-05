#![allow(unsafe_op_in_unsafe_fn)]

/// Trait representing a SIMD vector.
pub trait SimdVector: Copy {
    /// The number of `f32` lanes in a single SIMD vector.
    const SIZE: usize;

    /// Return whether this vector type is supported by current CPU.
    fn available() -> bool;

    /// Initialize a SIMD vector with zeroes.
    ///
    /// # Safety
    /// CPU should support the vector type.
    unsafe fn zero() -> Self;
    /// Initialize a SIMD vector with given floats.
    ///
    /// # Safety
    /// CPU should support the vector type.
    unsafe fn set<const N: usize>(val: [f32; N]) -> Self;
    /// Initialize a SIMD vector filled with given float.
    ///
    /// # Safety
    /// CPU should support the vector type.
    unsafe fn splat_f32(val: f32) -> Self;
    /// Load a SIMD vector from memory.
    ///
    /// The pointer doesn't need to be aligned.
    ///
    /// # Safety
    /// CPU should support the vector type, and the given pointer must be valid.
    unsafe fn load(ptr: *const f32) -> Self;
    /// Load a SIMD vector from memory with aligned pointer.
    ///
    /// # Safety
    /// CPU should support the vector type, and the given pointer must be valid and properly
    /// aligned.
    unsafe fn load_aligned(ptr: *const f32) -> Self;

    /// Extract a single element from the SIMD vector.
    ///
    /// # Safety
    /// CPU should support the vector type.
    unsafe fn extract_f32<const N: i32>(self) -> f32;
    /// Store the SIMD vector to memory.
    ///
    /// The pointer doesn't need to be aligned.
    ///
    /// # Safety
    /// CPU should support the vector type, and the given pointer must be valid.
    unsafe fn store(self, ptr: *mut f32);
    /// Store the SIMD vector to memory with aligned pointer.
    ///
    /// # Safety
    /// CPU should support the vector type, and the given pointer must be valid and properly
    /// aligned.
    unsafe fn store_aligned(self, ptr: *mut f32);

    /// Add two vectors element-wise.
    ///
    /// # Safety
    /// CPU should support the vector type.
    unsafe fn add(self, lhs: Self) -> Self;
    /// Subtract two vectors element-wise.
    ///
    /// # Safety
    /// CPU should support the vector type.
    unsafe fn sub(self, lhs: Self) -> Self;
    /// Multiply two vectors element-wise.
    ///
    /// # Safety
    /// CPU should support the vector type.
    unsafe fn mul(self, lhs: Self) -> Self;
    /// Divide two vectors element-wise.
    ///
    /// # Safety
    /// CPU should support the vector type.
    unsafe fn div(self, lhs: Self) -> Self;
    /// Compute the absolute value for each element of the vector.
    ///
    /// # Safety
    /// CPU should support the vector type.
    unsafe fn abs(self) -> Self;

    /// Computes `self * mul + add` element-wise.
    ///
    /// # Safety
    /// CPU should support the vector type.
    unsafe fn muladd(self, mul: Self, add: Self) -> Self;
    /// Computes `self * mul - add` element-wise.
    ///
    /// # Safety
    /// CPU should support the vector type.
    unsafe fn mulsub(self, mul: Self, sub: Self) -> Self;
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
    unsafe fn zero() -> Self {
        std::arch::x86_64::_mm_setzero_ps()
    }

    #[inline]
    unsafe fn set<const N: usize>(val: [f32; N]) -> Self {
        assert_eq!(N, Self::SIZE);
        std::arch::x86_64::_mm_set_ps(val[3], val[2], val[1], val[0])
    }

    #[inline]
    unsafe fn splat_f32(val: f32) -> Self {
        std::arch::x86_64::_mm_set1_ps(val)
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
    unsafe fn extract_f32<const N: i32>(self) -> f32 {
        assert!((N as usize) < Self::SIZE);
        let bits = std::arch::x86_64::_mm_extract_ps::<N>(self);
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
    unsafe fn add(self, lhs: Self) -> Self {
        std::arch::x86_64::_mm_add_ps(self, lhs)
    }

    #[inline]
    unsafe fn sub(self, lhs: Self) -> Self {
        std::arch::x86_64::_mm_sub_ps(self, lhs)
    }

    #[inline]
    unsafe fn mul(self, lhs: Self) -> Self {
        std::arch::x86_64::_mm_mul_ps(self, lhs)
    }

    #[inline]
    unsafe fn div(self, lhs: Self) -> Self {
        std::arch::x86_64::_mm_div_ps(self, lhs)
    }

    #[inline]
    unsafe fn abs(self) -> Self {
        let x = std::arch::x86_64::_mm_undefined_si128();
        let mask = std::arch::x86_64::_mm_srli_epi32::<1>(std::arch::x86_64::_mm_cmpeq_epi32(x, x));
        std::arch::x86_64::_mm_and_ps(std::arch::x86_64::_mm_castsi128_ps(mask), self)
    }

    #[inline]
    #[cfg(target_feature = "fma")]
    unsafe fn muladd(self, mul: Self, add: Self) -> Self {
        std::arch::x86_64::_mm_fmadd_ps(self, mul, add)
    }

    #[inline]
    #[cfg(target_feature = "fma")]
    unsafe fn mulsub(self, mul: Self, sub: Self) -> Self {
        std::arch::x86_64::_mm_fmsub_ps(self, mul, sub)
    }

    #[inline]
    #[cfg(not(target_feature = "fma"))]
    unsafe fn muladd(self, mul: Self, add: Self) -> Self {
        self.mul(mul).add(add)
    }

    #[inline]
    #[cfg(not(target_feature = "fma"))]
    unsafe fn mulsub(self, mul: Self, sub: Self) -> Self {
        self.mul(mul).sub(sub)
    }
}

#[cfg(target_arch = "x86_64")]
impl SimdVector for std::arch::x86_64::__m256 {
    const SIZE: usize = 8;

    #[inline]
    fn available() -> bool {
        is_x86_feature_detected!("avx2")
    }

    #[inline]
    unsafe fn zero() -> Self {
        std::arch::x86_64::_mm256_setzero_ps()
    }

    #[inline]
    unsafe fn set<const N: usize>(val: [f32; N]) -> Self {
        assert_eq!(N, Self::SIZE);
        std::arch::x86_64::_mm256_set_ps(
            val[7], val[6], val[5], val[4], val[3], val[2], val[1], val[0],
        )
    }

    #[inline]
    unsafe fn splat_f32(val: f32) -> Self {
        std::arch::x86_64::_mm256_set1_ps(val)
    }

    #[inline]
    unsafe fn load(ptr: *const f32) -> Self {
        std::arch::x86_64::_mm256_loadu_ps(ptr)
    }

    #[inline]
    unsafe fn load_aligned(ptr: *const f32) -> Self {
        std::arch::x86_64::_mm256_load_ps(ptr)
    }

    #[inline]
    unsafe fn extract_f32<const N: i32>(self) -> f32 {
        unsafe fn inner<const HI: i32, const LO: i32>(val: std::arch::x86_64::__m256) -> f32 {
            std::arch::x86_64::_mm256_extractf128_ps::<HI>(val).extract_f32::<LO>()
        }

        assert!((N as usize) < Self::SIZE);
        match N {
            0..=3 => inner::<0, N>(self),
            4 => inner::<1, 0>(self),
            5 => inner::<1, 1>(self),
            6 => inner::<1, 2>(self),
            7 => inner::<1, 3>(self),
            // SAFETY: 0 <= N < 8 by assertion.
            _ => std::hint::unreachable_unchecked(),
        }
    }

    #[inline]
    unsafe fn store(self, ptr: *mut f32) {
        std::arch::x86_64::_mm256_storeu_ps(ptr, self);
    }

    #[inline]
    unsafe fn store_aligned(self, ptr: *mut f32) {
        std::arch::x86_64::_mm256_store_ps(ptr, self);
    }

    #[inline]
    unsafe fn add(self, lhs: Self) -> Self {
        std::arch::x86_64::_mm256_add_ps(self, lhs)
    }

    #[inline]
    unsafe fn sub(self, lhs: Self) -> Self {
        std::arch::x86_64::_mm256_sub_ps(self, lhs)
    }

    #[inline]
    unsafe fn mul(self, lhs: Self) -> Self {
        std::arch::x86_64::_mm256_mul_ps(self, lhs)
    }

    #[inline]
    unsafe fn div(self, lhs: Self) -> Self {
        std::arch::x86_64::_mm256_div_ps(self, lhs)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn abs(self) -> Self {
        let x = std::arch::x86_64::_mm256_undefined_si256();
        let mask =
            std::arch::x86_64::_mm256_srli_epi32::<1>(std::arch::x86_64::_mm256_cmpeq_epi32(x, x));
        std::arch::x86_64::_mm256_and_ps(std::arch::x86_64::_mm256_castsi256_ps(mask), self)
    }

    #[inline]
    #[cfg(target_feature = "fma")]
    unsafe fn muladd(self, mul: Self, add: Self) -> Self {
        std::arch::x86_64::_mm256_fmadd_ps(self, mul, add)
    }

    #[inline]
    #[cfg(target_feature = "fma")]
    unsafe fn mulsub(self, mul: Self, sub: Self) -> Self {
        std::arch::x86_64::_mm256_fmsub_ps(self, mul, sub)
    }

    #[inline]
    #[cfg(not(target_feature = "fma"))]
    unsafe fn muladd(self, mul: Self, add: Self) -> Self {
        self.mul(mul).add(add)
    }

    #[inline]
    #[cfg(not(target_feature = "fma"))]
    unsafe fn mulsub(self, mul: Self, sub: Self) -> Self {
        self.mul(mul).sub(sub)
    }
}

#[cfg(target_arch = "aarch64")]
impl SimdVector for std::arch::aarch64::float32x4_t {
    const SIZE: usize = 4;

    #[inline]
    fn available() -> bool {
        std::arch::is_aarch64_feature_detected!("neon")
    }

    #[inline]
    unsafe fn zero() -> Self {
        std::arch::aarch64::vdupq_n_f32(0f32)
    }

    #[inline]
    unsafe fn set<const N: usize>(val: [f32; N]) -> Self {
        assert_eq!(N, Self::SIZE);
        std::arch::aarch64::vld1q_f32(val.as_ptr())
    }

    #[inline]
    unsafe fn splat_f32(val: f32) -> Self {
        std::arch::aarch64::vdupq_n_f32(val)
    }

    #[inline]
    unsafe fn load(ptr: *const f32) -> Self {
        std::arch::aarch64::vld1q_f32(ptr)
    }

    #[inline]
    unsafe fn load_aligned(ptr: *const f32) -> Self {
        std::arch::aarch64::vld1q_f32(ptr)
    }

    #[inline]
    unsafe fn extract_f32<const N: i32>(self) -> f32 {
        std::arch::aarch64::vgetq_lane_f32::<N>(self)
    }

    #[inline]
    unsafe fn store(self, ptr: *mut f32) {
        std::arch::aarch64::vst1q_f32(ptr, self)
    }

    #[inline]
    unsafe fn store_aligned(self, ptr: *mut f32) {
        std::arch::aarch64::vst1q_f32(ptr, self)
    }

    #[inline]
    unsafe fn add(self, lhs: Self) -> Self {
        std::arch::aarch64::vaddq_f32(self, lhs)
    }

    #[inline]
    unsafe fn sub(self, lhs: Self) -> Self {
        std::arch::aarch64::vsubq_f32(self, lhs)
    }

    #[inline]
    unsafe fn mul(self, lhs: Self) -> Self {
        std::arch::aarch64::vmulq_f32(self, lhs)
    }

    #[inline]
    unsafe fn div(self, lhs: Self) -> Self {
        std::arch::aarch64::vdivq_f32(self, lhs)
    }

    #[inline]
    unsafe fn abs(self) -> Self {
        std::arch::aarch64::vabsq_f32(self)
    }

    #[inline]
    unsafe fn muladd(self, mul: Self, add: Self) -> Self {
        std::arch::aarch64::vfmaq_f32(add, self, mul)
    }

    #[inline]
    unsafe fn mulsub(self, mul: Self, sub: Self) -> Self {
        std::arch::aarch64::vfmsq_f32(sub, self, mul)
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
impl SimdVector for std::arch::wasm32::v128 {
    const SIZE: usize = 4;

    #[inline]
    fn available() -> bool {
        true
    }

    #[inline]
    unsafe fn zero() -> Self {
        std::arch::wasm32::f32x4_splat(0f32)
    }

    #[inline]
    unsafe fn set<const N: usize>(val: [f32; N]) -> Self {
        assert_eq!(N, Self::SIZE);
        std::arch::wasm32::f32x4(val[0], val[1], val[2], val[3])
    }

    #[inline]
    unsafe fn splat_f32(val: f32) -> Self {
        std::arch::wasm32::f32x4_splat(val)
    }

    #[inline]
    unsafe fn load(ptr: *const f32) -> Self {
        std::arch::wasm32::v128_load(ptr as *const _)
    }

    #[inline]
    unsafe fn load_aligned(ptr: *const f32) -> Self {
        std::arch::wasm32::v128_load(ptr as *const _)
    }

    #[inline]
    unsafe fn extract_f32<const N: i32>(self) -> f32 {
        match N {
            0 => std::arch::wasm32::f32x4_extract_lane::<0>(self),
            1 => std::arch::wasm32::f32x4_extract_lane::<1>(self),
            2 => std::arch::wasm32::f32x4_extract_lane::<2>(self),
            3 => std::arch::wasm32::f32x4_extract_lane::<3>(self),
            _ => panic!(),
        }
    }

    #[inline]
    unsafe fn store(self, ptr: *mut f32) {
        std::arch::wasm32::v128_store(ptr as *mut _, self)
    }

    #[inline]
    unsafe fn store_aligned(self, ptr: *mut f32) {
        std::arch::wasm32::v128_store(ptr as *mut _, self)
    }

    #[inline]
    unsafe fn add(self, lhs: Self) -> Self {
        std::arch::wasm32::f32x4_add(self, lhs)
    }

    #[inline]
    unsafe fn sub(self, lhs: Self) -> Self {
        std::arch::wasm32::f32x4_sub(self, lhs)
    }

    #[inline]
    unsafe fn mul(self, lhs: Self) -> Self {
        std::arch::wasm32::f32x4_mul(self, lhs)
    }

    #[inline]
    unsafe fn div(self, lhs: Self) -> Self {
        std::arch::wasm32::f32x4_div(self, lhs)
    }

    #[inline]
    unsafe fn abs(self) -> Self {
        std::arch::wasm32::f32x4_abs(self)
    }

    #[inline]
    unsafe fn muladd(self, mul: Self, add: Self) -> Self {
        self.mul(mul).add(add)
    }

    #[inline]
    unsafe fn mulsub(self, mul: Self, sub: Self) -> Self {
        self.mul(mul).sub(sub)
    }
}
