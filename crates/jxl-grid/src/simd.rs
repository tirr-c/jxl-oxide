pub trait SimdLane: Copy {
    const SIZE: usize;

    fn zero() -> Self;
    fn set<const N: usize>(val: [f32; N]) -> Self;
    fn splat_f32(val: f32) -> Self;
    unsafe fn load(ptr: *const f32) -> Self;
    unsafe fn load_aligned(ptr: *const f32) -> Self;

    fn extract_f32<const N: i32>(self) -> f32;
    unsafe fn store(self, ptr: *mut f32);
    unsafe fn store_aligned(self, ptr: *mut f32);

    fn add(self, lhs: Self) -> Self;
    fn sub(self, lhs: Self) -> Self;
    fn mul(self, lhs: Self) -> Self;
    fn div(self, lhs: Self) -> Self;
}

#[cfg(target_arch = "x86_64")]
impl SimdLane for std::arch::x86_64::__m128 {
    const SIZE: usize = 4;

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
}
