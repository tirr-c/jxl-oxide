#![no_main]

const DIM_LIMIT: u32 = 65536;

// 128 MiB
const SIZE_LIMIT: usize = 128 * 1024 * 1024;

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    jxl_oxide_fuzz::fuzz_decode(data, DIM_LIMIT, SIZE_LIMIT);
});
