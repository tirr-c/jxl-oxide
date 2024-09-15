use std::ffi::{c_char, c_int, c_void, CStr};

use rusty_ffmpeg::ffi as ffmpeg;

mod context;
mod filter;

pub use context::VideoContext;

extern "C" {
    fn jxl_oxide_ffmpeg_log_c(vacl: *mut c_void, level: c_int, fmt: *const c_char, vl: *mut i8);
}

#[no_mangle]
extern "C" fn jxl_oxide_ffmpeg_log(
    avcl: *mut *const ffmpeg::AVClass,
    level: c_int,
    line: *const c_char,
) {
    let log_level = match level {
        ..=16 => 1,
        17..=24 => 2,
        25..=32 => 3,
        33..=40 => 4,
        _ => return,
    };

    let line = unsafe { std::ffi::CStr::from_ptr(line) };
    let line = line.to_string_lossy();
    let line = line.trim_end();

    let avc = if avcl.is_null() {
        std::ptr::null()
    } else {
        unsafe { *avcl }
    };

    let header = if !avc.is_null() {
        let item_name = unsafe {
            let item_name_fn = (*avc).item_name.unwrap_or(ffmpeg::av_default_item_name);
            let name = CStr::from_ptr(item_name_fn(avcl as *mut _));
            name.to_string_lossy()
        };
        format!("[{item_name}] ")
    } else {
        String::new()
    };

    match log_level {
        1 => tracing::error!(target: "ffmpeg", "{header}{line}"),
        2 => tracing::warn!(target: "ffmpeg", "{header}{line}"),
        3 => tracing::info!(target: "ffmpeg", "{header}{line}"),
        _ => tracing::debug!(target: "ffmpeg", "{header}{line}"),
    }
}

fn init_ffmpeg_log() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| unsafe {
        ffmpeg::av_log_set_callback(Some(jxl_oxide_ffmpeg_log_c));
    });
}
