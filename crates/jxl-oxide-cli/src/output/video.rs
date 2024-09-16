use std::ffi::{c_char, c_int, c_void, CStr};

use rusty_ffmpeg::ffi as ffmpeg;

mod context;
mod filter;

pub use context::VideoContext;

#[allow(improper_ctypes_definitions)]
unsafe extern "C" fn jxl_oxide_ffmpeg_log(
    avcl: *mut c_void,
    level: c_int,
    fmt: *const c_char,
    vl: ffmpeg::va_list,
) {
    let mut out = vec![0u8; 65536];
    ffmpeg::vsnprintf(out.as_mut_ptr() as *mut c_char, 65536, fmt, vl);

    let len = out.iter().position(|&v| v == 0).unwrap();
    let line = String::from_utf8_lossy(&out[..len]);
    let line = line.trim_end();

    let log_level = match level {
        ..=16 => 1,
        17..=24 => 2,
        25..=32 => 3,
        33..=40 => 4,
        _ => return,
    };

    let avcl = avcl as *mut *const ffmpeg::AVClass;
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
        ffmpeg::av_log_set_callback(Some(jxl_oxide_ffmpeg_log));
    });
}
