use std::path::PathBuf;

pub fn debug_enabled() -> bool {
    std::env::var("JXL_OXIDE_DEBUG").is_ok()
}

pub fn conformance_testcases_dir() -> PathBuf {
    let testcases_env = std::env::var_os("JXL_OXIDE_CONFORMANCE_TESTCASES");
    match testcases_env {
        Some(testcases_dir) if !testcases_dir.is_empty() => PathBuf::from(testcases_dir),
        _ => {
            let mut testcases_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            testcases_dir.push("conformance/testcases");
            testcases_dir
        }
    }
}

pub fn conformance_path(name: &str) -> PathBuf {
    let mut path = conformance_testcases_dir();
    path.push(name);
    path.push("input.jxl");
    path
}

pub fn decode_testcases_dir() -> PathBuf {
    let testcases_env = std::env::var_os("JXL_OXIDE_DECODE_TESTCASES");
    match testcases_env {
        Some(testcases_dir) if !testcases_dir.is_empty() => PathBuf::from(testcases_dir),
        _ => {
            let mut testcases_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            testcases_dir.push("decode");
            testcases_dir
        }
    }
}

pub fn cache_dir() -> PathBuf {
    let cache_env = std::env::var_os("JXL_OXIDE_CACHE");
    match cache_env {
        Some(cache_dir) if !cache_dir.is_empty() => PathBuf::from(cache_dir),
        _ => {
            let mut cache_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            cache_dir.push("tests/cache");
            cache_dir
        }
    }
}

pub fn download_object_with_cache(hash: &str, ext: &str) -> Vec<u8> {
    let url = format!(
        "https://storage.googleapis.com/storage/v1/b/jxl-conformance/o/objects%2F{hash}?alt=media"
    );
    let name = format!("{hash}.{ext}");
    download_url_with_cache(&url, &name)
}

pub fn download_url_with_cache(url: &str, name: &str) -> Vec<u8> {
    let cache_dir = cache_dir();
    let path = cache_dir.join(name);

    if let Ok(buf) = std::fs::read(&path) {
        buf
    } else {
        #[cfg(feature = "net")]
        {
            let bytes = reqwest::blocking::get(url)
                .and_then(|resp| resp.error_for_status())
                .and_then(|resp| resp.bytes())
                .expect("Cannot download the given URL");
            std::fs::write(path, &bytes).ok();
            bytes.to_vec()
        }

        #[cfg(not(feature = "net"))]
        {
            panic!(
                "Fixture {name} ({url}) not found in {}, but network is disabled",
                cache_dir.to_string_lossy()
            );
        }
    }
}

pub fn cache_file_path(cache_dir: impl AsRef<std::path::Path>, name: &str) -> PathBuf {
    cache_dir.as_ref().join(name)
}
