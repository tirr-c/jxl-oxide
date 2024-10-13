#![allow(dead_code)]

use std::path::PathBuf;

fn testcases_dir() -> PathBuf {
    let testcases_env = std::env::var_os("JXL_OXIDE_TESTCASES");
    match testcases_env {
        Some(testcases_dir) if !testcases_dir.is_empty() => PathBuf::from(testcases_dir),
        _ => {
            let mut testcases_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            testcases_dir.push("tests/conformance/testcases");
            testcases_dir
        }
    }
}

pub fn conformance_path(name: &str) -> PathBuf {
    let mut path = testcases_dir();
    path.push(name);
    path.push("input.jxl");
    path
}
