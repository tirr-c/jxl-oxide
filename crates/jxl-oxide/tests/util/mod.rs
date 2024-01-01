#![allow(dead_code)]

pub fn conformance_path(name: &str) -> std::path::PathBuf {
    let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/conformance/testcases");
    path.push(name);
    path.push("input.jxl");
    path
}
