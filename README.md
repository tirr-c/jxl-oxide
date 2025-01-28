# jxl-oxide
[![crates.io](https://img.shields.io/crates/v/jxl-oxide.svg)](https://crates.io/crates/jxl-oxide)
[![docs.rs](https://docs.rs/jxl-oxide/badge.svg)](https://docs.rs/crate/jxl-oxide/)
[![Build Status](https://img.shields.io/github/actions/workflow/status/tirr-c/jxl-oxide/build.yml?branch=main)](https://github.com/tirr-c/jxl-oxide/actions/workflows/build.yml?query=branch%3Amain)

A spec-conforming JPEG XL decoder written in pure Rust[^1].

If you want to use it as a library, add `jxl-oxide` in `Cargo.toml`. `jxl-oxide` is a blanket crate
which covers various components of jxl-oxide.

```toml
[dependencies]
jxl-oxide = "0.11.1"
```

## Installing command line tool

Install `jxl-oxide-cli` using `cargo install`. It will install a binary named `jxl-oxide`.

```
cargo install jxl-oxide-cli
```

## Feature flags

`jxl-oxide` and `jxl-oxide-cli` have different sets of feature flags.

**For `jxl-oxide`:**
- `rayon` (default): Enable multithreading using `rayon`.
- `lcms2`: Integrate into Little CMS 2 which supports arbitrary ICC profiles and enables CMYK to RGB
  conversion. (Note that this will add dependencies written in C.)
- `image`: Integrate into the `image` crate. `jxl_oxide::integration::JxlDecoder` will be made
  available.

**For `jxl-oxide-cli`:**
- `rayon` (default): Enable multithreading using `rayon`.
- `mimalloc` (default): Use mimalloc as memory allocator.
- `__devtools` (unstable): Enable devtool subcommands.
- `__ffmpeg` (unstable): Link to FFmpeg and enable video encoding in devtool subcommands.

---

Dual-licensed under MIT and Apache 2.0.

[^1]: Integration with Little CMS 2, which is written in C, can be enabled with `lcms2` feature.

[conformance]: https://github.com/libjxl/conformance
