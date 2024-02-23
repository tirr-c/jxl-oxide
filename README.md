# jxl-oxide
[![crates.io](https://img.shields.io/crates/v/jxl-oxide.svg)](https://crates.io/crates/jxl-oxide)
[![docs.rs](https://docs.rs/jxl-oxide/badge.svg)](https://docs.rs/crate/jxl-oxide/)
[![Build Status](https://img.shields.io/github/actions/workflow/status/tirr-c/jxl-oxide/build.yml?branch=main)](https://github.com/tirr-c/jxl-oxide/actions/workflows/build.yml?query=branch%3Amain)

A spec-conforming JPEG XL decoder written in pure Rust[^1].

If you want to use jxl-oxide in a terminal, install `jxl-oxide-cli` using `cargo install`. It will
install a binary named `jxl-oxide`.

```
cargo install jxl-oxide-cli
```

If you want to use it as a library, add `jxl-oxide` in `Cargo.toml`. `jxl-oxide` is a blanket crate
which covers various components of jxl-oxide.

```toml
[dependencies]
jxl-oxide = "0.6.0"
```

---

Dual-licensed under MIT and Apache 2.0.

[^1]: Integration with Little CMS 2, which is written in C, can be enabled with `lcms2` feature.

[conformance]: https://github.com/libjxl/conformance
