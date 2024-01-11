# jxl-oxide fuzz target
This directory contains a single fuzz target, `libfuzzer-decode`.

Run [`cargo fuzz`][cargo-fuzz] at the *repository root*:

```
cargo +nightly fuzz run libfuzzer-decode
```

Optionally provide initial corpus to get better fuzzing results.

---

If you find a new crash, please open an issue on the main repository along with a test case that
reproduces the crash.

[cargo-fuzz]: https://github.com/rust-fuzz/cargo-fuzz
