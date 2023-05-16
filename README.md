# jxl-oxide
JPEG XL decoder written in pure Rust.

jxl-oxide consists of small library crates (`jxl-bitstream`, `jxl-coding`, ...), a blanket library
crate `jxl-oxide`, and a binary crate `jxl-oxide-cli`. If you want to use jxl-oxide in a terminal,
install it using `cargo install`. Cargo will install two binaries, `jxl-dec` and `jxl-info`.

```
cargo install jxl-oxide-cli
```

If you want to use it as a library, specify it in `Cargo.toml`:

```toml
[dependencies]
jxl-oxide = "0.1.0"
```

Note that you'll need a color management system to correctly display some JXL images. (`jxl-dec`
uses `lcms2` for the color management.)

---

Dual-licensed under MIT and Apache 2.0.
