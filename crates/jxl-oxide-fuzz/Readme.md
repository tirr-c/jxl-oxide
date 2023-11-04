# Fuzzing harness for jxl-oxide
This crate houses the fuzzing harness for jxl-oxide. 

## Setup
First install honggfuzz:
```shell
cargo install honggfuzz
```

## Running
```shell
cargo hfuzz run fuzz_decode
```

If you find a new crash, please open an issue on the main repository along with a test case that reproduces the crash.
You can move the .fuzz file to crates/jxl-oxide/tests/fuzz_findings/ and add the name to the test_fuzz_findings.rs file.