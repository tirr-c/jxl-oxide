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

If you find a crash, please open an issue on the main repository along with a test case that reproduces the crash.