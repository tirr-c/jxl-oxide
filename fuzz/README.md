## Using the fuzzer

* Get fuzzing corpus (e.g. from <https://github.com/libjxl/testdata/tree/main/oss-fuzz>)

- Install afl:
```sh
cargo install afl
```
- Build fuzz target:

```sh
cargo afl build --release
```

- Run afl:

```sh
mkdir '/tmp/fuzz'
cargo afl fuzz -i '<corpus_path>' -o './tmp/fuzz' './target/release/jxl-oxide-fuzz-afl'

```

## To reproduce a crash:

```sh
./target/release/jxl-oxide-reproduce '<path_to_image>'
```
