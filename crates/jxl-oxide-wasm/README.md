# jxl-oxide-wasm

## Building the demo
You'll need:
- [`wasm-pack`]
- A recent version of Node.js (version 20.x is recommended)
- The latest version of Yarn

First, build jxl-oxide-wasm WebAssembly module.
```shell
cd crates/jxl-oxide-wasm
wasm-pack build --no-pack --out-dir www/wasm --out-name jxl-oxide
```

Then, bundle the demo with Webpack.
```shell
cd www
yarn build
```

Now www/dist/ contains the demo. Run HTTP file server in that directory. You need to use `localhost`
URL (or use HTTPS) since Service Worker requires secure context.

[`wasm-pack`]: https://rustwasm.github.io/wasm-pack/
