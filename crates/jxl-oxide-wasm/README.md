# jxl-oxide-wasm
[Live demo]

## Building the demo locally
You'll need:
- [`wasm-pack`]
- Node.js (version 20.x is recommended)
- Yarn

First, build jxl-oxide-wasm WebAssembly module.
```shell
cd crates/jxl-oxide-wasm
wasm-pack build --no-pack --out-dir www/wasm --out-name jxl-oxide
```

Then, bundle the demo with Webpack.
```shell
cd www
yarn install --immutable
yarn build
```

Now `www/dist/` contains the demo. Run HTTP file server in that directory. You need to use
`localhost` URL (or use HTTPS) since Service Worker requires secure context.

[Live demo]: https://jxl-oxide.tirr.dev/demo/index.html
[`wasm-pack`]: https://rustwasm.github.io/wasm-pack/
