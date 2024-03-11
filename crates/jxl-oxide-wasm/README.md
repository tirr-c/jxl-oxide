# jxl-oxide-wasm
This package provides WebAssembly build of jxl-oxide, and its JavaScript bindings and TypeScript
type definitions.

See [tirr-c/jxl-oxide-wasm-demo] for a demo.

[tirr-c/jxl-oxide-wasm-demo]: https://github.com/tirr-c/jxl-oxide-wasm-demo

```javascript
import init, { JxlImage } from 'jxl-oxide-wasm';
await init();

// Use `JxlImage` after initialization.
const image = new JxlImage();
```

## WebAssembly SIMD support
This package also provides WebAssembly module with SIMD enabled. Import `jxl-oxide-wasm/simd128`
instead.
