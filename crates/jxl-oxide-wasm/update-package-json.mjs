import fs from 'node:fs';
import path from 'node:path';

const packagePath = process.argv[2] ?? './pkg';
const packageJsonPath = path.resolve(packagePath, 'package.json');

const packageJsonStr = fs.readFileSync(packageJsonPath, { encoding: 'utf-8' });
const packageJson = JSON.parse(packageJsonStr);

packageJson.exports = {
  './package.json': './package.json',
  '.': {
    types: './jxl_oxide_wasm.d.ts',
    import: './jxl_oxide_wasm.js'
  },
  './simd128': {
    types: './jxl_oxide_wasm.d.ts',
    import: './simd128/jxl_oxide_wasm.js'
  },
  './module.wasm': {
    types: './jxl_oxide_wasm_bg.wasm.d.ts',
    import: './jxl_oxide_wasm_bg.wasm'
  },
  './simd128/module.wasm': {
    types: './jxl_oxide_wasm_bg.wasm.d.ts',
    import: './simd128/jxl_oxide_wasm_bg.wasm'
  },
};
packageJson.files = [
  'jxl_oxide_wasm_bg.wasm',
  'jxl_oxide_wasm.js',
  'jxl_oxide_wasm.d.ts',
  'simd128/jxl_oxide_wasm_bg.wasm',
  'simd128/jxl_oxide_wasm.js',
  'LICENSE-APACHE',
  'LICENSE-MIT',
  'README.md',
];

fs.writeFileSync(
  packageJsonPath,
  JSON.stringify(packageJson, null, 2),
  { encoding: 'utf-8' },
);
