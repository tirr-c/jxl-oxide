# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.8.1] - 2024-07-30

### Fixed
- `jxl-modular`: Fix incorrect color with complex inverse palette (#312).
- `jxl-color`: Fix color conversion with out-of-gamut inputs (#316).

## [0.8.0] - 2024-03-25

### Added
- `jxl-color`: Add an option to pass sRGB samples to CMS (#267).
- `jxl-oxide-wasm`: Port SIMD routines (#274).

### Changed
- `jxl-color`: Make peak detection non-default (#267).
- Rename `Lz77Mode` variants, make `IncludeMeta` the default (#275).

### Removed
- `jxl-oxide-cli`: Hide `--lz77-mode` (#275).

## [0.7.2] - 2024-03-03

### Added
- `jxl-oxide-cli`: Add decode argument `--lz77-mode` (#272).

### Fixed
- `jxl-modular`: Support "legacy" method of computing LZ77 `dist_multiplier` (#269, #271).

## [0.7.1] - 2024-02-29

### Fixed
- `jxl-modular`: Fix Squeeze bug when image dimension is slightly larger than group boundary (#258).

## [0.7.0] - 2024-02-24

### Added
- `jxl-oxide-wasm` which provides WebAssembly bindings (#223).

### Changed
- `jxl-modular`: Change interface to support 16-bit buffers.
- `jxl-render`: Frame blending is deferred for better performance.

## [0.6.0] - 2024-01-13

### Added
- `jxl-color`: HDR tone mapping and gamut mapping.
- `jxl-color`: Add interface to integrate external CMS into `jxl-color`.
- `jxl-render`: Add interface to request specific color encoding.

### Changed
- `jxl-oxide-cli`: `jxl-dec` and `jxl-info` are merged into `jxl-oxide` binary (#161).

## [0.5.2] - 2023-12-23

### Fixed
- `jxl-render`: Fix aarch64 forward DCT bug (#150).

## [0.5.1] - 2023-11-21

### Fixed
- `jxl-oxide`: Link to crate-level docs in `README.md` (#140).

## [0.5.0] - 2023-11-21 \[YANKED\]

### Added
- Introduce parallelism in various places of decoding process.
- Renderer is now thread safe.
- Add SIMD routines for aarch64.
- Allow decoding of incomplete, partial bitstreams.

### Changed
- Rewrite bitstream API so that data are pushed into decoder.

## [0.4.0] - 2023-09-04

### Added
- `jxl-render`: Introduce cropped decoding API.
- `jxl-oxide-cli`: `--output-format` option (#59).

### Changed
- `jxl-oxide`: Merge `JxlRenderer` into `JxlImage`.

### Removed
- `jxl-frame`: Remove `PassGroup` and `HfCoeff`, instead they are decoded into buffer directly.

### Fixed
- `jxl-oxide-cli`: Report image dimension with orientation applied (#72).

## [0.3.0] - 2023-06-16

### Added
- `jxl-render`: Add option to render spot color channel (#48).
- `jxl-oxide`: Add method to retrieve ICC profile of rendered images (#49).

## [0.2.0] - 2023-05-27

### Removed
- `jxl-image`: Remove public field `signature` from `ImageHeader`.

### Fixed
- `jxl-oxide-cli`: Fix float rounding bug (#23).
- Fix panics in various exotic bitstreams.

## [0.1.0] - 2023-05-17

### Added
- This is the first official release of jxl-oxide, a JPEG XL decoder written in Rust.
- Frequently used features are mostly implemented.

[0.8.1]: https://github.com/tirr-c/jxl-oxide/releases/tag/0.8.1
[0.8.0]: https://github.com/tirr-c/jxl-oxide/releases/tag/0.8.0
[0.7.2]: https://github.com/tirr-c/jxl-oxide/releases/tag/0.7.2
[0.7.1]: https://github.com/tirr-c/jxl-oxide/releases/tag/0.7.1
[0.7.0]: https://github.com/tirr-c/jxl-oxide/releases/tag/0.7.0
[0.6.0]: https://github.com/tirr-c/jxl-oxide/releases/tag/0.6.0
[0.5.2]: https://github.com/tirr-c/jxl-oxide/releases/tag/0.5.2
[0.5.1]: https://github.com/tirr-c/jxl-oxide/releases/tag/0.5.1
[0.5.0]: https://github.com/tirr-c/jxl-oxide/compare/0.4.0...0.5.0
[0.4.0]: https://github.com/tirr-c/jxl-oxide/releases/tag/0.4.0
[0.3.0]: https://github.com/tirr-c/jxl-oxide/releases/tag/0.3.0
[0.2.0]: https://github.com/tirr-c/jxl-oxide/releases/tag/0.2.0
[0.1.0]: https://github.com/tirr-c/jxl-oxide/releases/tag/0.1.0
