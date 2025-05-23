# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.12.2] - 2025-05-13

### Added
- `jxl-oxide`: Add `JxlImage::current_image_region` (#462).
- `jxl-oxide-wasm`: Add `JxlImage#renderingRegion` property (#462).

### Fixed
- `jxl-oxide-wasm`: Fix HDR images not being encoded to 16-bit PNG (#461).

## [0.12.1] - 2025-05-10

### Fixed
- `jxl-render`: Fix blending of partial frames (#457).
- `jxl-frame`: Allow partial `GlobalModular` in `All` group (#458).

## [0.12.0] - 2025-04-27

### Added
- `jxl-oxide`: Add `moxcms` integration for external CMS (#435).
- `jxl-oxide-cli`: Add `--color` option to control whether it formats console output (#452).

### Changed
- Update codebase to Rust 2024 (#446). Now jxl-oxide requires Rust 1.85.0 or newer to compile.
- `ColorManagementSystem` trait has been updated so that color transforms can be cached (#435).
- `jxl-oxide-cli`: Print log to stderr instead of stdout (#452).

### Fixed
- `jxl-oxide-cli`: Fix colors not being printed correctly in Windows conhost (#452).

## [0.11.4] - 2025-03-29

### Fixed
- `jxl-color`: Fix cicp tag in ICC profile synthesis (#434).

## [0.11.3] - 2025-03-08

### Added
- `jxl-oxide-wasm`: Add `JxlImage#width` and `JxlImage#height` (#426).

### Fixed
- `jxl-jbr`: Fix dimension aligning of subsampled JPEG (#427).

## [0.11.2] - 2025-02-19

### Added
- `jxl-threadpool`: Add `rayon_global` (#420).

### Changed
- `jxl-oxide`: Default thread pool now uses global Rayon pool, instead of creating new pool (#420).

## [0.11.1] - 2025-01-25

### Fixed
- `jxl-render`: Fix incorrect upsampling factor when using higher-level LF frame (#412).
- `jxl-render`: Fix edge condition with higher-level LF frame (#413).

## [0.11.0] - 2024-12-28

### Added
- Implement JPEG bitstream reconstruction (#390).
- `jxl-oxide`: Extract (potentially Brotli-compressed) Exif metadata (#389).

### Fixed
- `jxl-render`: Fix panic in alpha blending without alpha channel (#403).

## [0.10.2] - 2024-12-07

### Fixed
- `jxl-modular`: Check MA tree depth limit while decoding (#391).
- `jxl-frame`: Track group buffer allocation (#393).

## [0.10.1] - 2024-11-10

### Changed
- `jxl-oxide`: Enable `image` feature in Docs.rs (#382).

## [0.10.0] - 2024-11-01

### Added
- `jxl-oxide`: Accept `u8` and `u16` output buffers in `ImageStream::write_to_buffer` (#366).
- `jxl-oxide`: Add `image` integration under a feature flag (#368).

### Changed
- `jxl-color`: Use better PQ to HLG method (#348).

### Fixed
- `jxl-render`: Fix requested color encoding not applied in some cases (#369).
- `jxl-oxide`: Fix CMYK to RGB conversion (#370).

## [0.9.1] - 2024-10-12

### Fixed
- `jxl-color`: Fix generated `mluc` tag in ICC profile (#347).
- `jxl-oxide`: Fix panic while decoding fully loaded intermediate frame (#345).

## [0.9.0] - 2024-09-10

### Added
- `jxl-oxide-cli`: Use mimalloc (#287, #288).
- `jxl-oxide-cli`: Add `--num-reps` (#292).

### Changed
- `jxl-grid`: Reorganize modules (#303). Types are renamed.
- `jxl-image`: Move `ImageMetadata::encoded_color_channels` into `jxl_frame::FrameHeader` (#322).

### Removed
- `jxl-oxide`: Remove `Render::image` (#334). Use `Render::stream` instead.

### Fixed
- `jxl-render`: Fix typo in forward DCT (#301).

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

[0.12.2]: https://github.com/tirr-c/jxl-oxide/releases/tag/0.12.2
[0.12.1]: https://github.com/tirr-c/jxl-oxide/releases/tag/0.12.1
[0.12.0]: https://github.com/tirr-c/jxl-oxide/releases/tag/0.12.0
[0.11.4]: https://github.com/tirr-c/jxl-oxide/releases/tag/0.11.4
[0.11.3]: https://github.com/tirr-c/jxl-oxide/releases/tag/0.11.3
[0.11.2]: https://github.com/tirr-c/jxl-oxide/releases/tag/0.11.2
[0.11.1]: https://github.com/tirr-c/jxl-oxide/releases/tag/0.11.1
[0.11.0]: https://github.com/tirr-c/jxl-oxide/releases/tag/0.11.0
[0.10.2]: https://github.com/tirr-c/jxl-oxide/releases/tag/0.10.2
[0.10.1]: https://github.com/tirr-c/jxl-oxide/releases/tag/0.10.1
[0.10.0]: https://github.com/tirr-c/jxl-oxide/releases/tag/0.10.0
[0.9.1]: https://github.com/tirr-c/jxl-oxide/releases/tag/0.9.1
[0.9.0]: https://github.com/tirr-c/jxl-oxide/releases/tag/0.9.0
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
