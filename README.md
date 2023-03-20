# jxl-oxide
JPEG XL decoder written in pure Rust.

## Supported features
- Simple Modular and VarDCT images
- Progressive LF (LF frames)
- XYB
- HDR images
- Chroma subsampled YCbCr
- Restoration filters
- Image features
  - Patches
- Frame composition
- Cropped decoding (sort of)

## TODO
- Rendering spot color channels
- Image features
  - Non-separable upsampling
  - Splines
  - Noise
- Animation
- Multithreading
