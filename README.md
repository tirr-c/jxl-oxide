# jxl-oxide
JPEG XL decoder written in pure Rust.

## Supported features
- Single-frame Modular and VarDCT
- XYB
- Chroma subsampled YCbCr
- Restoration filters
- Cropped decoding (sort of)

## TODO
- LF frames
- Rendering extra channels
- Image features
  - Non-separable upsampling
  - Patches
  - Splines
  - Noise
- Handling multiple frames (layers, animation)
- Multithreading
- Color management
