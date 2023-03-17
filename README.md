# jxl-oxide
JPEG XL decoder written in pure Rust.

## Supported features
- Simple Modular and VarDCT images
- Progressive LF (LF frames)
- XYB
- Chroma subsampled YCbCr
- Restoration filters
- Cropped decoding (sort of)

## TODO
- Rendering extra channels
- Image features
  - Non-separable upsampling
  - Patches
  - Splines
  - Noise
- Handling multiple frames (layers, animation)
- Multithreading
- Color management
