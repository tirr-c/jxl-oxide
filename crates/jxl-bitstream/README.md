# jxl-bitstream
This crate provides a JPEG XL bitstream reader. The bitstream reader supports both bare codestream
and container format, and it can detect which format to read.

Consumers of this crate can use `Bitstream::new_detect` to create a bitstream reader.
