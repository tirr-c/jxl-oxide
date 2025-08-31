//! Constants used in JPEG XL bitstreams.

/// JPEG XL bare codestream signature, `ff 0a`.
pub const CODESTREAM_SIG: [u8; 2] = [0xff, 0x0a];

/// JPEG XL container format signature, `00 00 00 0c 4a 58 4a 20 0d 0a 87 0a`.
pub const CONTAINER_SIG: [u8; 12] = *b"\x00\x00\x00\x0cJXL \x0d\x0a\x87\x0a";
