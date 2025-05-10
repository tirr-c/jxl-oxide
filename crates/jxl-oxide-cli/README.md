# jxl-oxide-cli

This binary crate provides command line interface of jxl-oxide.

## Feature flags
- `rayon` (default): Enable multithreading using `rayon`.
- `mimalloc` (default): Use mimalloc as memory allocator.
- `__devtools` (unstable): Enable devtool subcommands.
- `__ffmpeg` (unstable): Link to FFmpeg and enable video encoding in devtool subcommands.
