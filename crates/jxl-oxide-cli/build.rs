fn main() {
    #[cfg(feature = "__ffmpeg")]
    {
        cc::Build::new()
            .file("src/output/ffmpeg_log.c")
            .compile("jxl-oxide-cli-helper");
    }
}
