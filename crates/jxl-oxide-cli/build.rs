fn main() {
    if !std::env::var("CARGO_FEATURE___FFMPEG")
        .map(|v| !v.is_empty())
        .unwrap_or(false)
    {
        return;
    }

    cc::Build::new()
        .file("src/output/ffmpeg_log.c")
        .compile("jxl-oxide-cli-helper");
}
