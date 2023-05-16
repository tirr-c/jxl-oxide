# jxl-oxide

jxl-oxide is a JPEG XL decoder written in pure Rust. It's internally organized into a few small
crates. This crate acts as a blanket and provides a simple interface made from those crates to
decode the actual image.

# Decoding an image

Decoding a JPEG XL image starts with constructing `JxlImage`. If you're reading a file, you can use
`JxlImage::open`:

```rust
use jxl_oxide::JxlImage;

let image = JxlImage::open("input.jxl").expect("Failed to read image header");
println!("{:?}", image.image_header()); // Prints the image header
```

Or, if you're reading from a reader that implements `Read`, you can use `JxlImage::from_reader`:

```rust
use jxl_oxide::JxlImage;

let image = JxlImage::from_reader(reader).expect("Failed to read image header");
println!("{:?}", image.image_header()); // Prints the image header
```

`JxlImage` parses the image header and embedded ICC profile (if there's any). Use
`JxlImage::renderer` to start rendering the image. You might need to use `JxlRenderer::rendered_icc`
to do color management correctly.

```rust
use jxl_oxide::{JxlImage, RenderResult};

let mut renderer = image.renderer();
loop {
    let result = renderer.render_next_frame()?;
    match result {
        RenderResult::Done(render) => {
            present_image(render);
        },
        RenderResult::NeedMoreData => {
            wait_for_data();
        },
        RenderResult::NoMoreFrames => break,
    }
}
```
