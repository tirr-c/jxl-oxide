fn main() {
    let file = std::fs::File::open("input.jxl").expect("Failed to open file");
    let mut bitstream = jxl_bitstream::Bitstream::new(file);
    let bitstream = &mut bitstream;
    let headers = jxl_bitstream::read_bits!(bitstream, Bundle(jxl_bitstream::header::Headers)).expect("Failed to read headers");
    dbg!(headers);
    dbg!(std::mem::size_of::<jxl_bitstream::header::Headers>());
}
