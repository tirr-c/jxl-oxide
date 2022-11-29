fn main() {
    let file = std::fs::File::open("input.jxl").expect("Failed to open file");
    let mut bitstream = jxl_bitstream::Bitstream::new(file);
    let bitstream = &mut bitstream;
    let headers = jxl_bitstream::read_bits!(bitstream, Bundle(jxl_bitstream::header::Headers)).expect("Failed to read headers");
    dbg!(&headers);

    if headers.metadata.colour_encoding.want_icc {
        let enc_size = jxl_bitstream::read_bits!(bitstream, U64).unwrap();
        dbg!(enc_size);
        dbg!(bitstream.global_pos());
        let decoder = jxl_coding::Decoder::parse(bitstream, 41);
        dbg!(decoder);
        dbg!(bitstream.global_pos());
    }
}
