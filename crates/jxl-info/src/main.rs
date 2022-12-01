fn main() {
    let file = std::fs::File::open("input.jxl").expect("Failed to open file");
    let mut bitstream = jxl_bitstream::Bitstream::new(file);
    let bitstream = &mut bitstream;
    let headers = jxl_bitstream::read_bits!(bitstream, Bundle(jxl_bitstream::header::Headers)).expect("Failed to read headers");
    dbg!(&headers);

    if headers.metadata.colour_encoding.want_icc {
        let enc_size = jxl_bitstream::read_bits!(bitstream, U64).unwrap();
        dbg!(enc_size);
        let mut decoder = jxl_coding::Decoder::parse(bitstream, 41)
            .expect("failed to decode ICC entropy coding distribution");
        dbg!(&decoder);

        let mut encoded_icc = vec![0u8; enc_size as usize];
        let mut b1 = 0u8;
        let mut b2 = 0u8;
        for (idx, b) in encoded_icc.iter_mut().enumerate() {
            let sym = decoder.read_varint(bitstream, get_icc_ctx(idx, b1, b2))
                .expect("Failed to read encoded ICC stream");
            if sym >= 256 {
                panic!("Decoded symbol out of range");
            }
            *b = sym as u8;

            b2 = b1;
            b1 = *b;
        }

        std::fs::write("encoded_icc", &encoded_icc).unwrap();
    }

    let test_frame = jxl_bitstream::read_bits!(bitstream, Bundle(jxl_bitstream::header::FrameHeader), &headers).expect("Failed to read frame header");
    dbg!(test_frame);
}

fn get_icc_ctx(idx: usize, b1: u8, b2: u8) -> u32 {
    if idx <= 128 {
        return 0;
    }

    let p1 = match b1 {
        | b'a'..=b'z'
        | b'A'..=b'Z' => 0,
        | b'0'..=b'9'
        | b'.'
        | b',' => 1,
        | 0..=1 => 2 + b1 as u32,
        | 2..=15 => 4,
        | 241..=254 => 5,
        | 255 => 6,
        | _ => 7,
    };
    let p2 = match b2 {
        | b'a'..=b'z'
        | b'A'..=b'Z' => 0,
        | b'0'..=b'9'
        | b'.'
        | b',' => 1,
        | 0..=15 => 2,
        | 241..=255 => 3,
        | _ => 4,
    };

    1 + p1 + 8 * p2
}
