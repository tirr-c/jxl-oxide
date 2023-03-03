use jxl_bitstream::Bitstream;

pub fn read_icc<R: std::io::Read>(bitstream: &mut Bitstream<R>) -> crate::Result<Vec<u8>> {
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

    let enc_size = jxl_bitstream::read_bits!(bitstream, U64)?;
    let mut decoder = jxl_coding::Decoder::parse(bitstream, 41)?;

    let mut encoded_icc = vec![0u8; enc_size as usize];
    let mut b1 = 0u8;
    let mut b2 = 0u8;
    decoder.begin(bitstream).unwrap();
    for (idx, b) in encoded_icc.iter_mut().enumerate() {
        let sym = decoder.read_varint(bitstream, get_icc_ctx(idx, b1, b2))?;
        if sym >= 256 {
            panic!("Decoded symbol out of range");
        }
        *b = sym as u8;

        b2 = b1;
        b1 = *b;
    }

    Ok(encoded_icc)
}
