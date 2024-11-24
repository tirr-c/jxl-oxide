pub struct BitWriter {
    output: Vec<u8>,
    buf: u64,
    valid_buf_bits: usize,
}

impl BitWriter {
    pub fn new() -> Self {
        Self {
            output: Vec::new(),
            buf: 0,
            valid_buf_bits: 0,
        }
    }

    fn flush_buf(&mut self, next_buf: u64) {
        let out = self.buf;
        self.valid_buf_bits -= 64;
        self.buf = next_buf;
        if !has_ff_byte(out) {
            self.output.extend_from_slice(&out.to_be_bytes());
            return;
        }

        for idx in 0..8 {
            let shift = (7 - idx) * 8;
            self.emit_byte((out >> shift) as u8);
        }
    }

    fn emit_byte(&mut self, b: u8) {
        self.output.push(b);
        if b == 0xff {
            self.output.push(0);
        }
    }

    pub fn write_huffman(&mut self, bits: u64, len: u8) {
        let shifted = bits >> self.valid_buf_bits;
        self.buf |= shifted;
        self.valid_buf_bits += len as usize;

        if let Some(extra) = self.valid_buf_bits.checked_sub(64) {
            self.flush_buf(bits << (len as usize - extra));
        }
    }

    pub fn write_raw(&mut self, bits: u64, len: u8) {
        if len == 0 {
            return;
        }
        self.write_huffman(bits << (64 - len), len);
    }

    pub fn padding_bits(&self) -> usize {
        (8 - self.valid_buf_bits % 8) % 8
    }

    pub fn finalize(mut self) -> Vec<u8> {
        let out = self.buf;
        let valid_bytes = self.valid_buf_bits.div_ceil(8);
        if valid_bytes == 0 {
            return self.output;
        }

        if !has_ff_byte(out) {
            self.output
                .extend_from_slice(&out.to_be_bytes()[..valid_bytes]);
            return self.output;
        }

        for idx in 0..valid_bytes {
            let shift = (7 - idx) * 8;
            self.emit_byte((out >> shift) as u8);
        }

        self.output
    }
}

fn has_ff_byte(val: u64) -> bool {
    ((!val).wrapping_sub(0x_01010101_01010101u64) & val & 0x_80808080_80808080u64) != 0
}
