#![allow(unused_variables, unused_mut, dead_code)]

#[derive(Debug)]
pub struct NoiseParameters {
    lut: [f32; 8],
}

impl<Ctx> jxl_bitstream::Bundle<Ctx> for NoiseParameters {
    type Error = crate::Error;

    fn parse<R: std::io::Read>(
        bitstream: &mut jxl_bitstream::Bitstream<R>,
        _: Ctx,
    ) -> crate::Result<Self> {
        let mut lut = [0.0f32; 8];
        for slot in &mut lut {
            *slot = bitstream.read_bits(10)? as f32 / (1 << 10) as f32;
        }

        Ok(Self { lut })
    }
}

const N: usize = 8;
struct XorShift128Plus {
    s0: [u64; N],
    s1: [u64; N],
    batch: [u64; N],
    batch_pos: usize,
}

impl XorShift128Plus {
    fn new(seed0: u64, seed1: u64) -> Self {
        let mut s0 = [0u64; N];
        let mut s1 = [0u64; N];
        s0[0] = split_mix_64(seed0 + 0x9E3779B97F4A7C15);
        s1[0] = split_mix_64(seed1 + 0x9E3779B97F4A7C15);
        for i in 1..8 {
            s0[i] = split_mix_64(s0[i - 1]);
            s1[i] = split_mix_64(s1[i - 1]);
        }
        Self {
            s0,
            s1,
            batch: [0u64; N],
            batch_pos: 0,
        }
    }

    fn n(&self) -> usize {
        N
    }

    pub fn fill(&mut self, bits: &mut [u32; N * 2]) {
        self.fill_batch();
        (0..N * 2).for_each(|i| {
            let l = self.batch[self.batch_pos];
            self.batch_pos += 1;
            bits[i] = (l & 0xFF_FF_FF_FF) as u32;
            bits[i] = ((l >> 32) & 0xFF_FF_FF_FF) as u32;
        });
    }

    fn fill_batch(&mut self) {
        (0..N).for_each(|i| {
            let mut s1 = self.s0[i];
            let s0 = self.s1[i];
            self.batch[i] = s1 + s0;
            self.s0[i] = s0;
            s1 ^= s1 << 23;
            self.s1[i] = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
        });
    }
}

fn split_mix_64(z: u64) -> u64 {
    let z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9;
    let z = (z ^ (z >> 27)) * 0x94D049BB133111EB;
    z ^ (z >> 31)
}
