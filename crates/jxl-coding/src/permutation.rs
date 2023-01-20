use jxl_bitstream::Bitstream;

pub fn read_permutation<R: std::io::Read>(bitstream: &mut Bitstream<R>, decoder: &mut crate::Decoder, size: u32, skip: u32) -> crate::Result<Vec<usize>> {
    let end = decoder.read_varint(bitstream, get_context(size))?;

    let mut lehmer = vec![0u32; size as usize];
    let mut prev_val = 0u32;
    for val in &mut lehmer[skip as usize..][..end as usize] {
        *val = decoder.read_varint(bitstream, get_context(prev_val))?;
        prev_val = *val;
    }

    let mut temp = (0..(size as usize)).collect::<Vec<_>>();
    let mut permutation = Vec::with_capacity(size as usize);
    for idx in lehmer {
        let idx = idx as usize;
        if idx >= temp.len() {
            return Err(crate::Error::InvalidPermutation);
        }
        permutation.push(temp.remove(idx));
    }

    Ok(permutation)
}

fn get_context(x: u32) -> u32 {
    crate::add_log2_ceil(x).min(7)
}
