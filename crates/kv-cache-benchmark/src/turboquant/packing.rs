/// Bit-packing for 3-bit and 4-bit quantized indices.
///
/// 4-bit: two values per byte (trivial nibble packing)
/// 3-bit: 8 values into 3 bytes (24 bits)

/// Pack quantized indices into a byte buffer.
pub fn pack_indices(indices: &[u8], bits: u8, out: &mut Vec<u8>) {
    match bits {
        4 => pack_4bit(indices, out),
        3 => pack_3bit(indices, out),
        _ => panic!("unsupported bit width: {bits}"),
    }
}

/// Unpack indices from a byte buffer.
pub fn unpack_indices(data: &[u8], count: usize, bits: u8) -> Vec<u8> {
    match bits {
        4 => unpack_4bit(data, count),
        3 => unpack_3bit(data, count),
        _ => panic!("unsupported bit width: {bits}"),
    }
}

/// Size of packed data in bytes (not including the norm).
pub fn packed_size(count: usize, bits: u8) -> usize {
    match bits {
        4 => count.div_ceil(2),
        3 => (count * 3).div_ceil(8),
        _ => panic!("unsupported bit width: {bits}"),
    }
}

fn pack_4bit(indices: &[u8], out: &mut Vec<u8>) {
    for chunk in indices.chunks(2) {
        let lo = chunk[0] & 0x0F;
        let hi = if chunk.len() > 1 { chunk[1] & 0x0F } else { 0 };
        out.push(lo | (hi << 4));
    }
}

fn unpack_4bit(data: &[u8], count: usize) -> Vec<u8> {
    let mut result = Vec::with_capacity(count);
    for (i, &byte) in data.iter().enumerate() {
        let lo = byte & 0x0F;
        let hi = (byte >> 4) & 0x0F;
        result.push(lo);
        if i * 2 + 1 < count {
            result.push(hi);
        }
    }
    result.truncate(count);
    result
}

fn pack_3bit(indices: &[u8], out: &mut Vec<u8>) {
    // Pack 8 3-bit values into 3 bytes (24 bits)
    for chunk in indices.chunks(8) {
        let mut bits: u32 = 0;
        for (j, &idx) in chunk.iter().enumerate() {
            bits |= ((idx as u32) & 0x07) << (j * 3);
        }
        out.push((bits & 0xFF) as u8);
        out.push(((bits >> 8) & 0xFF) as u8);
        out.push(((bits >> 16) & 0xFF) as u8);
    }
}

fn unpack_3bit(data: &[u8], count: usize) -> Vec<u8> {
    let mut result = Vec::with_capacity(count);
    for chunk in data.chunks(3) {
        let mut bits: u32 = 0;
        for (j, &byte) in chunk.iter().enumerate() {
            bits |= (byte as u32) << (j * 8);
        }
        for j in 0..8 {
            if result.len() >= count {
                break;
            }
            result.push(((bits >> (j * 3)) & 0x07) as u8);
        }
    }
    result.truncate(count);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_4bit_roundtrip() {
        let indices: Vec<u8> = (0..256).map(|i| (i % 16) as u8).collect();
        let mut packed = Vec::new();
        pack_indices(&indices, 4, &mut packed);
        let unpacked = unpack_indices(&packed, indices.len(), 4);
        assert_eq!(indices, unpacked);
    }

    #[test]
    fn test_3bit_roundtrip() {
        let indices: Vec<u8> = (0..256).map(|i| (i % 8) as u8).collect();
        let mut packed = Vec::new();
        pack_indices(&indices, 3, &mut packed);
        let unpacked = unpack_indices(&packed, indices.len(), 3);
        assert_eq!(indices, unpacked);
    }

    #[test]
    fn test_4bit_packed_size() {
        assert_eq!(packed_size(256, 4), 128);
        assert_eq!(packed_size(255, 4), 128);
        assert_eq!(packed_size(1, 4), 1);
    }

    #[test]
    fn test_3bit_packed_size() {
        assert_eq!(packed_size(8, 3), 3);
        assert_eq!(packed_size(256, 3), 96);
    }
}
