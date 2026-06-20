use alloc::vec::Vec;

/// Map continuous values to {-1, 0, +1} trits.
///
/// - v > scale/2  → +1
/// - v < -scale/2 → -1
/// - otherwise    → 0
pub fn quantise_to_trits(values: &[f64], scale: f64) -> Vec<i8> {
    let half = scale / 2.0;
    values
        .iter()
        .map(|&v| {
            if v > half {
                1
            } else if v < -half {
                -1
            } else {
                0
            }
        })
        .collect()
}

/// Encode 128 trits as a 32-byte I2_S block (Microsoft strided layout).
///
/// Trit code: -1 → 0b00, 0 → 0b01, +1 → 0b10
/// Byte p packs elements at indices {p, p+32, p+64, p+96}, two bits each (low→high).
/// Element at offset `k` (k in 0..4) uses bits (k*2) of byte (k%32 + (k/32... no, byte p = element % 32).
///
/// Mapping: element index `elem_idx` → byte = elem_idx % 32, bit_offset = (elem_idx / 32) * 2
pub fn pack_i2s_block(trits: &[i8; 128]) -> [u8; 32] {
    let mut out = [0u8; 32];
    for p in 0..32usize {
        let mut byte = 0u8;
        for k in 0..4usize {
            let elem = trits[p + k * 32];
            let code = (elem + 1) as u8 & 0x3; // -1→0, 0→1, +1→2
            byte |= code << (k * 2);
        }
        out[p] = byte;
    }
    out
}

/// Decode a 32-byte I2_S block back to 128 trits.
pub fn unpack_i2s_block(bytes: &[u8; 32]) -> [i8; 128] {
    let mut out = [0i8; 128];
    for p in 0..32usize {
        let byte = bytes[p];
        for k in 0..4usize {
            let code = (byte >> (k * 2)) & 0x3;
            out[p + k * 32] = code as i8 - 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantise_positive() {
        let values = [0.8_f64, 0.2, -0.9, 0.0];
        let trits = quantise_to_trits(&values, 0.5);
        assert_eq!(trits, vec![1i8, 0, -1, 0]);
    }

    #[test]
    fn i2s_round_trips_all_plus_one() {
        let block = [1i8; 128];
        let packed = pack_i2s_block(&block);
        let unpacked = unpack_i2s_block(&packed);
        assert_eq!(unpacked, block);
    }

    #[test]
    fn i2s_round_trips_all_minus_one() {
        let block = [-1i8; 128];
        let packed = pack_i2s_block(&block);
        let unpacked = unpack_i2s_block(&packed);
        assert_eq!(unpacked, block);
    }

    #[test]
    fn i2s_round_trips_all_zero() {
        let block = [0i8; 128];
        let packed = pack_i2s_block(&block);
        let unpacked = unpack_i2s_block(&packed);
        assert_eq!(unpacked, block);
    }

    #[test]
    fn i2s_mixed_values_round_trip() {
        let mut block = [0i8; 128];
        for (i, slot) in block.iter_mut().enumerate() {
            *slot = match i % 3 {
                0 => 1,
                1 => -1,
                _ => 0,
            };
        }
        let packed = pack_i2s_block(&block);
        let unpacked = unpack_i2s_block(&packed);
        assert_eq!(unpacked, block);
    }
}
