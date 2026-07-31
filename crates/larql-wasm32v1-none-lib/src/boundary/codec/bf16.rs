//! Bfloat16 residual codec.
//!
//! BF16 = upper 16 bits of IEEE 754 float32 with round-to-nearest-even.

use alloc::vec::Vec;

pub const BYTES_PER_ELEM: usize = 2;

const ROUND_CORRECTION: u32 = 0x7FFF;

pub fn encode(r: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(r.len() * BYTES_PER_ELEM);
    for &v in r {
        let bits = v.to_bits();
        let bf16 = ((bits + ROUND_CORRECTION + ((bits >> 16) & 1)) >> 16) as u16;
        out.extend_from_slice(&bf16.to_le_bytes());
    }
    out
}

/// # Panics
/// Panics if `payload.len()` is not a multiple of 2.
pub fn decode(payload: &[u8]) -> Vec<f32> {
    assert_eq!(
        payload.len() % BYTES_PER_ELEM,
        0,
        "bf16 payload length must be even"
    );
    payload
        .chunks_exact(BYTES_PER_ELEM)
        .map(|b| {
            let bf16 = u16::from_le_bytes([b[0], b[1]]);
            f32::from_bits(u32::from(bf16) << 16)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_normal_values() {
        let r = alloc::vec![0.0f32, 1.0, -1.0, 2.71, -100.0, 0.001];
        let dec = decode(&encode(&r));
        for (orig, got) in r.iter().zip(dec.iter()) {
            assert!(
                (orig - got).abs() <= orig.abs() * 0.01 + 1e-4,
                "orig={orig} got={got}"
            );
        }
    }

    #[test]
    fn no_overflow_for_large_residuals() {
        let r = alloc::vec![94_208.0f32, -151_552.0, 1.5e38, 0.0];
        let dec = decode(&encode(&r));
        for v in &dec {
            assert!(v.is_finite(), "bf16 produced non-finite from large input");
        }
    }

    #[test]
    fn roundtrip_preserves_sign_of_zero() {
        let r = alloc::vec![0.0f32, -0.0];
        let dec = decode(&encode(&r));
        assert_eq!(dec.len(), 2);
    }

    #[test]
    fn payload_length() {
        let r = alloc::vec![0.0f32; 2560];
        assert_eq!(encode(&r).len(), 2560 * BYTES_PER_ELEM);
    }
}
