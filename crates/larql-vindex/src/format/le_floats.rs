//! Little-endian f32 ⇄ byte codecs shared by the on-disk formats
//! (`config::dtype`, `patch::format`, `quant::convert`).
//!
//! Source byte buffers are frequently **unaligned** — mmap slices at
//! arbitrary byte offsets, base64 payloads — so decoding must never
//! reinterpret-cast to `&[f32]` (that's UB on a misaligned pointer).
//! Explicit `from_le_bytes`/`to_le_bytes` also pins the byte order:
//! every supported target is little-endian today, so this matches the
//! historical native-endian writers bit-for-bit while making the
//! formats portable.

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// On-disk width of one f32 value.
const F32_BYTES: usize = core::mem::size_of::<f32>();

/// Decode a little-endian f32 byte stream. Works on unaligned input.
/// Trailing bytes beyond the last complete 4-byte word are ignored —
/// callers that require an exact length validate it themselves.
pub fn decode_f32_le(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(F32_BYTES)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Encode f32 values as a little-endian byte stream.
pub fn encode_f32_le(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * F32_BYTES);
    for v in vals {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_is_bit_exact() {
        let vals = [0.0f32, 1.0, -1.0, 3.25, f32::MAX, f32::MIN_POSITIVE, -0.0];
        let bytes = encode_f32_le(&vals);
        assert_eq!(bytes.len(), vals.len() * F32_BYTES);
        let back = decode_f32_le(&bytes);
        for (a, b) in vals.iter().zip(back.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    #[test]
    fn decode_handles_misaligned_input() {
        // Prefix one pad byte so the f32 payload starts at an odd
        // address — the transmute-based decoder this replaced was UB
        // on exactly this input.
        let vals = [1.5f32, -2.25, 1e-20];
        let mut buf = vec![0xAAu8];
        buf.extend_from_slice(&encode_f32_le(&vals));
        let back = decode_f32_le(&buf[1..]);
        assert_eq!(back.len(), vals.len());
        for (a, b) in vals.iter().zip(back.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    #[test]
    fn decode_ignores_trailing_partial_word() {
        let mut bytes = encode_f32_le(&[42.0f32]);
        bytes.extend_from_slice(&[1, 2, 3]); // 3 trailing bytes < one f32
        assert_eq!(decode_f32_le(&bytes), vec![42.0f32]);
    }

    #[test]
    fn encode_matches_le_byte_layout() {
        // 1.0f32 = 0x3F800000 → LE bytes [00, 00, 80, 3F].
        assert_eq!(encode_f32_le(&[1.0]), vec![0x00, 0x00, 0x80, 0x3F]);
    }
}
