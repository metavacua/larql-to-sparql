//! FP4 E2M1 ↔ f32 conversion and nibble-pair packing.
//!
//! FP4 E2M1 per the OCP MXFP4 v1.0 specification:
//! 1 sign bit, 2 exponent bits (bias 1), 1 mantissa bit.
//! Representable values: `{±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}`.
//!
//! The value table matches `crate::quant::mxfp4::MXFP4_TABLE`; this
//! module exposes the same lookup through a stable entry point for the
//! LARQL FP4 vindex format (exp 26), plus the nibble-pair packing and
//! f32→E2M1 encoder that are not in the mxfp4 module (which is
//! dequantisation-only for GPT-OSS inbound weights).
//!
//! Byte packing convention: `byte[i] = (v[2i+1] << 4) | (v[2i] & 0x0F)`
//! — lower nibble holds the even-indexed element. This matches the
//! LARQL format spec §5.1.

/// FP4 E2M1 value lookup. Index 0..15 maps the 4-bit encoding to f32.
/// Must remain byte-identical to `mxfp4::MXFP4_TABLE`.
pub const FP4_E2M1_TABLE: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// The 8 positive representable magnitudes (not counting ±0).
const POSITIVE_MAGS: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// Convert a 4-bit E2M1 code to f32.
#[inline]
pub fn e2m1_to_f32(code: u8) -> f32 {
    FP4_E2M1_TABLE[(code & 0x0F) as usize]
}

/// Convert f32 to the nearest E2M1 4-bit code using round-to-nearest-even.
///
/// Saturates to ±6 on overflow. FP4 has no NaN representation; NaN
/// inputs map to +0 (matching DeepSeek-V4's behaviour and OCP guidance
/// that NaNs should not appear in FP4 storage).
#[inline]
pub fn f32_to_e2m1(value: f32) -> u8 {
    if value.is_nan() {
        return 0x00;
    }

    let sign_bit: u8 = if value.is_sign_negative() { 0x08 } else { 0x00 };
    let mag = value.abs();

    // FP4 has no Inf. ±Inf saturates to ±6 (code 7 / 15). Without this
    // early-out, the iteration below computes `(Inf - m).abs() = Inf`
    // for every magnitude, and `err < best_err` never fires → bestidx
    // stays at 0 (zero), which is wrong: saturating to 6 is the
    // documented contract.
    if mag.is_infinite() {
        return sign_bit | 7;
    }

    // Find the best magnitude slot via round-to-nearest-even. Representable
    // positive magnitudes: [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0].
    let mut best_idx = 0usize;
    let mut best_err = (mag - POSITIVE_MAGS[0]).abs();
    for (i, &m) in POSITIVE_MAGS.iter().enumerate().skip(1) {
        let err = (mag - m).abs();
        if err < best_err {
            best_idx = i;
            best_err = err;
        } else if err == best_err {
            // Tie: pick the one whose encoded index is even.
            if (i & 1) == 0 {
                best_idx = i;
            }
        }
    }
    sign_bit | (best_idx as u8)
}

/// Pack a slice of E2M1 codes (length must be even) into nibble-packed
/// bytes. `byte[i] = (code[2i+1] << 4) | (code[2i] & 0x0F)`.
pub fn pack_nibbles(codes: &[u8]) -> Vec<u8> {
    assert!(
        codes.len().is_multiple_of(2),
        "nibble packing requires even length"
    );
    let mut out = Vec::with_capacity(codes.len() / 2);
    for pair in codes.chunks_exact(2) {
        out.push(((pair[1] & 0x0F) << 4) | (pair[0] & 0x0F));
    }
    out
}

/// Unpack nibble-packed bytes into E2M1 codes.
pub fn unpack_nibbles(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(b & 0x0F);
        out.push((b >> 4) & 0x0F);
    }
    out
}

/// Decode a nibble-packed FP4 byte slice directly to f32 values via the
/// lookup table. `out.len()` must be `bytes.len() * 2`.
#[inline]
pub fn decode_fp4_into(bytes: &[u8], out: &mut [f32]) {
    debug_assert_eq!(out.len(), bytes.len() * 2);
    for (i, &b) in bytes.iter().enumerate() {
        out[2 * i] = FP4_E2M1_TABLE[(b & 0x0F) as usize];
        out[2 * i + 1] = FP4_E2M1_TABLE[((b >> 4) & 0x0F) as usize];
    }
}

/// Quantise f32 values to E2M1 codes (no packing). Round-to-nearest-even
/// on ties. Length preserved.
pub fn quantise_fp4(values: &[f32]) -> Vec<u8> {
    values.iter().map(|&v| f32_to_e2m1(v)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp4_table_matches_mxfp4() {
        use crate::quant::mxfp4;
        // Exported table must be byte-identical to the MXFP4 one; otherwise
        // downstream code that reuses MXFP4 would disagree with ours.
        for (i, (&a, &b)) in FP4_E2M1_TABLE
            .iter()
            .zip(mxfp4::MXFP4_TABLE.iter())
            .enumerate()
        {
            assert_eq!(a.to_bits(), b.to_bits(), "disagreement at index {i}");
        }
    }

    #[test]
    fn fp4_representable_round_trip() {
        // Every representable value round-trips exactly.
        for code in 0..16u8 {
            let f = e2m1_to_f32(code);
            let back = f32_to_e2m1(f);
            // ±0 both map to 0.0; accept either code.
            if f == 0.0 {
                assert!(back == 0x00 || back == 0x08);
                continue;
            }
            assert_eq!(back, code, "code {code:#x} → {f} → {back:#x}");
        }
    }

    #[test]
    fn fp4_saturation() {
        assert_eq!(e2m1_to_f32(f32_to_e2m1(100.0)), 6.0);
        assert_eq!(e2m1_to_f32(f32_to_e2m1(-100.0)), -6.0);
    }

    #[test]
    fn fp4_rounding_to_nearest_even() {
        // Halfway between 4.0 (code 0b110, odd-index 6) and 6.0 (code 0b111,
        // odd-index 7). Round-to-nearest-even prefers even index → 4.0.
        let mid = 5.0;
        let f = e2m1_to_f32(f32_to_e2m1(mid));
        assert_eq!(f, 4.0);
    }

    #[test]
    fn nibble_pack_unpack_round_trip() {
        let codes: Vec<u8> = (0..32u8).map(|i| i & 0x0F).collect();
        let packed = pack_nibbles(&codes);
        assert_eq!(packed.len(), codes.len() / 2);
        let unpacked = unpack_nibbles(&packed);
        assert_eq!(unpacked, codes);
    }

    #[test]
    fn nibble_pack_order_lower_is_even_index() {
        // Pin the convention: byte[0] lower nibble = code[0], upper = code[1].
        let codes = [0x03u8, 0x0Cu8];
        let packed = pack_nibbles(&codes);
        assert_eq!(packed, vec![0xC3], "lower=0x3 (even), upper=0xC (odd)");
    }

    #[test]
    fn decode_fp4_into_matches_table() {
        let bytes = [0xC3u8, 0x01u8];
        let mut out = [0.0f32; 4];
        decode_fp4_into(&bytes, &mut out);
        // byte 0xC3: lower=3 (→1.5), upper=0xC=12 (→-2.0)
        // byte 0x01: lower=1 (→0.5), upper=0 (→0.0)
        assert_eq!(out, [1.5, -2.0, 0.5, 0.0]);
    }

    // ── Edge cases ──────────────────────────────────────────────────────────

    /// FP4 E2M1 has no NaN representation. Our encoder maps NaN → +0
    /// (code 0x00), matching DeepSeek-V4 and OCP guidance that NaNs
    /// should never appear in FP4 storage.
    #[test]
    fn fp4_nan_input_maps_to_zero() {
        assert_eq!(f32_to_e2m1(f32::NAN), 0x00);
        assert_eq!(e2m1_to_f32(f32_to_e2m1(f32::NAN)), 0.0);
    }

    /// FP4 has no Inf either — ±Inf saturate to ±6 (the max representable).
    #[test]
    fn fp4_inf_saturates() {
        assert_eq!(e2m1_to_f32(f32_to_e2m1(f32::INFINITY)), 6.0);
        assert_eq!(e2m1_to_f32(f32_to_e2m1(f32::NEG_INFINITY)), -6.0);
    }

    /// Very-small positive values that fall below FP4's smallest
    /// non-zero magnitude (0.5) should round to either 0 or 0.5
    /// depending on distance. RTE picks even tie-break.
    #[test]
    fn fp4_subnormal_like_values() {
        // 0.24 is closer to 0 than to 0.5 → rounds to 0.
        assert_eq!(e2m1_to_f32(f32_to_e2m1(0.24)), 0.0);
        // 0.26 is closer to 0.5 → rounds to 0.5.
        assert_eq!(e2m1_to_f32(f32_to_e2m1(0.26)), 0.5);
        // Exactly halfway (0.25): RTE picks the even code. Code 0
        // (magnitude 0.0) is even, code 1 (0.5) is odd → picks 0.
        assert_eq!(e2m1_to_f32(f32_to_e2m1(0.25)), 0.0);
    }

    /// The value encoding preserves sign bit across zero.
    #[test]
    fn fp4_signed_zero() {
        // 0.0 and -0.0 both quantise to *some* code encoding 0.0. The
        // canonical positive zero is 0x00; the negative zero is 0x08.
        // Either is acceptable for round-trip; we only assert the
        // recovered f32 is zero (with correct sign when possible).
        let pos = f32_to_e2m1(0.0);
        let neg = f32_to_e2m1(-0.0);
        // Both should decode to something magnitude-zero.
        assert_eq!(e2m1_to_f32(pos).abs(), 0.0);
        assert_eq!(e2m1_to_f32(neg).abs(), 0.0);
    }

    /// Nibble packing is stable across varying lengths.
    #[test]
    fn fp4_nibble_packing_assorted_lengths() {
        for n in [2usize, 4, 16, 64, 256] {
            let codes: Vec<u8> = (0..n).map(|i| (i as u8) & 0x0F).collect();
            let packed = pack_nibbles(&codes);
            assert_eq!(packed.len(), n / 2);
            let unpacked = unpack_nibbles(&packed);
            assert_eq!(unpacked, codes);
        }
    }
}
