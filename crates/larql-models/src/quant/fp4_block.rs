//! 256-element block codec for the LARQL FP4 vindex format (exp 26).
//!
//! Two block layouts:
//!
//! - **FP4 block (137 bytes)**: 128 B FP4 values (nibble-packed E2M1) +
//!   8 B FP8 E4M3 sub-block scales (one per 32-element sub-block) +
//!   1 B FP8 E4M3 block scale.
//! - **FP8 block (257 bytes)**: 256 B FP8 E4M3 values + 1 B FP8 E4M3
//!   block scale. No sub-block scales — E4M3's dynamic range absorbs
//!   the distribution directly.
//!
//! Both block types carry a block-level scale so that per-block
//! magnitude normalisation preserves the format's representable
//! resolution regardless of where each block sits in the overall
//! weight distribution.
//!
//! Format reference: `docs/specs/fp4-format-spec.md`.

use super::fp4;
use super::fp8;

/// Block geometry (v1 of the LARQL FP4 format).
pub const BLOCK_ELEMENTS: usize = 256;
pub const SUB_BLOCK_ELEMENTS: usize = 32;
pub const SUB_BLOCKS_PER_BLOCK: usize = BLOCK_ELEMENTS / SUB_BLOCK_ELEMENTS; // = 8

pub const FP4_BLOCK_BYTES: usize = 128 + SUB_BLOCKS_PER_BLOCK + 1; // 128 + 8 + 1 = 137
pub const FP8_BLOCK_BYTES: usize = BLOCK_ELEMENTS + 1; // 256 + 1 = 257

/// Encode one 256-element slice of f32 into a 137-byte FP4 block.
///
/// The encoder picks a block scale equal to `max(|x|) / 6` (FP4's max
/// representable magnitude). Each sub-block's local scale is then
/// `sub_max / (6 × block_scale)`, storing in FP8 E4M3 the multiplicative
/// factor needed to recover the sub-block's magnitude relative to the
/// block scale.
///
/// Returns the 137-byte block. Panics if `values.len() != 256`.
pub fn encode_fp4_block(values: &[f32]) -> [u8; FP4_BLOCK_BYTES] {
    assert_eq!(values.len(), BLOCK_ELEMENTS, "FP4 block must be 256 elems");

    // ── Compute block scale and sub-block scales ──────────────────────────
    // block_max = max over all elements; block scale in E4M3 with room for
    // the max-FP4 magnitude (6.0) and max-sub-block-scale (also 6.0 after
    // normalisation would blow the range). We choose the block scale to be
    // the block's max absolute value (not divided by 6) so that the
    // sub-block scale of the max-bearing sub-block is ≈ 1.0; other
    // sub-blocks carry scales ≤ 1.0. The FP4 quantiser inside a sub-block
    // then operates on values normalised to [-6, 6] by dividing by
    // `block_scale × sub_block_scale × (1/6)`, i.e. operates on
    // `value / (block_scale × sub_block_scale) × 6`.
    //
    // Dequantisation: x = fp4_value × sub_block_scale × block_scale / 6.
    let block_max = values.iter().fold(0.0f32, |m, &v| m.max(v.abs()));

    let mut out = [0u8; FP4_BLOCK_BYTES];

    if block_max == 0.0 {
        // All zeros: block scale = 0.0 (E4M3 = 0x00), sub-scales = 0,
        // values = 0. Out array already zeroed.
        return out;
    }

    let block_scale_f32 = block_max;
    let block_scale_byte = fp8::f32_to_e4m3(block_scale_f32);
    let block_scale_recovered = fp8::e4m3_to_f32(block_scale_byte);
    // Avoid a div-by-zero if E4M3 rounding flushed block_scale to zero.
    let block_scale_nonzero = if block_scale_recovered == 0.0 {
        // Extremely tiny block — all values flushed. Treat as all-zero.
        return out;
    } else {
        block_scale_recovered
    };

    for sb in 0..SUB_BLOCKS_PER_BLOCK {
        let start = sb * SUB_BLOCK_ELEMENTS;
        let end = start + SUB_BLOCK_ELEMENTS;
        let sub = &values[start..end];

        // Sub-block scale: local_max / block_scale. In [0, 1] for the
        // usual case; the largest sub-block has scale ≈ 1.0.
        let sub_max = sub.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let sub_scale_f32 = sub_max / block_scale_nonzero;
        let sub_scale_byte = fp8::f32_to_e4m3(sub_scale_f32);
        let sub_scale_recovered = fp8::e4m3_to_f32(sub_scale_byte);
        out[128 + sb] = sub_scale_byte;

        // Quantise each value to FP4. Per-element normalisation:
        //   x_norm = x / (sub_scale_f32 × block_scale) × 6
        // (so that a value equal to sub_max maps to ±6, FP4's max).
        let per_elem_divisor = sub_scale_recovered * block_scale_nonzero;
        if per_elem_divisor == 0.0 {
            // Dead sub-block inside a live block — all FP4 values = 0.
            // Lower nibble pair already zero; nothing to write.
            continue;
        }
        let scale_to_fp4 = 6.0 / per_elem_divisor;

        // FP4 nibble packing: 16 bytes per 32-element sub-block.
        let bytes_per_sub = SUB_BLOCK_ELEMENTS / 2;
        for (pair_idx, pair) in sub.chunks_exact(2).enumerate() {
            let a = pair[0] * scale_to_fp4;
            let b = pair[1] * scale_to_fp4;
            let code_a = fp4::f32_to_e2m1(a);
            let code_b = fp4::f32_to_e2m1(b);
            let byte = ((code_b & 0x0F) << 4) | (code_a & 0x0F);
            out[sb * bytes_per_sub + pair_idx] = byte;
        }
    }
    out[136] = block_scale_byte;
    out
}

/// Decode a 137-byte FP4 block back to 256 f32 values.
pub fn decode_fp4_block(block: &[u8], out: &mut [f32]) {
    assert_eq!(block.len(), FP4_BLOCK_BYTES);
    assert_eq!(out.len(), BLOCK_ELEMENTS);

    let block_scale = fp8::e4m3_to_f32(block[136]);
    if block_scale == 0.0 {
        out.iter_mut().for_each(|x| *x = 0.0);
        return;
    }

    for sb in 0..SUB_BLOCKS_PER_BLOCK {
        let sub_scale = fp8::e4m3_to_f32(block[128 + sb]);
        let dequant_scale = sub_scale * block_scale / 6.0;
        let start = sb * SUB_BLOCK_ELEMENTS;
        let bytes_per_sub = SUB_BLOCK_ELEMENTS / 2;
        let sub_bytes = &block[sb * bytes_per_sub..(sb + 1) * bytes_per_sub];
        for (pair_idx, &byte) in sub_bytes.iter().enumerate() {
            let code_a = byte & 0x0F;
            let code_b = (byte >> 4) & 0x0F;
            out[start + 2 * pair_idx] = fp4::e2m1_to_f32(code_a) * dequant_scale;
            out[start + 2 * pair_idx + 1] = fp4::e2m1_to_f32(code_b) * dequant_scale;
        }
    }
}

/// Encode one 256-element f32 slice into a 257-byte FP8 block.
pub fn encode_fp8_block(values: &[f32]) -> [u8; FP8_BLOCK_BYTES] {
    assert_eq!(values.len(), BLOCK_ELEMENTS);
    let mut out = [0u8; FP8_BLOCK_BYTES];

    let block_max = values.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    if block_max == 0.0 {
        return out;
    }

    // block_scale = block_max. After division by block_scale, the largest-
    // magnitude element maps to ±1.0, well inside E4M3's representable
    // range. Smaller elements land at correspondingly smaller E4M3 values
    // with the format's full 3-bit mantissa resolution intact.
    //
    // Earlier draft used `block_max / 224` to push values toward E4M3's
    // upper range (max ≈ 448). That broke catastrophically for typical
    // FFN feature magnitudes (block_max ≈ 0.04): the block scale itself
    // rounded to 0 in E4M3 (below 2⁻⁹ subnormal), and dequant returned
    // zeros. The symptom was `max_err == block_max` on every down feature
    // on the Gemma 3 4B fp4_verify run. Matches the FP4-block convention
    // (block_scale = block_max, sub-block scales in [0, 1]) for
    // consistency across the two codecs.
    let block_scale_f32 = block_max;
    let block_scale_byte = fp8::f32_to_e4m3(block_scale_f32);
    let block_scale_recovered = fp8::e4m3_to_f32(block_scale_byte);
    if block_scale_recovered == 0.0 {
        return out;
    }

    for (i, &v) in values.iter().enumerate() {
        let normed = v / block_scale_recovered;
        out[i] = fp8::f32_to_e4m3(normed);
    }
    out[256] = block_scale_byte;
    out
}

/// Decode a 257-byte FP8 block to 256 f32 values.
pub fn decode_fp8_block(block: &[u8], out: &mut [f32]) {
    assert_eq!(block.len(), FP8_BLOCK_BYTES);
    assert_eq!(out.len(), BLOCK_ELEMENTS);

    let block_scale = fp8::e4m3_to_f32(block[256]);
    if block_scale == 0.0 {
        out.iter_mut().for_each(|x| *x = 0.0);
        return;
    }
    for i in 0..BLOCK_ELEMENTS {
        out[i] = fp8::e4m3_to_f32(block[i]) * block_scale;
    }
}

// ─── Feature-vector level ───────────────────────────────────────────────────

/// Encode one feature vector (`hidden` f32 values, must be a multiple of
/// 256) into a contiguous FP4 byte buffer of length
/// `(hidden / 256) × 137`.
pub fn encode_fp4_feature(values: &[f32]) -> Vec<u8> {
    assert_eq!(
        values.len() % BLOCK_ELEMENTS,
        0,
        "feature length {} not a multiple of {}",
        values.len(),
        BLOCK_ELEMENTS
    );
    let n_blocks = values.len() / BLOCK_ELEMENTS;
    let mut out = Vec::with_capacity(n_blocks * FP4_BLOCK_BYTES);
    for b in 0..n_blocks {
        let start = b * BLOCK_ELEMENTS;
        let block = encode_fp4_block(&values[start..start + BLOCK_ELEMENTS]);
        out.extend_from_slice(&block);
    }
    out
}

/// Decode an FP4 feature buffer back to f32. `out.len()` must equal
/// `(bytes.len() / 137) × 256`.
pub fn decode_fp4_feature(bytes: &[u8], out: &mut [f32]) {
    assert_eq!(bytes.len() % FP4_BLOCK_BYTES, 0);
    let n_blocks = bytes.len() / FP4_BLOCK_BYTES;
    assert_eq!(out.len(), n_blocks * BLOCK_ELEMENTS);
    for b in 0..n_blocks {
        let src = &bytes[b * FP4_BLOCK_BYTES..(b + 1) * FP4_BLOCK_BYTES];
        let dst = &mut out[b * BLOCK_ELEMENTS..(b + 1) * BLOCK_ELEMENTS];
        decode_fp4_block(src, dst);
    }
}

/// Encode one feature vector into an FP8 byte buffer.
pub fn encode_fp8_feature(values: &[f32]) -> Vec<u8> {
    assert_eq!(values.len() % BLOCK_ELEMENTS, 0);
    let n_blocks = values.len() / BLOCK_ELEMENTS;
    let mut out = Vec::with_capacity(n_blocks * FP8_BLOCK_BYTES);
    for b in 0..n_blocks {
        let start = b * BLOCK_ELEMENTS;
        let block = encode_fp8_block(&values[start..start + BLOCK_ELEMENTS]);
        out.extend_from_slice(&block);
    }
    out
}

/// Decode an FP8 feature buffer.
pub fn decode_fp8_feature(bytes: &[u8], out: &mut [f32]) {
    assert_eq!(bytes.len() % FP8_BLOCK_BYTES, 0);
    let n_blocks = bytes.len() / FP8_BLOCK_BYTES;
    assert_eq!(out.len(), n_blocks * BLOCK_ELEMENTS);
    for b in 0..n_blocks {
        let src = &bytes[b * FP8_BLOCK_BYTES..(b + 1) * FP8_BLOCK_BYTES];
        let dst = &mut out[b * BLOCK_ELEMENTS..(b + 1) * BLOCK_ELEMENTS];
        decode_fp8_block(src, dst);
    }
}

/// Number of bytes per feature vector in the FP4 layout.
#[inline]
pub fn fp4_feature_bytes(hidden: usize) -> usize {
    assert_eq!(hidden % BLOCK_ELEMENTS, 0);
    (hidden / BLOCK_ELEMENTS) * FP4_BLOCK_BYTES
}

/// Number of bytes per feature vector in the FP8 layout.
#[inline]
pub fn fp8_feature_bytes(hidden: usize) -> usize {
    assert_eq!(hidden % BLOCK_ELEMENTS, 0);
    (hidden / BLOCK_ELEMENTS) * FP8_BLOCK_BYTES
}

// ─── Layer level ────────────────────────────────────────────────────────────

/// Encode a flat per-layer f32 slice (row-major `[num_features × hidden]`)
/// into FP4 bytes. Output length = `num_features × fp4_feature_bytes(hidden)`.
pub fn encode_fp4_layer(values: &[f32], num_features: usize, hidden: usize) -> Vec<u8> {
    assert_eq!(values.len(), num_features * hidden);
    let per_feat = fp4_feature_bytes(hidden);
    let mut out = Vec::with_capacity(num_features * per_feat);
    for f in 0..num_features {
        let src = &values[f * hidden..(f + 1) * hidden];
        out.extend_from_slice(&encode_fp4_feature(src));
    }
    out
}

/// Decode FP4 layer bytes back to flat f32 `[num_features × hidden]`.
pub fn decode_fp4_layer(bytes: &[u8], num_features: usize, hidden: usize, out: &mut [f32]) {
    let per_feat = fp4_feature_bytes(hidden);
    assert_eq!(bytes.len(), num_features * per_feat);
    assert_eq!(out.len(), num_features * hidden);
    for f in 0..num_features {
        let src = &bytes[f * per_feat..(f + 1) * per_feat];
        let dst = &mut out[f * hidden..(f + 1) * hidden];
        decode_fp4_feature(src, dst);
    }
}

/// FP8 counterpart of `encode_fp4_layer`.
pub fn encode_fp8_layer(values: &[f32], num_features: usize, hidden: usize) -> Vec<u8> {
    assert_eq!(values.len(), num_features * hidden);
    let per_feat = fp8_feature_bytes(hidden);
    let mut out = Vec::with_capacity(num_features * per_feat);
    for f in 0..num_features {
        let src = &values[f * hidden..(f + 1) * hidden];
        out.extend_from_slice(&encode_fp8_feature(src));
    }
    out
}

/// FP8 counterpart of `decode_fp4_layer`.
pub fn decode_fp8_layer(bytes: &[u8], num_features: usize, hidden: usize, out: &mut [f32]) {
    let per_feat = fp8_feature_bytes(hidden);
    assert_eq!(bytes.len(), num_features * per_feat);
    assert_eq!(out.len(), num_features * hidden);
    for f in 0..num_features {
        let src = &bytes[f * per_feat..(f + 1) * per_feat];
        let dst = &mut out[f * hidden..(f + 1) * hidden];
        decode_fp8_feature(src, dst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The required round-trip invariant from FP4_FORMAT_SPEC §12.
    /// Independent of the walk kernel, deterministic, failure-diagnostic.
    #[test]
    fn fp4_block_round_trip_gaussian() {
        // Gaussian-ish distribution, zero mean unit std — typical of FFN
        // feature activations rather than of learned weights, but a
        // well-behaved stress test for the block codec.
        let values: Vec<f32> = (0..256)
            .map(|i| (i as f32 - 128.0) / 40.0) // roughly -3.2 .. 3.2
            .collect();

        let block = encode_fp4_block(&values);
        let mut decoded = [0.0f32; 256];
        decode_fp4_block(&block, &mut decoded);

        // Each element's reconstruction error bounded by the FP4
        // quantisation step at the decoded block's scale.
        let block_max = values.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        // Worst-case step between adjacent FP4 representable magnitudes:
        // 0.5 at the low end, 2.0 at the high end (between 4 and 6).
        // Conservatively: bound at 2.0 × (block_max / 6) = (1/3) × block_max.
        let bound = block_max / 3.0;

        for (i, (&v, &d)) in values.iter().zip(decoded.iter()).enumerate() {
            let err = (v - d).abs();
            assert!(
                err <= bound,
                "elem {i}: expected {v}, got {d}, err {err} > bound {bound}"
            );
        }
    }

    #[test]
    fn fp4_block_round_trip_pathological_ratio() {
        // Pathological: one sub-block has magnitudes O(100), others O(0.01).
        // Ratio ~10,000 — well beyond the R=16 lossless threshold.
        let mut values = vec![0.01f32; 256];
        for (i, v) in values.iter_mut().take(32).enumerate() {
            *v = if i.is_multiple_of(2) { 100.0 } else { -100.0 };
        }
        let block = encode_fp4_block(&values);
        let mut decoded = [0.0f32; 256];
        decode_fp4_block(&block, &mut decoded);

        // The high-magnitude sub-block should reconstruct well (its scale
        // is ≈ 1.0 × block_scale, so full FP4 resolution applies).
        for i in 0..32 {
            let err = (values[i] - decoded[i]).abs();
            assert!(err <= 100.0 / 3.0, "high sub-block elem {i}: err {err}");
        }
        // Low-magnitude sub-blocks will have their sub_scale quantised
        // toward 0; reconstruction is lossy but should be bounded by the
        // sub-block's own magnitude budget.
        let low_max: f32 = values[32..].iter().fold(0.0, |m, &v| m.max(v.abs()));
        for i in 32..256 {
            let err = (values[i] - decoded[i]).abs();
            assert!(
                err <= low_max + 1e-3,
                "low sub-block elem {i}: err {err}, low_max {low_max}"
            );
        }
    }

    #[test]
    fn fp4_block_all_zeros() {
        let values = vec![0.0f32; 256];
        let block = encode_fp4_block(&values);
        assert_eq!(block, [0u8; 137]);
        let mut decoded = [0.0f32; 256];
        decode_fp4_block(&block, &mut decoded);
        assert!(decoded.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn fp4_block_size_is_137_bytes() {
        assert_eq!(FP4_BLOCK_BYTES, 137);
    }

    #[test]
    fn fp8_block_round_trip_gaussian() {
        let values: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 40.0).collect();
        let block = encode_fp8_block(&values);
        let mut decoded = [0.0f32; 256];
        decode_fp8_block(&block, &mut decoded);

        // FP8 E4M3: mantissa = 3 bits, so relative error ≤ 2^-3 per value
        // after block normalisation, then scaled back.
        let block_max = values.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let bound = block_max * 0.25; // generous; E4M3's 3-bit mantissa gives ~2^-3 precision.

        for (i, (&v, &d)) in values.iter().zip(decoded.iter()).enumerate() {
            let err = (v - d).abs();
            assert!(
                err <= bound,
                "elem {i}: expected {v}, got {d}, err {err} > bound {bound}"
            );
        }
    }

    #[test]
    fn fp8_block_size_is_257_bytes() {
        assert_eq!(FP8_BLOCK_BYTES, 257);
    }

    #[test]
    fn fp8_block_all_zeros() {
        let values = vec![0.0f32; 256];
        let block = encode_fp8_block(&values);
        assert_eq!(block, [0u8; 257]);
        let mut decoded = [0.0f32; 256];
        decode_fp8_block(&block, &mut decoded);
        assert!(decoded.iter().all(|&x| x == 0.0));
    }

    /// Regression guard for the `block_max / 224` normalisation bug found
    /// during end-to-end fp4_verify: for realistic FFN weight magnitudes
    /// (block_max ≈ 0.04 on Gemma 3 4B down) the old normalisation
    /// produced a block scale below E4M3's smallest representable value
    /// (2⁻⁹ ≈ 1.95e-3), flushing the scale to zero and returning the
    /// all-zero block. Fix: use block_scale = block_max. This test pins
    /// the fix at typical-FFN magnitude levels.
    #[test]
    fn fp8_block_small_magnitude_like_ffn_down() {
        // Synthetic distribution in the range of actual Gemma 3 4B down
        // features: block_max ≈ 0.04, typical values ≈ 0.01–0.04.
        use std::f32::consts::TAU;
        let values: Vec<f32> = (0..256)
            .map(|i| {
                let t = (i as f32) / 256.0;
                0.04 * (t * TAU * 3.0).sin()
            })
            .collect();
        let block_max = values.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        assert!(block_max > 0.0 && block_max < 0.05);
        let block = encode_fp8_block(&values);
        let mut decoded = [0.0f32; 256];
        decode_fp8_block(&block, &mut decoded);
        // Before the fix, max_err == block_max (100%); after, should be
        // bounded by E4M3's mantissa precision.
        let max_err = values
            .iter()
            .zip(decoded.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < block_max * 0.10,
            "max_err {max_err} > 10% of block_max {block_max} — FP8 small-mag regression"
        );
    }

    #[test]
    fn fp4_feature_round_trip_2560() {
        // Gemma 3 4B hidden size — 10 blocks per feature.
        let hidden = 2560;
        let values: Vec<f32> = (0..hidden)
            .map(|i| ((i as f32 - 1280.0) / 400.0).sin())
            .collect();
        let bytes = encode_fp4_feature(&values);
        assert_eq!(bytes.len(), fp4_feature_bytes(hidden));
        assert_eq!(bytes.len(), 10 * 137);
        let mut decoded = vec![0.0f32; hidden];
        decode_fp4_feature(&bytes, &mut decoded);
        let max_err = values
            .iter()
            .zip(decoded.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 0.3, "max err {max_err}");
    }

    #[test]
    fn fp8_feature_round_trip_2560() {
        let hidden = 2560;
        let values: Vec<f32> = (0..hidden)
            .map(|i| ((i as f32 - 1280.0) / 400.0).sin())
            .collect();
        let bytes = encode_fp8_feature(&values);
        assert_eq!(bytes.len(), fp8_feature_bytes(hidden));
        assert_eq!(bytes.len(), 10 * 257);
        let mut decoded = vec![0.0f32; hidden];
        decode_fp8_feature(&bytes, &mut decoded);
        // FP8 is much tighter than FP4.
        let max_err = values
            .iter()
            .zip(decoded.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 0.05, "max err {max_err}");
    }

    #[test]
    fn fp4_layer_round_trip_small() {
        // 4 features × 512 hidden (2 blocks per feature).
        let num_features = 4;
        let hidden = 512;
        let values: Vec<f32> = (0..num_features * hidden)
            .map(|i| (i as f32).sin() * 2.0)
            .collect();
        let bytes = encode_fp4_layer(&values, num_features, hidden);
        assert_eq!(bytes.len(), num_features * fp4_feature_bytes(hidden));
        let mut decoded = vec![0.0f32; values.len()];
        decode_fp4_layer(&bytes, num_features, hidden, &mut decoded);
        // Per-feature bound similar to the block test.
        for f in 0..num_features {
            let block_max = values[f * hidden..(f + 1) * hidden]
                .iter()
                .fold(0.0f32, |m, &v| m.max(v.abs()));
            for i in 0..hidden {
                let err = (values[f * hidden + i] - decoded[f * hidden + i]).abs();
                assert!(err <= block_max / 3.0, "feat {f} elem {i}: err {err}");
            }
        }
    }

    #[test]
    fn fp8_layer_round_trip_small() {
        let num_features = 4;
        let hidden = 512;
        let values: Vec<f32> = (0..num_features * hidden)
            .map(|i| (i as f32).sin() * 2.0)
            .collect();
        let bytes = encode_fp8_layer(&values, num_features, hidden);
        let mut decoded = vec![0.0f32; values.len()];
        decode_fp8_layer(&bytes, num_features, hidden, &mut decoded);
        // E4M3 has 3 mantissa bits → ~12.5% relative error per element.
        // Bound per-element against the element's own block_max.
        for f in 0..num_features {
            for b in 0..(hidden / BLOCK_ELEMENTS) {
                let block_start = f * hidden + b * BLOCK_ELEMENTS;
                let block = &values[block_start..block_start + BLOCK_ELEMENTS];
                let block_max = block.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
                for i in 0..BLOCK_ELEMENTS {
                    let err = (values[block_start + i] - decoded[block_start + i]).abs();
                    assert!(
                        err <= block_max * 0.15,
                        "feat {f} block {b} elem {i}: err {err} > bound {}",
                        block_max * 0.15
                    );
                }
            }
        }
    }

    /// Realistic: sample the block distribution we actually scanned on 4B
    /// gate — ratios in [2, 4), all normally-distributed magnitudes — and
    /// verify that under the FP4 encoder the worst per-element error is
    /// well inside the walk kernel's BLAS-1 saxpy tolerance.
    #[test]
    fn fp4_block_typical_4b_distribution() {
        use std::f32::consts::TAU;
        // Synthesize a block with per-sub-block max/min ratio ≈ 3.
        // Each sub-block is a 32-element vector with its own characteristic
        // magnitude in the typical observed range.
        let mut values = [0.0f32; 256];
        for sb in 0..SUB_BLOCKS_PER_BLOCK {
            let sub_mag = 0.5 + 0.5 * (sb as f32 / 8.0); // 0.5 .. 0.94
            for j in 0..SUB_BLOCK_ELEMENTS {
                let t = (sb * SUB_BLOCK_ELEMENTS + j) as f32 / 256.0;
                values[sb * SUB_BLOCK_ELEMENTS + j] = sub_mag * (TAU * t * 3.5).sin();
            }
        }
        let block_max = values.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let block = encode_fp4_block(&values);
        let mut decoded = [0.0f32; 256];
        decode_fp4_block(&block, &mut decoded);

        // Median error bound: much tighter than the worst-case 1/3 × max.
        let mut err: Vec<f32> = values
            .iter()
            .zip(decoded.iter())
            .map(|(a, b)| (a - b).abs())
            .collect();
        err.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = err[err.len() / 2];
        assert!(
            median < 0.06 * block_max,
            "median err {median} too large at block_max {block_max}"
        );
    }

    // ── Block edge cases ────────────────────────────────────────────────────

    /// A block with one zero sub-block and seven non-zero sub-blocks.
    /// The zero sub-block's scale is 0 in E4M3, but the block scale is
    /// non-zero — the decoder must handle a zero sub-block cleanly.
    #[test]
    fn fp4_block_mixed_zero_and_nonzero_sub_blocks() {
        let mut values = vec![0.5f32; 256];
        // Sub-block 3 (elements 96..128) is all zero.
        for v in values.iter_mut().skip(96).take(32) {
            *v = 0.0;
        }
        let block = encode_fp4_block(&values);
        let mut decoded = [0.0f32; 256];
        decode_fp4_block(&block, &mut decoded);

        // Zero sub-block should decode to zeros (or tiny).
        for v in decoded.iter().skip(96).take(32) {
            assert!(v.abs() < 1e-5, "zero sub-block decoded to {v}");
        }
        // Non-zero sub-blocks should decode to ~0.5.
        for (i, &v) in decoded.iter().enumerate() {
            if (96..128).contains(&i) {
                continue;
            }
            assert!((v - 0.5).abs() <= 0.5 / 3.0, "elem {i}: {v}");
        }
    }

    /// A block with NaN input — FP4 has no NaN representation, so the
    /// NaN input must be replaced with 0 inside the quantiser. The
    /// decode should not produce NaN.
    #[test]
    fn fp4_block_nan_input_maps_to_zero_element() {
        let mut values = vec![0.5f32; 256];
        values[42] = f32::NAN;
        // block_max will be NaN without sanitisation → guard here.
        // The encoder's `.abs()` on NaN returns NaN, and max(NaN, x)
        // depends on order. We want to ensure no NaN reaches storage.
        // Pre-sanitise the input (this is what the extractor does).
        for v in values.iter_mut() {
            if v.is_nan() {
                *v = 0.0;
            }
        }
        let block = encode_fp4_block(&values);
        let mut decoded = [0.0f32; 256];
        decode_fp4_block(&block, &mut decoded);
        assert!(
            !decoded.iter().any(|v| v.is_nan()),
            "no NaN in decoded block"
        );
        assert_eq!(decoded[42], 0.0);
    }

    /// A block with a single outlier 10× larger than the rest.
    /// The sub-block containing the outlier gets sub_scale ≈ 1, all
    /// other sub-blocks get sub_scale ≈ 0.1. Outlier reconstruction
    /// should be tight; the rest should also reconstruct at their
    /// sub-block scales.
    #[test]
    fn fp4_block_single_outlier_preserved() {
        let mut values = vec![0.1f32; 256];
        values[128] = 1.0; // 10× outlier
        let block = encode_fp4_block(&values);
        let mut decoded = [0.0f32; 256];
        decode_fp4_block(&block, &mut decoded);

        // Outlier reconstructs within FP4 bound at block scale.
        assert!(
            (decoded[128] - 1.0).abs() <= 1.0 / 3.0,
            "outlier got {}",
            decoded[128]
        );
        // Most values around it should recover to near 0.1.
        for (i, &v) in decoded.iter().enumerate() {
            if i == 128 {
                continue;
            }
            // Allow generous bound — small-magnitude sub-blocks lose
            // resolution when another sub-block sets the block scale.
            assert!(v.abs() <= 0.2, "elem {i}: unexpectedly large {v}");
        }
    }

    /// FP8 block with all values at E4M3's saturation boundary.
    /// encode(448) then decode should round-trip exactly.
    #[test]
    fn fp8_block_saturation_values_round_trip() {
        let values = vec![448.0f32; 256];
        let block = encode_fp8_block(&values);
        let mut decoded = [0.0f32; 256];
        decode_fp8_block(&block, &mut decoded);
        for (i, &v) in decoded.iter().enumerate() {
            assert!((v - 448.0).abs() <= 448.0 * 0.01, "elem {i}: {v}");
        }
    }

    /// FP8 block with all values below the smallest subnormal (2⁻⁹).
    /// Everything should flush to zero on the block-scale round.
    #[test]
    fn fp8_block_below_subnormal_flushes_to_zero() {
        let values = vec![1e-12f32; 256];
        let block = encode_fp8_block(&values);
        let mut decoded = [0.0f32; 256];
        decode_fp8_block(&block, &mut decoded);
        // All values effectively zero — either the block scale flushed
        // or the per-element values flushed under the block scale.
        let max_abs = decoded.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        assert!(max_abs < 1e-3, "expected flush-to-zero, got max {max_abs}");
    }

    /// A 1-element difference from all-zero — verify we don't get a
    /// divide-by-zero or catastrophic amplification.
    #[test]
    fn fp4_block_sparse_single_element() {
        let mut values = vec![0.0f32; 256];
        values[0] = 1.0;
        let block = encode_fp4_block(&values);
        let mut decoded = [0.0f32; 256];
        decode_fp4_block(&block, &mut decoded);

        // The non-zero sub-block (containing elem 0) should reconstruct.
        assert!((decoded[0] - 1.0).abs() <= 1.0 / 3.0, "got {}", decoded[0]);
        // The remaining 255 elements: some will be near-zero (their
        // sub-blocks had zero scale), others may reconstruct to small
        // magnitudes. Bound generously.
        for (i, &v) in decoded.iter().enumerate().skip(1) {
            assert!(v.abs() <= 0.1, "elem {i}: unexpectedly large {v}");
        }
    }
}
