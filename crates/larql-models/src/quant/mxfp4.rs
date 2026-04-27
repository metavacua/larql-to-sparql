//! MXFP4 dequantization — OpenAI's microscaling 4-bit float format.
//!
//! Used by GPT-OSS models. Each weight is stored as a 4-bit value (two per byte)
//! with shared e8m0 (exponent-only) scales per group of 32 elements.
//!
//! Format:
//!   blocks: [experts, out_features, groups, 16] as U8 (each byte = 2 × 4-bit values)
//!   scales: [experts, out_features, groups] as U8 (e8m0 exponent)

use crate::detect::ModelError;

/// MXFP4 lookup table: maps 4-bit value to float.
/// Bit layout: [sign(1)][exponent(2)][mantissa(1)]
/// Values: ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}
const MXFP4_TABLE: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Convert e8m0 scale byte to float multiplier.
/// e8m0 = pure exponent, no mantissa: value = 2^(exponent - 127)
pub fn e8m0_to_f32(byte: u8) -> f32 {
    if byte == 0 { return 0.0; }
    if byte == 255 { return f32::NAN; }
    f32::from_bits((byte as u32) << 23)
}

/// Dequantize a single expert's projection from MXFP4 blocks + scales.
///
/// `blocks` must contain at least `out_features * groups * 16` bytes and
/// `scales` at least `out_features * groups`. Returns `ModelError::Parse` on
/// short input rather than panicking on a slice OOB.
pub fn dequantize_expert(
    blocks: &[u8],
    scales: &[u8],
    out_features: usize,
    groups: usize,
) -> Result<Vec<f32>, ModelError> {
    let in_features = groups * 32;
    let need_blocks = out_features
        .checked_mul(groups)
        .and_then(|v| v.checked_mul(16))
        .ok_or_else(|| {
            ModelError::Parse(format!(
                "MXFP4: block count overflow ({out_features}×{groups}×16)"
            ))
        })?;
    let need_scales = out_features.checked_mul(groups).ok_or_else(|| {
        ModelError::Parse(format!(
            "MXFP4: scale count overflow ({out_features}×{groups})"
        ))
    })?;
    if blocks.len() < need_blocks {
        return Err(ModelError::Parse(format!(
            "MXFP4: blocks too short: {} bytes < expected {need_blocks} (out_features={out_features}, groups={groups})",
            blocks.len()
        )));
    }
    if scales.len() < need_scales {
        return Err(ModelError::Parse(format!(
            "MXFP4: scales too short: {} bytes < expected {need_scales} (out_features={out_features}, groups={groups})",
            scales.len()
        )));
    }

    let mut output = vec![0.0f32; out_features * in_features];

    for row in 0..out_features {
        for g in 0..groups {
            let scale = e8m0_to_f32(scales[row * groups + g]);
            let block_offset = (row * groups + g) * 16;

            for b in 0..16 {
                let byte = blocks[block_offset + b];
                let lo = (byte & 0x0F) as usize;
                let hi = ((byte >> 4) & 0x0F) as usize;

                let out_idx = row * in_features + g * 32 + b * 2;
                output[out_idx] = MXFP4_TABLE[lo] * scale;
                output[out_idx + 1] = MXFP4_TABLE[hi] * scale;
            }
        }
    }

    Ok(output)
}

/// Dequantize all experts from packed MXFP4 tensors.
///
/// Validates that `blocks_data` and `scales_data` hold enough bytes for
/// `num_experts` experts before slicing; returns `ModelError::Parse`
/// otherwise.
pub fn dequantize_all_experts(
    blocks_data: &[u8],
    scales_data: &[u8],
    num_experts: usize,
    out_features: usize,
    groups: usize,
) -> Result<Vec<Vec<f32>>, ModelError> {
    let blocks_per_expert = out_features
        .checked_mul(groups)
        .and_then(|v| v.checked_mul(16))
        .ok_or_else(|| {
            ModelError::Parse(format!(
                "MXFP4: blocks_per_expert overflow ({out_features}×{groups}×16)"
            ))
        })?;
    let scales_per_expert = out_features.checked_mul(groups).ok_or_else(|| {
        ModelError::Parse(format!(
            "MXFP4: scales_per_expert overflow ({out_features}×{groups})"
        ))
    })?;
    let need_blocks = num_experts.checked_mul(blocks_per_expert).ok_or_else(|| {
        ModelError::Parse(format!("MXFP4: total blocks overflow ({num_experts} experts)"))
    })?;
    let need_scales = num_experts.checked_mul(scales_per_expert).ok_or_else(|| {
        ModelError::Parse(format!("MXFP4: total scales overflow ({num_experts} experts)"))
    })?;
    if blocks_data.len() < need_blocks {
        return Err(ModelError::Parse(format!(
            "MXFP4: blocks_data too short: {} bytes < expected {need_blocks} ({num_experts} experts × {blocks_per_expert})",
            blocks_data.len()
        )));
    }
    if scales_data.len() < need_scales {
        return Err(ModelError::Parse(format!(
            "MXFP4: scales_data too short: {} bytes < expected {need_scales} ({num_experts} experts × {scales_per_expert})",
            scales_data.len()
        )));
    }

    (0..num_experts)
        .map(|e| {
            let b_start = e * blocks_per_expert;
            let s_start = e * scales_per_expert;
            dequantize_expert(
                &blocks_data[b_start..b_start + blocks_per_expert],
                &scales_data[s_start..s_start + scales_per_expert],
                out_features,
                groups,
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e8m0_zero() { assert_eq!(e8m0_to_f32(0), 0.0); }

    #[test]
    fn e8m0_one() { assert_eq!(e8m0_to_f32(127), 1.0); }

    #[test]
    fn e8m0_powers_of_two() {
        assert_eq!(e8m0_to_f32(128), 2.0);
        assert_eq!(e8m0_to_f32(126), 0.5);
        assert_eq!(e8m0_to_f32(129), 4.0);
        assert_eq!(e8m0_to_f32(125), 0.25);
    }

    #[test]
    fn e8m0_nan() { assert!(e8m0_to_f32(255).is_nan()); }

    #[test]
    fn table_positive() {
        assert_eq!(MXFP4_TABLE[0], 0.0);
        assert_eq!(MXFP4_TABLE[2], 1.0);
        assert_eq!(MXFP4_TABLE[7], 6.0);
    }

    #[test]
    fn table_negative() {
        assert_eq!(MXFP4_TABLE[10], -1.0);
        assert_eq!(MXFP4_TABLE[15], -6.0);
    }

    #[test]
    fn dequant_all_ones() {
        let blocks = vec![0x22u8; 16]; // lo=2(1.0), hi=2(1.0)
        let scales = vec![127u8]; // scale=1.0
        let result = dequantize_expert(&blocks, &scales, 1, 1).unwrap();
        assert_eq!(result.len(), 32);
        for &v in &result { assert!((v - 1.0).abs() < 1e-6); }
    }

    #[test]
    fn dequant_with_scale() {
        let blocks = vec![0x22u8; 16];
        let scales = vec![128u8]; // scale=2.0
        let result = dequantize_expert(&blocks, &scales, 1, 1).unwrap();
        for &v in &result { assert!((v - 2.0).abs() < 1e-6); }
    }

    #[test]
    fn dequant_negative() {
        let blocks = vec![0xAAu8; 16]; // lo=10(-1.0), hi=10(-1.0)
        let scales = vec![127u8];
        let result = dequantize_expert(&blocks, &scales, 1, 1).unwrap();
        for &v in &result { assert!((v - (-1.0)).abs() < 1e-6); }
    }

    #[test]
    fn dequant_zero_scale() {
        let blocks = vec![0xFFu8; 16];
        let scales = vec![0u8];
        let result = dequantize_expert(&blocks, &scales, 1, 1).unwrap();
        for &v in &result { assert_eq!(v, 0.0); }
    }

    #[test]
    fn dequant_mixed_nibbles() {
        let blocks = vec![0x37u8; 16]; // lo=7(6.0), hi=3(1.5)
        let scales = vec![127u8];
        let result = dequantize_expert(&blocks, &scales, 1, 1).unwrap();
        assert!((result[0] - 6.0).abs() < 1e-6);
        assert!((result[1] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn dequant_two_groups() {
        let blocks = vec![0x22u8; 32]; // 2 groups
        let scales = vec![127u8, 128u8]; // [1.0, 2.0]
        let result = dequantize_expert(&blocks, &scales, 1, 2).unwrap();
        assert_eq!(result.len(), 64);
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[32] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn dequant_two_experts() {
        let blocks = vec![0x22u8; 32];
        let scales = vec![127u8, 128u8];
        let results = dequantize_all_experts(&blocks, &scales, 2, 1, 1).unwrap();
        assert_eq!(results.len(), 2);
        assert!((results[0][0] - 1.0).abs() < 1e-6);
        assert!((results[1][0] - 2.0).abs() < 1e-6);
    }

    // ── Bounds-check rejection ──

    #[test]
    fn dequant_expert_rejects_short_blocks() {
        // Need 16 bytes; give 8.
        match dequantize_expert(&[0u8; 8], &[127], 1, 1) {
            Err(ModelError::Parse(msg)) => {
                assert!(msg.contains("blocks too short"), "got: {msg}");
            }
            other => panic!("expected Parse error, got {other:?}"),
        }
    }

    #[test]
    fn dequant_expert_rejects_short_scales() {
        // Need 2 scales for (out_features=1, groups=2); give 1.
        match dequantize_expert(&[0u8; 32], &[127], 1, 2) {
            Err(ModelError::Parse(msg)) => {
                assert!(msg.contains("scales too short"), "got: {msg}");
            }
            other => panic!("expected Parse error, got {other:?}"),
        }
    }

    #[test]
    fn dequant_all_experts_rejects_short_blocks() {
        // 2 experts × 16 bytes = 32; give 20.
        match dequantize_all_experts(&[0u8; 20], &[127, 128], 2, 1, 1) {
            Err(ModelError::Parse(msg)) => {
                assert!(msg.contains("blocks_data too short"), "got: {msg}");
            }
            other => panic!("expected Parse error, got {other:?}"),
        }
    }

    #[test]
    fn dequant_all_experts_rejects_short_scales() {
        match dequantize_all_experts(&[0u8; 32], &[127], 2, 1, 1) {
            Err(ModelError::Parse(msg)) => {
                assert!(msg.contains("scales_data too short"), "got: {msg}");
            }
            other => panic!("expected Parse error, got {other:?}"),
        }
    }

    #[test]
    fn dequant_zero_experts_ok() {
        let results = dequantize_all_experts(&[], &[], 0, 1, 1).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn dequant_all_experts_slices_scales_per_expert() {
        // Regression: each expert gets its own scale slice. Give expert 0 a
        // zero scale (all outputs 0) and expert 1 a 2.0 scale (nibble 2 → 2.0).
        let blocks = vec![0x22u8; 32]; // 2 experts × 16 bytes, nibbles = 2 = 1.0
        let scales = vec![0u8, 128u8]; // exp0 scale=0 → 0.0, exp1 scale=2 → 2.0
        let results = dequantize_all_experts(&blocks, &scales, 2, 1, 1).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results[0].iter().all(|&v| v == 0.0));
        assert!(results[1].iter().all(|&v| (v - 2.0).abs() < 1e-6));
    }
}
