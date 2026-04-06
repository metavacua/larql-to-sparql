//! MXFP4 dequantization — OpenAI's microscaling 4-bit float format.
//!
//! Used by GPT-OSS models. Each weight is stored as a 4-bit value (two per byte)
//! with shared e8m0 (exponent-only) scales per group of 32 elements.
//!
//! Format:
//!   blocks: [experts, out_features, groups, 16] as U8 (each byte = 2 × 4-bit values)
//!   scales: [experts, out_features, groups] as U8 (e8m0 exponent)

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
pub fn dequantize_expert(
    blocks: &[u8],
    scales: &[u8],
    out_features: usize,
    groups: usize,
) -> Vec<f32> {
    let in_features = groups * 32;
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

    output
}

/// Dequantize all experts from packed MXFP4 tensors.
pub fn dequantize_all_experts(
    blocks_data: &[u8],
    scales_data: &[u8],
    num_experts: usize,
    out_features: usize,
    groups: usize,
) -> Vec<Vec<f32>> {
    let blocks_per_expert = out_features * groups * 16;
    let scales_per_expert = out_features * groups;

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
        let result = dequantize_expert(&blocks, &scales, 1, 1);
        assert_eq!(result.len(), 32);
        for &v in &result { assert!((v - 1.0).abs() < 1e-6); }
    }

    #[test]
    fn dequant_with_scale() {
        let blocks = vec![0x22u8; 16];
        let scales = vec![128u8]; // scale=2.0
        let result = dequantize_expert(&blocks, &scales, 1, 1);
        for &v in &result { assert!((v - 2.0).abs() < 1e-6); }
    }

    #[test]
    fn dequant_negative() {
        let blocks = vec![0xAAu8; 16]; // lo=10(-1.0), hi=10(-1.0)
        let scales = vec![127u8];
        let result = dequantize_expert(&blocks, &scales, 1, 1);
        for &v in &result { assert!((v - (-1.0)).abs() < 1e-6); }
    }

    #[test]
    fn dequant_zero_scale() {
        let blocks = vec![0xFFu8; 16];
        let scales = vec![0u8];
        let result = dequantize_expert(&blocks, &scales, 1, 1);
        for &v in &result { assert_eq!(v, 0.0); }
    }

    #[test]
    fn dequant_mixed_nibbles() {
        let blocks = vec![0x37u8; 16]; // lo=7(6.0), hi=3(1.5)
        let scales = vec![127u8];
        let result = dequantize_expert(&blocks, &scales, 1, 1);
        assert!((result[0] - 6.0).abs() < 1e-6);
        assert!((result[1] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn dequant_two_groups() {
        let blocks = vec![0x22u8; 32]; // 2 groups
        let scales = vec![127u8, 128u8]; // [1.0, 2.0]
        let result = dequantize_expert(&blocks, &scales, 1, 2);
        assert_eq!(result.len(), 64);
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[32] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn dequant_two_experts() {
        let blocks = vec![0x22u8; 32];
        let scales = vec![127u8, 128u8];
        let results = dequantize_all_experts(&blocks, &scales, 2, 1, 1);
        assert_eq!(results.len(), 2);
        assert!((results[0][0] - 1.0).abs() < 1e-6);
        assert!((results[1][0] - 2.0).abs() < 1e-6);
    }
}
