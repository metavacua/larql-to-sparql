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
pub const MXFP4_TABLE: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Elements sharing one E8M0 scale — the microscaling group.
pub const MXFP4_GROUP_ELEMS: usize = 32;

/// Packed bytes per group: two 4-bit values per byte.
pub const MXFP4_GROUP_BYTES: usize = MXFP4_GROUP_ELEMS / 2;

/// Convert e8m0 scale byte to float multiplier.
/// e8m0 = pure exponent, no mantissa: value = 2^(exponent - 127)
pub fn e8m0_to_f32(byte: u8) -> f32 {
    if byte == 0 {
        return 0.0;
    }
    if byte == 255 {
        return f32::NAN;
    }
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
        ModelError::Parse(format!(
            "MXFP4: total blocks overflow ({num_experts} experts)"
        ))
    })?;
    let need_scales = num_experts.checked_mul(scales_per_expert).ok_or_else(|| {
        ModelError::Parse(format!(
            "MXFP4: total scales overflow ({num_experts} experts)"
        ))
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

/// Per-expert weight matrix: one inner `Vec<f32>` per expert, row-major.
pub type ExpertWeights = Vec<Vec<f32>>;

/// Number of projections fused into one `gate_up` tensor.
///
/// Public because it is also the *row stride* of the interleaved
/// arrangement — a consumer walking one half's rows advances by this much —
/// and a second `2` spelled at the walking site would be a second statement
/// of the convention [`FusedHalf`] owns.
pub const FUSED_HALVES: usize = 2;

/// Which projection to lift out of a fused `gate_up` tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusedHalf {
    /// The gate projection (`w1`) — the **even** fused rows.
    Gate,
    /// The up projection (`w3`) — the **odd** fused rows.
    Up,
}

impl FusedHalf {
    /// Fused-row index holding row `row` of this projection.
    ///
    /// Gate and up are **interleaved**, not contiguous — see
    /// [`deinterleave_fused_half`] for why.
    #[inline]
    pub const fn fused_row(self, row: usize) -> usize {
        match self {
            Self::Gate => FUSED_HALVES * row,
            Self::Up => FUSED_HALVES * row + 1,
        }
    }
}

/// Lift one projection out of a dequantised fused `gate_up` expert matrix.
///
/// GPT-OSS fuses gate and up into one tensor, **interleaved along the output
/// axis** — gate occupies the even rows and up the odd ones. That is not a
/// guess; it follows the reference chain in
/// `transformers/integrations/mxfp4.py` and
/// `transformers/models/gpt_oss/modeling_gpt_oss.py`:
///
/// ```text
/// blocks (E, 2*I, G, 16)
///   -> dequant             (E, 2*I, H)      row = the on-disk output row
///   -> .transpose(1, 2)    (E, H, 2*I)      on-disk row becomes the LAST axis
///   -> _apply_gate: gate = x[..., 0::2], up = x[..., 1::2]
/// ```
///
/// so `gate_up[..., 0::2]` selects even on-disk rows and `[..., 1::2]` odd ones.
///
/// Taking the first and second halves instead silently yields two 50/50
/// mixtures of the real gate and up rows. That is what larql did until
/// 2026-07-30, and it is near-invisible without a reference to diff against:
/// both halves come out with matching summary statistics, because both are the
/// same mixture. See `docs/k3-funnel.md` §4.7.
///
/// `data` is row-major `[out_features, in_features]`; the result is row-major
/// `[out_features / 2, in_features]`.
pub fn deinterleave_fused_half(
    data: &[f32],
    out_features: usize,
    in_features: usize,
    half: FusedHalf,
) -> Vec<f32> {
    let rows = out_features / FUSED_HALVES;
    let mut out = Vec::with_capacity(rows * in_features);
    for row in 0..rows {
        let start = half.fused_row(row) * in_features;
        out.extend_from_slice(&data[start..start + in_features]);
    }
    out
}

/// Dequantize and split a GPT-OSS fused gate_up packed tensor into separate
/// gate (w1) and up (w3) per-expert matrices.
///
/// GPT-OSS stores gate and up projections fused into a single MXFP4 tensor of
/// shape `[num_experts, 2*intermediate, groups, 16]`, **interleaved** along the
/// output axis. The de-interleaving convention is owned by
/// [`deinterleave_fused_half`].
///
/// Returns `(gate_experts, up_experts)` each an `ExpertWeights` of length
/// `num_experts`, where each inner `Vec` holds one expert's weight matrix
/// in row-major order with shape `[out_features/2, groups*32]`.
pub fn split_gate_up_experts(
    blocks: &[u8],
    scales: &[u8],
    num_experts: usize,
    out_features: usize,
    groups: usize,
) -> Result<(ExpertWeights, ExpertWeights), ModelError> {
    let expert_data = dequantize_all_experts(blocks, scales, num_experts, out_features, groups)?;
    let in_features = groups * 32;
    let mut gates = Vec::with_capacity(num_experts);
    let mut ups = Vec::with_capacity(num_experts);
    for data in expert_data {
        gates.push(deinterleave_fused_half(
            &data,
            out_features,
            in_features,
            FusedHalf::Gate,
        ));
        ups.push(deinterleave_fused_half(
            &data,
            out_features,
            in_features,
            FusedHalf::Up,
        ));
    }
    Ok((gates, ups))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e8m0_zero() {
        assert_eq!(e8m0_to_f32(0), 0.0);
    }

    #[test]
    fn e8m0_one() {
        assert_eq!(e8m0_to_f32(127), 1.0);
    }

    #[test]
    fn e8m0_powers_of_two() {
        assert_eq!(e8m0_to_f32(128), 2.0);
        assert_eq!(e8m0_to_f32(126), 0.5);
        assert_eq!(e8m0_to_f32(129), 4.0);
        assert_eq!(e8m0_to_f32(125), 0.25);
    }

    #[test]
    fn e8m0_nan() {
        assert!(e8m0_to_f32(255).is_nan());
    }

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
        for &v in &result {
            assert!((v - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn dequant_with_scale() {
        let blocks = vec![0x22u8; 16];
        let scales = vec![128u8]; // scale=2.0
        let result = dequantize_expert(&blocks, &scales, 1, 1).unwrap();
        for &v in &result {
            assert!((v - 2.0).abs() < 1e-6);
        }
    }

    #[test]
    fn dequant_negative() {
        let blocks = vec![0xAAu8; 16]; // lo=10(-1.0), hi=10(-1.0)
        let scales = vec![127u8];
        let result = dequantize_expert(&blocks, &scales, 1, 1).unwrap();
        for &v in &result {
            assert!((v - (-1.0)).abs() < 1e-6);
        }
    }

    #[test]
    fn dequant_zero_scale() {
        let blocks = vec![0xFFu8; 16];
        let scales = vec![0u8];
        let result = dequantize_expert(&blocks, &scales, 1, 1).unwrap();
        for &v in &result {
            assert_eq!(v, 0.0);
        }
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
    fn dequant_expert_rejects_blocks_overflow() {
        // groups=2 keeps `groups * 32 = 64` from panicking at L42, then
        // out_features × groups overflows the checked_mul chain.
        let big = usize::MAX / 2 + 1;
        match dequantize_expert(&[], &[], big, 2) {
            Err(ModelError::Parse(msg)) => {
                assert!(msg.contains("block count overflow"), "got: {msg}");
            }
            other => panic!("expected Parse error, got {other:?}"),
        }
    }

    #[test]
    fn dequant_all_experts_rejects_blocks_per_expert_overflow() {
        // Same pattern — keep groups small so `groups * 32` is fine, blow
        // up `out_features.checked_mul(groups)` instead.
        let big = usize::MAX / 2 + 1;
        match dequantize_all_experts(&[], &[], 1, big, 2) {
            Err(ModelError::Parse(msg)) => {
                assert!(msg.contains("blocks_per_expert overflow"), "got: {msg}");
            }
            other => panic!("expected Parse error, got {other:?}"),
        }
    }

    #[test]
    fn dequant_all_experts_rejects_total_overflow_on_huge_num_experts() {
        // blocks_per_expert = 16, scales_per_expert = 1. num_experts =
        // usize::MAX overflows the `num_experts × blocks_per_expert` check.
        match dequantize_all_experts(&[], &[], usize::MAX, 1, 1) {
            Err(ModelError::Parse(msg)) => {
                assert!(msg.contains("total blocks overflow"), "got: {msg}");
            }
            other => panic!("expected Parse error, got {other:?}"),
        }
    }

    // ── fused gate/up de-interleaving ──
    //
    // The convention under test is that gate occupies the EVEN fused rows and
    // up the ODD ones (`deinterleave_fused_half`). Note that a fixture with
    // `out_features = 2` cannot test it: row 0 is simultaneously "the first
    // half" and "the even rows", so both the interleaved and the (wrong)
    // contiguous-halves reading agree. That is precisely why the original
    // tests here passed against a broken split for months — they pinned the
    // shape, not the convention. Anything asserting the convention needs
    // `out_features >= 4`. See `docs/k3-funnel.md` §4.7.

    /// Four distinct fused rows, so gate-vs-up row selection is observable.
    /// Nibbles are all `2` (= 1.0), so each row's value is its own scale:
    /// rows = [1.0, 2.0, 4.0, 8.0].
    fn four_row_fused_expert() -> (Vec<u8>, Vec<u8>) {
        let blocks = vec![0x22u8; 4 * 16];
        let scales = vec![127u8, 128u8, 129u8, 130u8]; // 1.0, 2.0, 4.0, 8.0
        (blocks, scales)
    }

    #[test]
    fn fused_row_indices_interleave() {
        assert_eq!(FusedHalf::Gate.fused_row(0), 0);
        assert_eq!(FusedHalf::Gate.fused_row(1), 2);
        assert_eq!(FusedHalf::Gate.fused_row(7), 14);
        assert_eq!(FusedHalf::Up.fused_row(0), 1);
        assert_eq!(FusedHalf::Up.fused_row(1), 3);
        assert_eq!(FusedHalf::Up.fused_row(7), 15);
    }

    #[test]
    fn deinterleave_takes_even_rows_for_gate_and_odd_for_up() {
        // Two rows of two elements each, values = row index.
        let data = vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0];
        let gate = deinterleave_fused_half(&data, 4, 2, FusedHalf::Gate);
        let up = deinterleave_fused_half(&data, 4, 2, FusedHalf::Up);
        assert_eq!(gate, vec![0.0, 0.0, 2.0, 2.0], "gate must be the even rows");
        assert_eq!(up, vec![1.0, 1.0, 3.0, 3.0], "up must be the odd rows");
    }

    /// The regression that the two-row fixtures could not express: under the
    /// old contiguous-halves split this returns gate = [1.0, 2.0] and
    /// up = [4.0, 8.0], which is a 50/50 mixture of the real projections.
    #[test]
    fn split_gate_up_is_interleaved_not_contiguous_halves() {
        let (blocks, scales) = four_row_fused_expert();
        let (gates, ups) = split_gate_up_experts(&blocks, &scales, 1, 4, 1).unwrap();

        let gate_rows: Vec<f32> = gates[0].chunks(32).map(|r| r[0]).collect();
        let up_rows: Vec<f32> = ups[0].chunks(32).map(|r| r[0]).collect();

        assert_eq!(
            gate_rows,
            vec![1.0, 4.0],
            "gate must come from fused rows 0 and 2, not 0 and 1"
        );
        assert_eq!(
            up_rows,
            vec![2.0, 8.0],
            "up must come from fused rows 1 and 3, not 2 and 3"
        );
    }

    #[test]
    fn split_gate_up_shapes_and_minimal_two_row_case() {
        // 1 expert, out_features=2 (one row each), 1 group → 32 elements.
        // Shape-only: this fixture cannot distinguish the two conventions.
        let blocks = vec![0x22u8; 32];
        let scales = vec![127u8, 128u8]; // [1.0, 2.0]
        let (gates, ups) = split_gate_up_experts(&blocks, &scales, 1, 2, 1).unwrap();
        assert_eq!(gates.len(), 1);
        assert_eq!(ups.len(), 1);
        assert_eq!(gates[0].len(), 32);
        assert_eq!(ups[0].len(), 32);
        assert!(gates[0].iter().all(|&v| (v - 1.0).abs() < 1e-6));
        assert!(ups[0].iter().all(|&v| (v - 2.0).abs() < 1e-6));
    }

    #[test]
    fn split_gate_up_two_experts() {
        // 2 experts, out_features=2, 1 group each.
        // Expert 0 scale=1.0, expert 1 scale=2.0 (both use nibble 2 = 1.0).
        let blocks = vec![0x22u8; 64]; // 2 experts × 2 groups × 16 bytes
        let scales = vec![127u8, 127u8, 128u8, 128u8]; // e0:[1.0,1.0], e1:[2.0,2.0]
        let (gates, ups) = split_gate_up_experts(&blocks, &scales, 2, 2, 1).unwrap();
        assert_eq!(gates.len(), 2);
        assert!(gates[0].iter().all(|&v| (v - 1.0).abs() < 1e-6));
        assert!(gates[1].iter().all(|&v| (v - 2.0).abs() < 1e-6));
        assert!(ups[0].iter().all(|&v| (v - 1.0).abs() < 1e-6));
        assert!(ups[1].iter().all(|&v| (v - 2.0).abs() < 1e-6));
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
