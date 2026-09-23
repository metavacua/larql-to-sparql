//! Per-expert gate summarisation — the full-matrix vs top-K SVD decision.
//!
//! Shared by every MoE expert layout so the policy is stated once. Before
//! 2026-07-28 the packed-MXFP4 layout had no summary path at all, so
//! `--summary-features-per-expert` silently did nothing on exactly the
//! format that needs it most: MXFP4 stores weights at 4 bits, and writing
//! the full gate re-inflates them to the vindex dtype at full expert
//! multiplicity. On a large bank that inverts the size relationship with
//! the source model (896 experts × 92 layers ≈ 1.8 TB of gate against a
//! ~1.4 TB checkpoint).

use std::borrow::Cow;

use ndarray::ArrayView2;

/// Power iterations for the randomised range finder in the top-K SVD.
/// Four is enough for the spectral decay of a gate matrix; more costs
/// linearly and moves the singular vectors negligibly.
pub(super) const SVD_POWER_ITERATIONS: usize = 4;

/// Bit shift placing the layer index in the high half of a per-expert
/// seed, leaving the low half for the expert index.
const SEED_LAYER_SHIFT: u32 = 32;

/// Deterministic per-expert RNG seed.
///
/// Re-runs are bit-identical (the seed is a pure function of position)
/// while remaining uncorrelated between experts, so one expert's random
/// projection tells you nothing about its neighbour's.
pub(super) fn expert_seed(layer: usize, expert: usize) -> u64 {
    ((layer as u64) << SEED_LAYER_SHIFT) | (expert as u64)
}

/// Whether an expert's gate should be summarised rather than written whole.
///
/// `summary_k == 0` disables summarisation entirely. A `summary_k` at or
/// above the row count would be a no-op that costs an SVD, so it falls
/// through to the full path.
pub(super) fn should_summarise(summary_k: usize, full_rows: usize) -> bool {
    summary_k > 0 && full_rows > summary_k
}

/// One expert's gate rows, either verbatim or reduced to the top-K right
/// singular vectors, together with the row count actually produced.
///
/// The full path borrows when the input is contiguous, so the common
/// no-summary case stays copy-free.
pub(super) fn summarise_expert_gate<'a>(
    gate: ArrayView2<'a, f32>,
    summary_k: usize,
    seed: u64,
) -> (Cow<'a, [f32]>, usize) {
    let full_rows = gate.nrows();

    if !should_summarise(summary_k, full_rows) {
        return match gate.to_slice() {
            Some(contiguous) => (Cow::Borrowed(contiguous), full_rows),
            None => (Cow::Owned(gate.iter().copied().collect()), full_rows),
        };
    }

    let vt = crate::extract::moe_svd::top_k_right_singular_vectors(
        gate,
        summary_k,
        SVD_POWER_ITERATIONS,
        seed,
    );
    let data = vt
        .as_slice()
        .expect("top_k_right_singular_vectors returns a contiguous array")
        .to_vec();
    (Cow::Owned(data), summary_k)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn gate(rows: usize, cols: usize) -> Array2<f32> {
        Array2::from_shape_fn((rows, cols), |(r, c)| (r * cols + c) as f32 * 0.01)
    }

    // ── expert_seed ─────────────────────────────────────────────────────

    #[test]
    fn seed_is_deterministic() {
        assert_eq!(expert_seed(3, 7), expert_seed(3, 7));
    }

    #[test]
    fn seed_separates_layer_from_expert() {
        // Layer in the high half, expert in the low half — so (1, 0) and
        // (0, 1) must not collide.
        assert_ne!(expert_seed(1, 0), expert_seed(0, 1));
        assert_eq!(expert_seed(0, 5), 5);
        assert_eq!(expert_seed(1, 0), 1 << SEED_LAYER_SHIFT);
    }

    #[test]
    fn seed_distinguishes_adjacent_experts() {
        assert_ne!(expert_seed(4, 8), expert_seed(4, 9));
    }

    // ── should_summarise ────────────────────────────────────────────────

    #[test]
    fn zero_k_disables_summarisation() {
        assert!(!should_summarise(0, 4096));
    }

    #[test]
    fn k_below_row_count_summarises() {
        assert!(should_summarise(64, 2880));
    }

    #[test]
    fn k_at_or_above_row_count_is_a_no_op() {
        // An SVD that returns as many vectors as it consumed buys nothing.
        assert!(!should_summarise(2880, 2880));
        assert!(!should_summarise(4096, 2880));
    }

    // ── summarise_expert_gate ───────────────────────────────────────────

    #[test]
    fn full_path_returns_every_row_unchanged() {
        let g = gate(8, 5);
        let (data, rows) = summarise_expert_gate(g.view(), 0, expert_seed(0, 0));
        assert_eq!(rows, 8);
        assert_eq!(data.len(), 8 * 5);
        assert_eq!(&*data, g.as_slice().unwrap());
    }

    #[test]
    fn full_path_borrows_contiguous_input() {
        let g = gate(8, 5);
        let (data, _) = summarise_expert_gate(g.view(), 0, 0);
        assert!(
            matches!(data, Cow::Borrowed(_)),
            "contiguous input should not copy"
        );
    }

    #[test]
    fn full_path_copies_non_contiguous_input() {
        let g = gate(8, 6);
        // Every other column — a strided view with no contiguous slice.
        let strided = g.slice(ndarray::s![.., ..;2]);
        let (data, rows) = summarise_expert_gate(strided, 0, 0);
        assert!(matches!(data, Cow::Owned(_)));
        assert_eq!(rows, 8);
        assert_eq!(data.len(), 8 * 3);
    }

    #[test]
    fn summary_path_reduces_row_count() {
        let g = gate(64, 16);
        let (data, rows) = summarise_expert_gate(g.view(), 4, expert_seed(1, 2));
        assert_eq!(rows, 4);
        assert_eq!(data.len(), 4 * 16, "K rows of the input width");
    }

    #[test]
    fn summary_path_is_reproducible_for_the_same_seed() {
        let g = gate(64, 16);
        let (a, _) = summarise_expert_gate(g.view(), 4, expert_seed(1, 2));
        let (b, _) = summarise_expert_gate(g.view(), 4, expert_seed(1, 2));
        assert_eq!(&*a, &*b, "same seed must reproduce bit-identically");
    }

    #[test]
    fn summary_path_output_is_finite() {
        let g = gate(64, 16);
        let (data, _) = summarise_expert_gate(g.view(), 4, expert_seed(0, 0));
        assert!(
            data.iter().all(|v| v.is_finite()),
            "SVD produced non-finite values"
        );
    }

    #[test]
    fn summary_rows_are_unit_norm() {
        // Right singular vectors are orthonormal, so each row should have
        // unit length — a cheap check that we wrote V^T and not something
        // unnormalised.
        let g = gate(64, 16);
        let (data, rows) = summarise_expert_gate(g.view(), 4, expert_seed(0, 0));
        for r in 0..rows {
            let norm: f32 = data[r * 16..(r + 1) * 16]
                .iter()
                .map(|v| v * v)
                .sum::<f32>()
                .sqrt();
            assert!((norm - 1.0).abs() < 1e-3, "row {r} norm {norm} is not unit");
        }
    }
}
