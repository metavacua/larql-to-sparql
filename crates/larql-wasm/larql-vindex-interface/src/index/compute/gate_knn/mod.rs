//! Gate KNN dispatch — brute-force, batched, and HNSW. Storage-side
//! resolution (mmap fast path, decode caches, LRU bookkeeping) lives
//! in `crate::index::storage::gate_store`; this module only orchestrates
//! the dot-product → top-K compute.
//!
//! Split layout (M6 cleanup, 2026-05-01):
//! - `dispatch.rs`        — top-level KNN entry points (gate_knn,
//!                          gate_knn_expert, walk, gate_knn_batch,
//!                          gate_knn_adaptive, gate_knn_q4) + the
//!                          batched matmul gate_walk
//! - `scores_batch.rs`    — full-batch BLAS / GPU matmul paths
//!                          feeding the dispatch entry points
//!                          (gate_scores_batch / gate_scores_2d_*)
//! - `hnsw_lifecycle.rs`  — HNSW enable/disable, lazy + eager build,
//!                          per-layer + per-(layer,expert) caches,
//!                          and the HNSW-backed knn variants
//!
//! The `top_k_from_scores` impl method and the `top_k_by_abs` free
//! function live here so every submodule can share them without
//! cross-importing siblings.

use ndarray::Array1;

use crate::index::core::VectorIndex;

mod dispatch;
mod hnsw_lifecycle;
mod scores_batch;

/// Shared `top_k_from_scores` — every submodule routes through this.
impl VectorIndex {
    /// Pick the K scores with the largest absolute value out of N. Single
    /// scan with a min-heap of capacity K; allocation is O(K), not O(N).
    /// On Gemma 4B (N=10240, K=10, 34-layer walk) this is ~5.4 MB less
    /// allocation per token vs the previous Vec+select_nth approach. Mmap
    /// stays untouched — only the score-extract heap shrinks.
    pub(crate) fn top_k_from_scores(scores: &Array1<f32>, top_k: usize) -> Vec<(usize, f32)> {
        top_k_by_abs(scores.iter().copied(), top_k)
    }
}

/// Walk an iterator of f32 scores once, keep the K with largest |value|,
/// return them sorted by |value| descending (matching the prior Vec+select
/// behaviour at the call sites). Does not allocate beyond a `BinaryHeap`
/// of capacity K — for K=10 that's 240 B regardless of input length.
///
/// Panics on NaN inputs to preserve the previous `partial_cmp(...).unwrap()`
/// contract — gate scores from BLAS gemv are NaN-free as long as the
/// inputs are.
pub(super) fn top_k_by_abs<I>(scores: I, top_k: usize) -> Vec<(usize, f32)>
where
    I: IntoIterator<Item = f32>,
{
    use std::cmp::Ordering;
    use std::collections::BinaryHeap;

    if top_k == 0 {
        return Vec::new();
    }

    /// Wrapper that orders by `|val|`. Inverted `Ord` so `BinaryHeap`
    /// (max-heap by default) acts as a *min-heap on |val|*: `peek()`
    /// gives the smallest |val| currently in the heap, which is the
    /// candidate to evict when a bigger |val| arrives.
    #[derive(Copy, Clone)]
    struct AbsScore {
        idx: usize,
        val: f32,
    }
    impl PartialEq for AbsScore {
        fn eq(&self, other: &Self) -> bool {
            self.val.abs() == other.val.abs()
        }
    }
    impl Eq for AbsScore {}
    impl PartialOrd for AbsScore {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for AbsScore {
        fn cmp(&self, other: &Self) -> Ordering {
            // Reversed: smaller |val| ranks higher → max-heap pops it first.
            other.val.abs().partial_cmp(&self.val.abs()).unwrap()
        }
    }

    // Cap the heap pre-allocation so callers passing `usize::MAX`
    // (the unlimited walk path) don't request a TB-sized allocation.
    // The loop pushes at most `min(top_k, scores.len())` entries; we
    // size to the practical max (any vindex with > 1M features per
    // layer would need a different data structure anyway).
    const HEAP_CAP_LIMIT: usize = 1 << 20;
    let cap = top_k.min(HEAP_CAP_LIMIT);
    let mut heap: BinaryHeap<AbsScore> = BinaryHeap::with_capacity(cap);
    for (i, v) in scores.into_iter().enumerate() {
        if heap.len() < top_k {
            heap.push(AbsScore { idx: i, val: v });
        } else if v.abs() > heap.peek().unwrap().val.abs() {
            heap.pop();
            heap.push(AbsScore { idx: i, val: v });
        }
    }

    let mut out: Vec<(usize, f32)> = heap.into_iter().map(|a| (a.idx, a.val)).collect();
    out.sort_unstable_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
    out
}

#[cfg(test)]
mod tests {
    use super::top_k_by_abs;
    use ndarray::Array1;

    // ── Per-(layer, expert) HNSW unit tests ──────────────────────────────
    //
    // Construct a small synthetic VectorIndex with gate vectors laid out
    // as [features, hidden]. We split features into two "experts":
    // expert 0 holds features [0, 4), expert 1 holds [4, 8).  Test that
    // gate_knn_expert respects the expert range, and that the HNSW-enabled
    // path returns the same top hit as brute-force on a designed input.
    //
    // The HNSW path uses random projection + approximate graph search so
    // the EXACT top-K can differ from brute. We pick test inputs where the
    // top hit is far from the runners-up, so even approximate search lands
    // it correctly. This catches index-mapping bugs (slice→global offset),
    // empty-slice handling, and the HNSW toggle dispatch — without
    // promising graph-search recall guarantees the tests can't enforce.

    use crate::index::VectorIndex;
    use ndarray::Array2;
    use std::sync::atomic::Ordering;

    /// Build a 2-layer VectorIndex with 8 features × 4 hidden where
    /// `feature_i = e_(i mod 4)` (one-hot among the 4 hidden dims).  A
    /// query equal to `e_j` then dot-products to 1.0 exactly with
    /// features `j, j+4` and 0.0 with the others — predictable top-K.
    fn synth_index() -> VectorIndex {
        let num_layers = 2;
        let hidden = 4;
        let mut gate0 = Array2::<f32>::zeros((8, hidden));
        for f in 0..8 {
            gate0[[f, f % 4]] = 1.0;
        }
        let gate1 = gate0.clone();
        let gate = vec![Some(gate0), Some(gate1)];
        let down = vec![None, None];
        VectorIndex::new(gate, down, num_layers, hidden)
    }

    #[test]
    fn gate_knn_expert_brute_force_respects_range() {
        let v = synth_index();
        // Query e_2 → matches feature 2 (in expert 0) and feature 6 (in
        // expert 1) at score 1.0.  Restricting to expert 0 (feat 0..4)
        // should return feature 2 only at full score; feature 6 must NOT
        // appear.
        let q = Array1::from_vec(vec![0.0, 0.0, 1.0, 0.0]);
        let hits = v.gate_knn_expert(0, &q, 0, 4, 2);
        assert_eq!(hits[0].0, 2, "top hit must be feature 2");
        assert!((hits[0].1 - 1.0).abs() < 1e-5);
        for (idx, _) in &hits {
            assert!(*idx < 4, "feature {idx} leaked from expert 1");
        }
    }

    #[test]
    fn gate_knn_expert_hnsw_top_hit_matches_brute() {
        let v = synth_index();
        v.gate.hnsw_enabled.store(true, Ordering::Relaxed);
        // Same query as above; HNSW must agree on the top hit (the only
        // feature with perfect score 1.0 inside the expert-0 range).
        let q = Array1::from_vec(vec![0.0, 0.0, 1.0, 0.0]);
        let hits = v.gate_knn_expert(0, &q, 0, 4, 1);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, 2);
        assert!((hits[0].1 - 1.0).abs() < 1e-5);
        // Cache should now hold the unit index.
        let cache = v.gate.hnsw_unit_cache.lock().unwrap();
        assert!(
            cache.contains_key(&(0, 0)),
            "hnsw_unit_cache must contain (layer=0, feat_start=0)"
        );
    }

    #[test]
    fn gate_knn_expert_hnsw_offsets_to_global_indices() {
        let v = synth_index();
        v.gate.hnsw_enabled.store(true, Ordering::Relaxed);
        // Search expert 1 (features 4..8); query e_2 hits feature 6.
        // The HNSW unit indexes 0..4 internally; we must offset back to
        // global feature 6, not 2.
        let q = Array1::from_vec(vec![0.0, 0.0, 1.0, 0.0]);
        let hits = v.gate_knn_expert(0, &q, 4, 8, 1);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, 6, "expected global feature 6, got {}", hits[0].0);
    }

    #[test]
    fn warmup_hnsw_units_builds_requested_set() {
        let v = synth_index();
        let units = vec![(0, 0, 4), (0, 4, 8), (1, 0, 4), (1, 4, 8)];
        let n = v.warmup_hnsw_units(&units);
        assert_eq!(n, 4);
        let cache = v.gate.hnsw_unit_cache.lock().unwrap();
        for &(l, fs, _) in &units {
            assert!(
                cache.contains_key(&(l, fs)),
                "missing unit ({l}, {fs}) after warmup"
            );
        }
        // Idempotent: second call should build nothing new.
        drop(cache);
        let n2 = v.warmup_hnsw_units(&units);
        assert_eq!(n2, 0);
    }

    #[test]
    fn gate_knn_expert_hnsw_falls_back_when_slice_empty() {
        let v = synth_index();
        v.gate.hnsw_enabled.store(true, Ordering::Relaxed);
        // feat_start == feat_end → empty range → must return empty without
        // panicking on the HNSW path or installing a bogus cache entry.
        let q = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
        let hits = v.gate_knn_expert(0, &q, 4, 4, 1);
        assert!(hits.is_empty());
        let cache = v.gate.hnsw_unit_cache.lock().unwrap();
        assert!(!cache.contains_key(&(0, 4)));
    }

    #[test]
    fn top_k_by_abs_basic_ordering() {
        let scores: Vec<f32> = vec![0.1, -0.9, 0.5, 0.3];
        let result = top_k_by_abs(scores, 2);
        assert_eq!(result.len(), 2);
        // Top-2 by |val|: index 1 (|-0.9|=0.9) then index 2 (|0.5|=0.5).
        assert_eq!(result[0].0, 1);
        assert!((result[0].1 - (-0.9)).abs() < 1e-6);
        assert_eq!(result[1].0, 2);
    }

    #[test]
    fn top_k_by_abs_negative_values_selected_by_magnitude() {
        let scores: Vec<f32> = vec![1.0, -2.0, 0.5];
        let result = top_k_by_abs(scores, 1);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, 1); // |-2.0| is largest
    }

    #[test]
    fn top_k_by_abs_k_larger_than_input() {
        let scores: Vec<f32> = vec![1.0, 2.0];
        let result = top_k_by_abs(scores, 10);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn top_k_by_abs_k_zero_returns_empty() {
        let scores: Vec<f32> = vec![1.0, 2.0, 3.0];
        let result = top_k_by_abs(scores, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn top_k_by_abs_empty_input_returns_empty() {
        let result = top_k_by_abs(std::iter::empty::<f32>(), 5);
        assert!(result.is_empty());
    }

    #[test]
    fn top_k_by_abs_result_sorted_descending() {
        let scores: Vec<f32> = vec![0.3, 0.1, 0.9, 0.5, 0.7];
        let result = top_k_by_abs(scores, 3);
        assert_eq!(result.len(), 3);
        for w in result.windows(2) {
            assert!(w[0].1.abs() >= w[1].1.abs(), "not sorted: {:?}", result);
        }
    }

    #[test]
    fn top_k_from_scores_via_array1() {
        use crate::index::VectorIndex;
        let arr = Array1::from_vec(vec![0.1f32, -0.9, 0.5]);
        let result = VectorIndex::top_k_from_scores(&arr, 2);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].0, 1); // |-0.9| largest
    }
}
