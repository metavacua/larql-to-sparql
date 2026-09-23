//! Shared walk-path helpers.

use super::thresholds::{FULL_K_DENSITY_DEN, FULL_K_DENSITY_NUM};
use crate::vindex::walk_config::WalkFfnConfig;

/// True when the requested K is dense enough that the walk should be
/// routed through batched gemm rather than a per-feature loop.
/// "Dense enough" is K ≥ [`FULL_K_DENSITY_NUM`]/[`FULL_K_DENSITY_DEN`]
/// of the feature count (80%) — NOT K ≥ feature count: the gemv path
/// computes ALL features, so results in the [80%, 100%) band differ
/// from a true top-K walk (2026-07-30 review, finding M1). `None`
/// per-layer K (set by `::dense` / `--k full`) always routes to gemm.
///
/// When `config.force_walk` is set, returns false unconditionally so
/// the per-position walk runs even at full-K. Used to measure the walk
/// paradigm at faithful K without the dispatch failing over to gemv.
#[inline]
pub(super) fn hits_len_ge_intermediate(
    config: &WalkFfnConfig,
    layer: usize,
    intermediate: usize,
) -> bool {
    if config.force_walk {
        return false;
    }
    match config.k_for(layer) {
        Some(k) => k >= (intermediate * FULL_K_DENSITY_NUM) / FULL_K_DENSITY_DEN,
        None => true,
    }
}

/// Descending comparator for top-K selection weights. **Panics on NaN**
/// — matching the pinned `top_k_by_abs` contract in larql-vindex's
/// gate-KNN path (`index/compute/gate_knn/mod.rs`): a NaN gate/up/down
/// score must never silently scramble the selection order, which is
/// what the previous `partial_cmp(..).unwrap_or(Equal)` sites did
/// (2026-07-30 review, low tier: "NaN contract split").
#[inline]
pub(super) fn selection_weight_cmp_desc(a: f32, b: f32) -> std::cmp::Ordering {
    b.partial_cmp(&a).unwrap_or_else(|| {
        panic!("NaN in selector weight (a={a}, b={b}) — top-K selection requires NaN-free scores")
    })
}

/// Dispatch-trace entry: records which walk path fired for a given
/// `(forward_call, layer)`. Enabled via `WalkFfn::with_dispatch_trace()`.
///
/// Each walk path function calls `ctx.trace_path(layer, "name")` on
/// exit. Tests assert the expected sequence; the Q2 debugging flow
/// uses the trace to identify which path consumed a given vindex.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DispatchEntry {
    pub layer: usize,
    pub path: &'static str,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense_config(layers: usize) -> WalkFfnConfig {
        WalkFfnConfig::dense(layers)
    }

    fn sparse_config(layers: usize, k: usize) -> WalkFfnConfig {
        WalkFfnConfig::sparse(layers, k)
    }

    #[test]
    fn hits_len_full_k_returns_true() {
        // dense() sets k_per_layer to None — should always pick the
        // full-K path.
        let cfg = dense_config(2);
        assert!(hits_len_ge_intermediate(&cfg, 0, 1024));
        assert!(hits_len_ge_intermediate(&cfg, 1, 32));
    }

    #[test]
    fn hits_len_sparse_below_80_percent_returns_false() {
        // k=10, intermediate=100 → threshold 80 → 10 < 80.
        let cfg = sparse_config(1, 10);
        assert!(!hits_len_ge_intermediate(&cfg, 0, 100));
    }

    #[test]
    fn hits_len_sparse_at_80_percent_returns_true() {
        // k=80, intermediate=100 → threshold 80 → 80 >= 80.
        let cfg = sparse_config(1, 80);
        assert!(hits_len_ge_intermediate(&cfg, 0, 100));
    }

    #[test]
    fn hits_len_sparse_above_intermediate_returns_true() {
        // k=200 > intermediate=100 → full-K equivalent.
        let cfg = sparse_config(1, 200);
        assert!(hits_len_ge_intermediate(&cfg, 0, 100));
    }

    #[test]
    fn hits_len_force_walk_short_circuits_full_k() {
        // force_walk: even full-K must take the per-position path.
        let cfg = sparse_config(1, 200).with_force_walk(true);
        assert!(!hits_len_ge_intermediate(&cfg, 0, 100));
        let cfg = dense_config(1).with_force_walk(true);
        assert!(!hits_len_ge_intermediate(&cfg, 0, 100));
    }

    #[test]
    fn selection_weight_cmp_desc_orders_descending() {
        use std::cmp::Ordering;
        assert_eq!(selection_weight_cmp_desc(2.0, 1.0), Ordering::Less);
        assert_eq!(selection_weight_cmp_desc(1.0, 2.0), Ordering::Greater);
        assert_eq!(selection_weight_cmp_desc(1.0, 1.0), Ordering::Equal);
        let mut v = vec![1.0f32, 3.0, 2.0];
        v.sort_by(|a, b| selection_weight_cmp_desc(*a, *b));
        assert_eq!(v, vec![3.0, 2.0, 1.0]);
    }

    /// The NaN contract: panic, never silently reorder — same contract
    /// as `top_k_by_abs` in larql-vindex's gate-KNN path.
    #[test]
    #[should_panic(expected = "NaN in selector weight")]
    fn selection_weight_cmp_desc_panics_on_nan() {
        selection_weight_cmp_desc(f32::NAN, 1.0);
    }

    #[test]
    fn dispatch_entry_equality_is_field_wise() {
        let a = DispatchEntry {
            layer: 3,
            path: "interleaved_q4",
        };
        let b = DispatchEntry {
            layer: 3,
            path: "interleaved_q4",
        };
        let c = DispatchEntry {
            layer: 3,
            path: "sparse",
        };
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
