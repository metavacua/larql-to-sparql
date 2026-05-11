//! Shared walk-path helpers.

use crate::vindex::walk_config::WalkFfnConfig;

/// True when the user asked for full-K (K ≥ feature count) — the signal
/// that we should route the walk through batched gemm rather than a
/// per-feature loop. Treats `usize::MAX` (set by `::dense` / `--k full`)
/// as full-K; also caches the check when top-K happens to exceed the
/// layer's feature count.
#[inline]
pub(super) fn hits_len_ge_intermediate(
    config: &WalkFfnConfig,
    layer: usize,
    intermediate: usize,
) -> bool {
    match config.k_for(layer) {
        Some(k) => k >= (intermediate * 8) / 10,
        None => true,
    }
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
