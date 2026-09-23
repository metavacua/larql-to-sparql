//! Dispatch thresholds for the walk-FFN kernels — every numeric cutoff
//! that decides WHICH kernel runs lives here, named and documented in
//! one place. All are benchmark-derived heuristics (M3 Max numbers);
//! the execution-planner work (ROADMAP § "Vindex + WalkFFN review
//! (2026-07-30)", item 18) will surface them as config-visible plan
//! reasons instead of buried constants.

/// Minimum hit count for the rayon parallel Q4K-down-cache path in
/// `walk_ffn_sparse`. Below this, per-thread partial buffers plus the
/// serial reduce cost more than the serial scaled-add loop saves.
pub(super) const PARALLEL_DOWN_MIN_HITS: usize = 512;

/// Minimum route-pool size for the gather-contiguous Q4K kernel
/// (`gather_q4k_accumulate`). Below this the gather's byte-copy setup
/// outweighs the scattered per-feature loop's cache misses.
pub(super) const GATHER_MIN_FEATURES: usize = 256;

/// Requested-K density (as a fraction of the layer's feature count) at
/// or above which the sparse walk is rewritten to the full-K gemv fast
/// path — computing ALL features densely, not the requested top-K.
/// Kept as an exact integer ratio (4/5 = 80%).
///
/// This rewrite changes numerics relative to a true top-K walk, so K
/// sweeps above this density are silently dense unless
/// `WalkFfnConfig::force_walk` is set — flagged by the 2026-07-30
/// review (finding M1). The threshold itself predates the review; it
/// exists because the per-feature loop loses to gemm well below 100%
/// density.
pub(super) const FULL_K_DENSITY_NUM: usize = 4;
/// Denominator of [`FULL_K_DENSITY_NUM`].
pub(super) const FULL_K_DENSITY_DEN: usize = 5;
