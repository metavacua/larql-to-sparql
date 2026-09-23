//! The planning half of the routing ladder (2026-07-30 review, item
//! 18): every CONDITION check lives here, producing an [`FfnPlan`];
//! the execution half (`forward_ladder` / `execute_plan` in `mod.rs`)
//! only matches on the plan. One condition source, so `plan_for` and
//! `forward` cannot disagree about routing.
//!
//! Purity contract: planning never executes a path, never writes a
//! dispatch-trace entry, and never touches L1 hit/miss statistics
//! ([`WalkFfn::plan_for`] probes the cache via the stats-free
//! [`FfnL1Cache::peek`]). The only cost of planning on the hot
//! forward path is the flag checks the old inline ladder already did,
//! plus one small `skipped`-reasons Vec.

use ndarray::Array2;

use super::helpers::hits_len_ge_intermediate;
use super::observe::Observe;
use super::plan::{FfnPlan, PlanExclusions, PlanReason, PlanRung, SkippedRung, ThresholdCheck};
use super::thresholds::{FULL_K_DENSITY_DEN, FULL_K_DENSITY_NUM};
use super::WalkFfn;
use crate::vindex::l1_cache::FfnL1Cache;

/// Rungs 4–9 — the whole-layer, override-blind paths, in ladder
/// priority order. Shared by the planner, the main executor, and
/// `forward_unpatched_whole_layer` (base+delta's dense base), so the
/// ordering exists in exactly one place.
pub(super) const WHOLE_LAYER_RUNGS: [PlanRung; 6] = [
    PlanRung::KquantNative,
    PlanRung::InterleavedQ4,
    PlanRung::InterleavedF32,
    PlanRung::FullMmap,
    PlanRung::KquantDequant,
    PlanRung::Exact,
];

/// Skip reason for a rung whose static condition held but whose
/// execution returned `None` — the executor re-planned past it.
pub(super) const BECAUSE_DECLINED: &str = "declined at execution (returned None); re-planned";
pub(super) const BECAUSE_HAS_FEATURES: &str = "layer has vindex features (num_features > 0)";
pub(super) const BECAUSE_NO_OVERRIDES: &str = "no feature overrides at this layer";
pub(super) const BECAUSE_L1_OBSERVED: &str =
    "observed/traced call bypasses the L1 read (a cache hit observes nothing)";
pub(super) const BECAUSE_L1_MULTI_POSITION: &str =
    "multi-position input (L1 caches single-position outputs only)";
pub(super) const BECAUSE_L1_DISABLED: &str = "no L1 cache attached";
pub(super) const BECAUSE_L1_MISS: &str = "residual key not present in the L1 cache";
pub(super) const BECAUSE_NOT_SPARSE: &str = "no per-layer sparse K configured";
pub(super) const BECAUSE_NO_FP4: &str = "no FP4/FP8 feature storage";
pub(super) const BECAUSE_NO_KQUANT: &str = "no interleaved K-quant (Q4_K/Q6_K) storage";
pub(super) const BECAUSE_NO_Q4_SLAB: &str = "no interleaved Q4_0 slab";
pub(super) const BECAUSE_NO_Q4_BACKEND: &str =
    "interleaved Q4_0 slab present but no attached backend supports Q4_0";
pub(super) const BECAUSE_NO_INTERLEAVED_F32: &str = "no f32 interleaved storage";
pub(super) const BECAUSE_NO_FULL_MMAP: &str = "no full-mmap gate/up/down payload";
pub(super) const BECAUSE_NO_DOWN_FEATURES: &str = "no feature-major down mmap (down_features)";

const SEL_ZERO_FEATURES: &str = "layer has no vindex features — dense safetensors WeightFfn";
const SEL_BASE_DELTA: &str =
    "overridden layer; every base+delta exactness precondition holds — dense base + O(|patch|) corrections";
const SEL_OVERRIDE_SPARSE: &str =
    "overridden layer — override-aware sparse walk (whole-layer paths are override-blind)";
pub(super) const SEL_OVERRIDE_FALLBACK: &str =
    "overridden layer — override-aware safetensors fallback (must never fall through to override-blind paths)";
const SEL_L1_HIT: &str = "single-position hot call; residual key present in the L1 output cache";
const SEL_SPARSE: &str = "explicit per-layer sparse K configured";
const SEL_FP4: &str =
    "FP4/FP8 storage is per-feature by construction — routed through the sparse walk";
const SEL_KQUANT_NATIVE: &str = "interleaved K-quant storage — direct fused-decode matvec";
const SEL_INTERLEAVED_Q4: &str = "interleaved Q4_0 slab + backend with Q4_0 kernels";
const SEL_INTERLEAVED_F32: &str = "f32 interleaved mmap — three BLAS gemms";
const SEL_FULL_MMAP: &str = "gate/up/down full-mmap files present";
const SEL_KQUANT_DEQUANT: &str = "interleaved K-quant storage — whole-layer dequant fallback";
const SEL_EXACT: &str = "feature-major down mmap; gate/up from safetensors";
const SEL_LAST_RESORT: &str = "last resort — sparse matmul against safetensors weights";

/// Name of the full-K density cutoff in `thresholds.rs`, as reported
/// by the Sparse plan's [`ThresholdCheck`].
pub(super) const THRESHOLD_FULL_K_DENSITY: &str = "FULL_K_DENSITY";

/// Name of the two-stage shortlist width check (item 19, `shortlist.rs`)
/// in the Sparse plan's [`ThresholdCheck`]: `actual` = configured M,
/// `cutoff` = the layer's K (M must cover K), `satisfied` = the
/// two-stage selection will actually run (M ≥ K and the selector has a
/// rerank criterion — `Random` has none).
pub(super) const THRESHOLD_SHORTLIST_M: &str = "SHORTLIST_M";

/// Path-independent L1 key for a single-position input's residual row
/// — the same hashing `forward_ladder` has always used.
pub(super) fn residual_l1_key(x: &Array2<f32>) -> u64 {
    let row = x.row(0);
    if let Some(slice) = row.as_slice() {
        FfnL1Cache::residual_key(slice)
    } else {
        FfnL1Cache::residual_key(&row.to_vec())
    }
}

impl<'a> WalkFfn<'a> {
    /// Plan the routing decision for `layer` on input `x` WITHOUT
    /// executing anything — public inspection surface (2026-07-30
    /// review, item 18). Pure: no dispatch-trace entries, no L1
    /// hit/miss accounting, no kernel work.
    ///
    /// The plan describes what [`FfnBackend::forward`] would do right
    /// now (when the runtime trace is enabled, that call runs in
    /// observed mode and bypasses the L1 read — the plan reflects it).
    /// A fallible path can still decline at execution time; `forward`
    /// then re-plans from the next rung and the executed plan's
    /// `skipped` list records the decline.
    ///
    /// [`FfnBackend::forward`]: crate::ffn::FfnBackend::forward
    pub fn plan_for(&self, layer: usize, x: &Array2<f32>) -> FfnPlan {
        let observe = if self.record_trace {
            Observe::Record
        } else {
            Observe::Skip
        };
        let l1_hit = !observe.recording()
            && x.shape()[0] == 1
            && self
                .l1_cache
                .as_ref()
                .is_some_and(|cache| cache.peek(layer, residual_l1_key(x)));
        self.plan_layer(layer, x, observe, l1_hit, PlanExclusions::default())
    }

    /// The planner: walk the ladder's conditions top-down and return
    /// the first admissible rung, carrying WHY it fired and why every
    /// rung above it did not. `exclude` marks rungs that declined at
    /// execution (skipped with [`BECAUSE_DECLINED`]); `l1_hit` is the
    /// caller's cache probe so the planner itself stays stats-free.
    pub(super) fn plan_layer(
        &self,
        layer: usize,
        x: &Array2<f32>,
        observe: Observe,
        l1_hit: bool,
        exclude: PlanExclusions,
    ) -> FfnPlan {
        let seq_len = x.shape()[0];
        let num_features = self.index.num_features(layer);
        let has_overrides = self.index.has_overrides_at(layer);
        let mut skipped: Vec<SkippedRung> = Vec::new();
        let mut thresholds: Vec<ThresholdCheck> = Vec::new();

        macro_rules! decide {
            ($rung:expr, $selected:expr) => {
                return FfnPlan::from_rung(
                    $rung,
                    PlanReason {
                        layer,
                        seq_len,
                        num_features,
                        has_overrides,
                        selected: $selected,
                        skipped,
                        thresholds,
                    },
                )
            };
        }
        macro_rules! skip {
            ($rung:expr, $because:expr) => {
                skipped.push(SkippedRung {
                    rung: $rung,
                    because: $because,
                })
            };
        }

        // Rung 0 — no features at all.
        if num_features == 0 {
            decide!(PlanRung::ZeroFeaturesDense, SEL_ZERO_FEATURES);
        }
        skip!(PlanRung::ZeroFeaturesDense, BECAUSE_HAS_FEATURES);

        // Rungs 1a/1b — overridden layers route ONLY through the
        // override-aware arm (whole-layer paths and the L1 cache would
        // silently serve pre-patch weights).
        if has_overrides {
            if exclude.contains(PlanRung::OverrideBaseDelta) {
                skip!(PlanRung::OverrideBaseDelta, BECAUSE_DECLINED);
            } else {
                match self.base_delta_preconditions(layer, seq_len, x.shape()[1]) {
                    Ok(_slots) => decide!(PlanRung::OverrideBaseDelta, SEL_BASE_DELTA),
                    Err(why) => skip!(PlanRung::OverrideBaseDelta, why),
                }
            }
            if exclude.contains(PlanRung::OverrideSparse) {
                skip!(PlanRung::OverrideSparse, BECAUSE_DECLINED);
            } else {
                decide!(PlanRung::OverrideSparse, SEL_OVERRIDE_SPARSE);
            }
            decide!(PlanRung::WeightsFallback, SEL_OVERRIDE_FALLBACK);
        }
        skip!(PlanRung::OverrideBaseDelta, BECAUSE_NO_OVERRIDES);
        skip!(PlanRung::OverrideSparse, BECAUSE_NO_OVERRIDES);

        // L1 output cache — Skip-mode single-position calls only.
        // Infallible by construction: planned only when the caller's
        // probe already found the row, so it never re-plans.
        if observe.recording() {
            skip!(PlanRung::L1CacheHit, BECAUSE_L1_OBSERVED);
        } else if seq_len != 1 {
            skip!(PlanRung::L1CacheHit, BECAUSE_L1_MULTI_POSITION);
        } else if self.l1_cache.is_none() {
            skip!(PlanRung::L1CacheHit, BECAUSE_L1_DISABLED);
        } else if !l1_hit {
            skip!(PlanRung::L1CacheHit, BECAUSE_L1_MISS);
        } else {
            decide!(PlanRung::L1CacheHit, SEL_L1_HIT);
        }

        // Rung 2 — explicit sparse K from the user.
        if exclude.contains(PlanRung::Sparse) {
            skip!(PlanRung::Sparse, BECAUSE_DECLINED);
        } else if let Some(k) = self.config.k_for(layer) {
            // Pre-execution knowable: whether the sparse walk rewrites
            // to the full-K gemv. Single-sourced from `helpers.rs` so
            // the reported comparison IS the executed one (`satisfied`
            // also honours `force_walk`).
            thresholds.push(ThresholdCheck {
                name: THRESHOLD_FULL_K_DENSITY,
                actual: k,
                cutoff: (num_features * FULL_K_DENSITY_NUM) / FULL_K_DENSITY_DEN,
                satisfied: hits_len_ge_intermediate(&self.config, layer, num_features),
            });
            // Two-stage shortlist (item 19): recorded whenever the
            // selector-dispatch route would consult it (pools and the
            // cell router take precedence and carry their own
            // two-stage machinery).
            if let Some(m) = self.config.shortlist_m {
                if self.config.pool_per_layer.is_none() && self.config.cell_router.is_none() {
                    thresholds.push(ThresholdCheck {
                        name: THRESHOLD_SHORTLIST_M,
                        actual: m,
                        cutoff: k,
                        satisfied: m >= k
                            && super::shortlist::criterion_inputs(self.config.selector).is_some(),
                    });
                }
            }
            decide!(PlanRung::Sparse, SEL_SPARSE);
        } else {
            skip!(PlanRung::Sparse, BECAUSE_NOT_SPARSE);
        }

        // Rung 3 — FP4/FP8 storage has no dense path; sparse walk.
        if exclude.contains(PlanRung::Fp4Sparse) {
            skip!(PlanRung::Fp4Sparse, BECAUSE_DECLINED);
        } else if self.index.has_fp4_storage() {
            decide!(PlanRung::Fp4Sparse, SEL_FP4);
        } else {
            skip!(PlanRung::Fp4Sparse, BECAUSE_NO_FP4);
        }

        // Rungs 4–9 — whole-layer paths, perf-preference ordered.
        for rung in WHOLE_LAYER_RUNGS {
            if exclude.contains(rung) {
                skip!(rung, BECAUSE_DECLINED);
            } else if self.whole_layer_rung_available(rung) {
                decide!(rung, whole_layer_selected_reason(rung));
            } else {
                skip!(rung, self.whole_layer_unavailable_reason(rung));
            }
        }

        // Rung 10 — always admissible.
        decide!(PlanRung::WeightsFallback, SEL_LAST_RESORT);
    }

    /// Static availability condition for a whole-layer rung — the
    /// exact flag checks the pre-planner ladder inlined.
    pub(super) fn whole_layer_rung_available(&self, rung: PlanRung) -> bool {
        match rung {
            PlanRung::KquantNative | PlanRung::KquantDequant => self.index.has_interleaved_kquant(),
            PlanRung::InterleavedQ4 => {
                // Gate on the format the data actually is (Q4_0): gating
                // on Q4_K here used to admit `CpuBackend` into an
                // unwrap-on-None panic (2026-07-30 review, H2).
                self.index.has_interleaved_q4()
                    && self
                        .backend
                        .is_some_and(|be| be.supports_quant(::larql_compute::QuantFormat::Q4_0))
            }
            PlanRung::InterleavedF32 => self.index.has_interleaved(),
            PlanRung::FullMmap => self.index.has_full_mmap_ffn(),
            PlanRung::Exact => self.index.has_down_features(),
            // Not a whole-layer rung — has no availability flag here.
            _ => false,
        }
    }

    /// Why a whole-layer rung's availability condition did not hold.
    fn whole_layer_unavailable_reason(&self, rung: PlanRung) -> &'static str {
        match rung {
            PlanRung::KquantNative | PlanRung::KquantDequant => BECAUSE_NO_KQUANT,
            PlanRung::InterleavedQ4 => {
                if self.index.has_interleaved_q4() {
                    BECAUSE_NO_Q4_BACKEND
                } else {
                    BECAUSE_NO_Q4_SLAB
                }
            }
            PlanRung::InterleavedF32 => BECAUSE_NO_INTERLEAVED_F32,
            PlanRung::FullMmap => BECAUSE_NO_FULL_MMAP,
            _ => BECAUSE_NO_DOWN_FEATURES,
        }
    }
}

/// Selected-reason string for a whole-layer rung.
fn whole_layer_selected_reason(rung: PlanRung) -> &'static str {
    match rung {
        PlanRung::KquantNative => SEL_KQUANT_NATIVE,
        PlanRung::InterleavedQ4 => SEL_INTERLEAVED_Q4,
        PlanRung::InterleavedF32 => SEL_INTERLEAVED_F32,
        PlanRung::FullMmap => SEL_FULL_MMAP,
        PlanRung::KquantDequant => SEL_KQUANT_DEQUANT,
        _ => SEL_EXACT,
    }
}
