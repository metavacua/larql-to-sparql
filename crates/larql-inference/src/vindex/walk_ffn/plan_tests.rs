//! Planner tests (2026-07-30 review, item 18) — one test per ladder
//! rung asserting plan + reason on the same fixture styles the
//! dispatch/routing tests use, plus the decline→re-plan contract: a
//! path whose static condition holds but whose execution returns
//! `None` is re-planned past, and the executed plan's reason records
//! the decline. The routing behaviour itself is pinned UNCHANGED by
//! `dispatch_tests.rs` / `routing_tests.rs`.

use ndarray::Array2;

use super::observe::Observe;
use super::plan::{FfnPlan, PlanExclusions, PlanReason, PlanRung};
use super::planner;
use super::routing_tests::MockIndex;
use super::WalkFfn;
use crate::ffn::FfnBackend;
use crate::model::ModelWeights;
use crate::test_utils::make_test_weights;
use larql_vindex::{
    FeatureMeta, Fp4FfnAccess, GateLookup, NativeFfnAccess, OverrideSlot, PatchOverrides,
    QuantizedFfnAccess,
};
use std::sync::OnceLock;

fn shared_weights() -> &'static ModelWeights {
    static W: OnceLock<ModelWeights> = OnceLock::new();
    W.get_or_init(make_test_weights)
}

/// `MockIndex` with every capability flag off and `n` features.
fn base_mock(n: usize) -> MockIndex {
    MockIndex {
        num_features: n,
        has_overrides: false,
        has_fp4: false,
        has_q4_interleaved: false,
        has_interleaved: false,
        has_full_mmap: false,
        has_q4k: false,
        has_down_features: false,
        native_up: None,
        native_down: None,
    }
}

fn input(seq: usize, hidden: usize) -> Array2<f32> {
    Array2::from_shape_vec(
        (seq, hidden),
        (0..seq * hidden).map(|i| (i as f32 + 1.0) * 0.02).collect(),
    )
    .unwrap()
}

/// Skip reason recorded for `rung`, or panic if the rung isn't in the
/// skipped list.
fn because_for(plan: &FfnPlan, rung: PlanRung) -> &'static str {
    plan.reason()
        .skipped
        .iter()
        .find(|s| s.rung == rung)
        .unwrap_or_else(|| panic!("{rung:?} not in skipped list: {:?}", plan.reason().skipped))
        .because
}

// ── One plan test per rung ───────────────────────────────────────────

#[test]
fn plans_zero_features_dense() {
    let weights = shared_weights();
    let idx = base_mock(0);
    let ffn = WalkFfn::new_unlimited(weights, &idx);
    let plan = ffn.plan_for(0, &input(1, weights.hidden_size));
    assert_eq!(plan.rung(), PlanRung::ZeroFeaturesDense);
    let r = plan.reason();
    assert_eq!((r.layer, r.seq_len, r.num_features), (0, 1, 0));
    assert!(!r.has_overrides);
    assert!(r.skipped.is_empty(), "rung 0 has nothing above it");
}

#[test]
fn plans_weights_fallback_as_last_resort_with_full_skip_ladder() {
    let weights = shared_weights();
    let idx = base_mock(weights.intermediate_size);
    let ffn = WalkFfn::new_unlimited(weights, &idx);
    let plan = ffn.plan_for(1, &input(1, weights.hidden_size));
    assert_eq!(plan.rung(), PlanRung::WeightsFallback);
    // Every one of the 12 rungs above must be accounted for, in
    // ladder order, each with the reason it did not fire.
    let skipped: Vec<PlanRung> = plan.reason().skipped.iter().map(|s| s.rung).collect();
    assert_eq!(
        skipped,
        vec![
            PlanRung::ZeroFeaturesDense,
            PlanRung::OverrideBaseDelta,
            PlanRung::OverrideSparse,
            PlanRung::L1CacheHit,
            PlanRung::Sparse,
            PlanRung::Fp4Sparse,
            PlanRung::KquantNative,
            PlanRung::InterleavedQ4,
            PlanRung::InterleavedF32,
            PlanRung::FullMmap,
            PlanRung::KquantDequant,
            PlanRung::Exact,
        ]
    );
    assert_eq!(
        because_for(&plan, PlanRung::Exact),
        planner::BECAUSE_NO_DOWN_FEATURES
    );
    assert_eq!(
        because_for(&plan, PlanRung::L1CacheHit),
        planner::BECAUSE_L1_DISABLED
    );
}

#[test]
fn plans_sparse_with_full_k_density_threshold() {
    let weights = shared_weights();
    let idx = base_mock(weights.intermediate_size);
    // k=4 of 32 features → cutoff 32*4/5 = 25 → not dense enough.
    let ffn = WalkFfn::new(weights, &idx, 4);
    let plan = ffn.plan_for(0, &input(1, weights.hidden_size));
    assert_eq!(plan.rung(), PlanRung::Sparse);
    let t = &plan.reason().thresholds[0];
    assert_eq!(t.name, "FULL_K_DENSITY");
    assert_eq!((t.actual, t.cutoff, t.satisfied), (4, 25, false));

    // k=30 ≥ 25 → the walk is rewritten to the full-K gemv.
    let ffn = WalkFfn::new(weights, &idx, 30);
    let plan = ffn.plan_for(0, &input(1, weights.hidden_size));
    let t = &plan.reason().thresholds[0];
    assert_eq!((t.actual, t.cutoff, t.satisfied), (30, 25, true));
}

#[test]
fn plans_fp4_sparse_and_reports_why_not_sparse() {
    let weights = shared_weights();
    let mut idx = base_mock(weights.intermediate_size);
    idx.has_fp4 = true;
    let ffn = WalkFfn::new_unlimited(weights, &idx);
    let plan = ffn.plan_for(0, &input(1, weights.hidden_size));
    assert_eq!(plan.rung(), PlanRung::Fp4Sparse);
    assert_eq!(
        because_for(&plan, PlanRung::Sparse),
        planner::BECAUSE_NOT_SPARSE
    );
}

#[test]
fn plans_kquant_native_over_lower_rungs() {
    let weights = shared_weights();
    let mut idx = base_mock(weights.intermediate_size);
    idx.has_q4k = true;
    idx.has_interleaved = true;
    idx.has_full_mmap = true;
    let ffn = WalkFfn::new_unlimited(weights, &idx);
    let plan = ffn.plan_for(0, &input(1, weights.hidden_size));
    assert_eq!(plan.rung(), PlanRung::KquantNative);
}

#[test]
fn plans_interleaved_q4_only_with_a_q4_backend() {
    let weights = shared_weights();
    let mut idx = base_mock(weights.intermediate_size);
    idx.has_q4_interleaved = true;
    // No backend → the slab alone is not enough (H2 regression class).
    let ffn = WalkFfn::new_unlimited(weights, &idx);
    let plan = ffn.plan_for(0, &input(1, weights.hidden_size));
    assert_eq!(plan.rung(), PlanRung::WeightsFallback);
    assert_eq!(
        because_for(&plan, PlanRung::InterleavedQ4),
        planner::BECAUSE_NO_Q4_BACKEND
    );
    // CpuBackend supports Q4_0 → the rung fires.
    let ffn = WalkFfn::new_unlimited_with_backend(weights, &idx, &larql_compute::CpuBackend);
    let plan = ffn.plan_for(0, &input(1, weights.hidden_size));
    assert_eq!(plan.rung(), PlanRung::InterleavedQ4);
    // And with no slab at all the reason is the slab, not the backend.
    let bare = base_mock(weights.intermediate_size);
    let ffn = WalkFfn::new_unlimited(weights, &bare);
    let plan = ffn.plan_for(0, &input(1, weights.hidden_size));
    assert_eq!(
        because_for(&plan, PlanRung::InterleavedQ4),
        planner::BECAUSE_NO_Q4_SLAB
    );
}

#[test]
fn plans_interleaved_f32_full_mmap_and_exact() {
    let weights = shared_weights();
    let x = input(1, weights.hidden_size);
    for (set, rung) in [
        (
            Box::new(|m: &mut MockIndex| m.has_interleaved = true) as Box<dyn Fn(&mut MockIndex)>,
            PlanRung::InterleavedF32,
        ),
        (
            Box::new(|m: &mut MockIndex| m.has_full_mmap = true),
            PlanRung::FullMmap,
        ),
        (
            Box::new(|m: &mut MockIndex| m.has_down_features = true),
            PlanRung::Exact,
        ),
    ] {
        let mut idx = base_mock(weights.intermediate_size);
        set(&mut idx);
        let ffn = WalkFfn::new_unlimited(weights, &idx);
        assert_eq!(ffn.plan_for(0, &x).rung(), rung);
    }
}

#[test]
fn plans_kquant_dequant_when_native_declined() {
    let weights = shared_weights();
    let mut idx = base_mock(weights.intermediate_size);
    idx.has_q4k = true;
    let ffn = WalkFfn::new_unlimited(weights, &idx);
    let x = input(1, weights.hidden_size);
    let exclude = PlanExclusions::default().with(PlanRung::KquantNative);
    let plan = ffn.plan_layer(0, &x, Observe::Skip, false, exclude);
    assert_eq!(plan.rung(), PlanRung::KquantDequant);
    assert_eq!(
        because_for(&plan, PlanRung::KquantNative),
        planner::BECAUSE_DECLINED
    );
}

// ── Override arm ─────────────────────────────────────────────────────

/// `MockIndex` wrapper whose patch surface can enumerate override
/// slots — the missing piece for the base+delta preconditions.
struct OverrideEnumIndex {
    inner: MockIndex,
}

impl GateLookup for OverrideEnumIndex {
    fn gate_knn(
        &self,
        layer: usize,
        residual: &ndarray::Array1<f32>,
        top_k: usize,
    ) -> Vec<(usize, f32)> {
        self.inner.gate_knn(layer, residual, top_k)
    }
    fn feature_meta(&self, layer: usize, feature: usize) -> Option<FeatureMeta> {
        self.inner.feature_meta(layer, feature)
    }
    fn num_features(&self, layer: usize) -> usize {
        self.inner.num_features(layer)
    }
}
impl PatchOverrides for OverrideEnumIndex {
    fn has_overrides_at(&self, _layer: usize) -> bool {
        true
    }
    fn override_slots_at(&self, _layer: usize) -> Option<Vec<OverrideSlot>> {
        Some(Vec::new())
    }
}
impl NativeFfnAccess for OverrideEnumIndex {
    fn has_interleaved(&self) -> bool {
        self.inner.has_interleaved
    }
}
impl QuantizedFfnAccess for OverrideEnumIndex {}
impl Fp4FfnAccess for OverrideEnumIndex {}

#[test]
fn plans_override_base_delta_when_preconditions_hold() {
    let weights = shared_weights();
    let mut inner = base_mock(weights.intermediate_size);
    inner.has_interleaved = true; // exactly one coherent base (f32)
    let idx = OverrideEnumIndex { inner };
    let ffn = WalkFfn::new_unlimited(weights, &idx);
    let plan = ffn.plan_for(0, &input(1, weights.hidden_size));
    assert_eq!(plan.rung(), PlanRung::OverrideBaseDelta);
    assert!(plan.reason().has_overrides);
}

#[test]
fn plans_override_sparse_with_the_failed_precondition_as_reason() {
    let weights = shared_weights();
    // No storage at all → no coherent whole-layer base for the delta.
    let idx = OverrideEnumIndex {
        inner: base_mock(weights.intermediate_size),
    };
    let ffn = WalkFfn::new_unlimited(weights, &idx);
    let plan = ffn.plan_for(0, &input(1, weights.hidden_size));
    assert_eq!(plan.rung(), PlanRung::OverrideSparse);
    assert_eq!(
        because_for(&plan, PlanRung::OverrideBaseDelta),
        super::base_delta::BD_NO_COHERENT_BASE
    );
}

#[test]
fn plans_override_fallback_when_both_override_paths_declined() {
    let weights = shared_weights();
    let idx = OverrideEnumIndex {
        inner: base_mock(weights.intermediate_size),
    };
    let ffn = WalkFfn::new_unlimited(weights, &idx);
    let x = input(1, weights.hidden_size);
    let exclude = PlanExclusions::default()
        .with(PlanRung::OverrideBaseDelta)
        .with(PlanRung::OverrideSparse);
    let plan = ffn.plan_layer(0, &x, Observe::Skip, false, exclude);
    assert_eq!(plan.rung(), PlanRung::WeightsFallback);
    assert_eq!(plan.reason().selected, planner::SEL_OVERRIDE_FALLBACK);
    assert_eq!(
        because_for(&plan, PlanRung::OverrideSparse),
        planner::BECAUSE_DECLINED
    );
}

// ── L1 cache rung + plan_for purity ─────────────────────────────────

#[test]
fn plans_l1_cache_hit_and_plan_for_is_stats_free() {
    let weights = shared_weights();
    let idx = base_mock(weights.intermediate_size);
    let ffn = WalkFfn::new_unlimited(weights, &idx)
        .with_l1_cache(weights.num_layers)
        .with_dispatch_trace();
    let x = input(1, weights.hidden_size);

    // Cold: the plan honestly reports the miss...
    let plan = ffn.plan_for(0, &x);
    assert_eq!(
        because_for(&plan, PlanRung::L1CacheHit),
        planner::BECAUSE_L1_MISS
    );
    // ...and probing was stats-free.
    assert_eq!(ffn.l1_cache_stats(), Some((0, 0)));

    // Warm the slot, then the plan must switch to the hit — again
    // without disturbing the executed forward's accounting.
    ffn.forward(0, &x);
    let stats_after_forward = ffn.l1_cache_stats();
    let plan = ffn.plan_for(0, &x);
    assert_eq!(plan.rung(), PlanRung::L1CacheHit);
    assert_eq!(ffn.l1_cache_stats(), stats_after_forward);

    // The plan predicts exactly what the next hot call does.
    ffn.take_dispatch_trace();
    ffn.forward(0, &x);
    assert_eq!(
        ffn.take_dispatch_trace().last().map(|e| e.path),
        Some(plan.rung().name())
    );
}

#[test]
fn multi_position_input_skips_l1_with_the_right_reason() {
    let weights = shared_weights();
    let idx = base_mock(weights.intermediate_size);
    let ffn = WalkFfn::new_unlimited(weights, &idx).with_l1_cache(weights.num_layers);
    let plan = ffn.plan_for(0, &input(3, weights.hidden_size));
    assert_eq!(
        because_for(&plan, PlanRung::L1CacheHit),
        planner::BECAUSE_L1_MULTI_POSITION
    );
    assert_eq!(plan.reason().seq_len, 3);
}

// ── Decline → re-plan, end to end ───────────────────────────────────

/// Every capability flag lies (set with no bytes behind it) and the
/// config demands a sparse walk with nothing to walk on: Sparse, Q4K
/// native, Q4_0, f32 interleaved, full-mmap and Q4K dequant must each
/// decline at execution, the ladder re-plans past each one, and the
/// executed plan's reason records every decline. The dispatch trace
/// proves routing landed exactly where the pre-planner ladder did.
#[test]
fn declining_paths_replan_to_the_next_rung_and_record_the_decline() {
    let weights = shared_weights();
    let mut idx = base_mock(weights.intermediate_size);
    idx.has_q4k = true;
    idx.has_q4_interleaved = true;
    idx.has_interleaved = true;
    idx.has_full_mmap = true;
    let ffn = WalkFfn::new(weights, &idx, 2)
        .with_backend(&larql_compute::CpuBackend)
        .with_dispatch_trace()
        .with_trace();
    let x = input(1, weights.hidden_size);

    let out = ffn.forward(0, &x);
    assert_eq!(out.shape(), &[1, weights.hidden_size]);
    assert_eq!(
        ffn.take_dispatch_trace().last().map(|e| e.path),
        Some("weights_fallback:sparse"),
        "identical destination to the pre-planner ladder"
    );

    let records = ffn.take_runtime_trace();
    let reason = &records[0].plan_reason;
    for declined in [
        PlanRung::Sparse,
        PlanRung::KquantNative,
        PlanRung::InterleavedQ4,
        PlanRung::InterleavedF32,
        PlanRung::FullMmap,
        PlanRung::KquantDequant,
    ] {
        assert!(
            reason.contains(&format!(
                "{} ({})",
                declined.name(),
                planner::BECAUSE_DECLINED
            )),
            "missing decline record for {declined:?} in: {reason}"
        );
    }
}

// ── Type-level details ──────────────────────────────────────────────

#[test]
fn every_rung_round_trips_through_its_plan_variant() {
    const ALL_RUNGS: [PlanRung; 13] = [
        PlanRung::ZeroFeaturesDense,
        PlanRung::OverrideBaseDelta,
        PlanRung::OverrideSparse,
        PlanRung::L1CacheHit,
        PlanRung::Sparse,
        PlanRung::Fp4Sparse,
        PlanRung::KquantNative,
        PlanRung::InterleavedQ4,
        PlanRung::InterleavedF32,
        PlanRung::FullMmap,
        PlanRung::KquantDequant,
        PlanRung::Exact,
        PlanRung::WeightsFallback,
    ];
    for rung in ALL_RUNGS {
        let reason = PlanReason {
            layer: 3,
            seq_len: 1,
            num_features: 8,
            has_overrides: false,
            selected: "test",
            skipped: Vec::new(),
            thresholds: Vec::new(),
        };
        let plan = FfnPlan::from_rung(rung, reason.clone());
        assert_eq!(plan.rung(), rung);
        assert_eq!(plan.reason(), &reason);
        assert!(!rung.name().is_empty());
    }
}

#[test]
fn traced_walkffn_plans_in_observed_mode() {
    // `with_trace` upgrades every forward to Record, which bypasses
    // the L1 read — the plan must say so.
    let weights = shared_weights();
    let idx = base_mock(weights.intermediate_size);
    let ffn = WalkFfn::new_unlimited(weights, &idx)
        .with_l1_cache(weights.num_layers)
        .with_trace();
    let plan = ffn.plan_for(0, &input(1, weights.hidden_size));
    assert_eq!(
        because_for(&plan, PlanRung::L1CacheHit),
        planner::BECAUSE_L1_OBSERVED
    );
}

#[test]
fn non_contiguous_single_row_input_plans_via_owned_key() {
    let weights = shared_weights();
    let idx = base_mock(weights.intermediate_size);
    let ffn = WalkFfn::new_unlimited(weights, &idx).with_l1_cache(weights.num_layers);
    let mut x = input(1, weights.hidden_size);
    x.invert_axis(ndarray::Axis(1));
    assert!(
        x.row(0).as_slice().is_none(),
        "fixture must be non-contiguous"
    );
    let plan = ffn.plan_for(0, &x);
    assert_eq!(
        because_for(&plan, PlanRung::L1CacheHit),
        planner::BECAUSE_L1_MISS
    );
}

#[test]
fn exclusions_are_a_set() {
    let e = PlanExclusions::default();
    assert!(!e.contains(PlanRung::Sparse));
    let e = e.with(PlanRung::Sparse).with(PlanRung::FullMmap);
    assert!(e.contains(PlanRung::Sparse));
    assert!(e.contains(PlanRung::FullMmap));
    assert!(!e.contains(PlanRung::Exact));
}

#[test]
fn non_whole_layer_rungs_have_no_availability() {
    let weights = shared_weights();
    let idx = base_mock(weights.intermediate_size);
    let ffn = WalkFfn::new_unlimited(weights, &idx);
    assert!(!ffn.whole_layer_rung_available(PlanRung::Sparse));
}

#[test]
fn summary_carries_selection_thresholds_and_skips() {
    let weights = shared_weights();
    let idx = base_mock(weights.intermediate_size);
    let ffn = WalkFfn::new(weights, &idx, 4);
    let plan = ffn.plan_for(1, &input(1, weights.hidden_size));
    let s = plan.reason().summary();
    assert!(s.contains("layer=1"), "{s}");
    assert!(
        s.contains("FULL_K_DENSITY actual=4 cutoff=25 satisfied=false"),
        "{s}"
    );
    assert!(s.contains(planner::BECAUSE_NO_OVERRIDES), "{s}");
    // The skipped section names the rung directly above.
    assert!(s.contains("l1_cache_hit ("), "{s}");
}
