//! Tests for two-stage shortlist-then-rerank selection (item 19,
//! `shortlist.rs`): equivalence at M = N against the full-projection
//! `joint_gate_knn`, the shortlist actually restricting stage 2 to the
//! stage-1 candidates, observable declines, the default-off pin, the
//! cost pin (no full projection in two-stage mode), and the planner /
//! runtime-trace integration.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use ndarray::{Array1, Array2};

use super::plan::PlanRung;
use super::planner::THRESHOLD_SHORTLIST_M;
use super::shortlist::{criterion_inputs, criterion_weight, PATH_SHORTLIST_DECLINED};
use crate::test_utils::{
    make_test_q4k_vindex_with_model_gate, make_test_q4k_weights, make_test_weights,
};
use crate::vindex::walk_config::FeatureSelector;
use crate::vindex::{WalkFfn, WalkFfnConfig};
use larql_vindex::{
    FeatureMeta, Fp4FfnAccess, GateLookup, NativeFfnAccess, PatchOverrides, QuantizedFfnAccess,
    VectorIndex,
};

/// Top-K used throughout — small enough that selections are
/// unambiguous on the 256-feature Q4K fixture.
const K: usize = 4;

/// The five scored selectors (everything except `Random`, which has no
/// rerank criterion).
const SCORED: [FeatureSelector; 5] = [
    FeatureSelector::GateOnly,
    FeatureSelector::GateXDownNorm,
    FeatureSelector::GateXUpDownNorm,
    FeatureSelector::GateXUpScore,
    FeatureSelector::ActXUpScoreXDownNorm,
];

fn residual_ramp(hidden: usize) -> Array1<f32> {
    Array1::from_vec((0..hidden).map(|i| (i as f32 + 1.0) * 0.02).collect())
}

// ── criterion table ─────────────────────────────────────────────────────

/// Pin the needs table: which stage-2 inputs each criterion consumes,
/// and that `Random` is unscored (no two-stage).
#[test]
fn criterion_inputs_match_the_formulas() {
    for kind in SCORED {
        let needs = criterion_inputs(kind).expect("scored selector has a criterion");
        let expected = match kind {
            FeatureSelector::GateOnly => (false, false, false),
            FeatureSelector::GateXDownNorm => (false, false, true),
            FeatureSelector::GateXUpDownNorm => (false, true, true),
            FeatureSelector::GateXUpScore => (true, false, false),
            FeatureSelector::ActXUpScoreXDownNorm => (true, false, true),
            FeatureSelector::Random => unreachable!(),
        };
        assert_eq!(
            (needs.up_score, needs.up_norm, needs.down_norm),
            expected,
            "{kind:?}"
        );
    }
    assert!(criterion_inputs(FeatureSelector::Random).is_none());
}

/// `Random` has no weight formula — reaching `criterion_weight` with it
/// is a caller bug and must panic, never fabricate a score.
#[test]
#[should_panic(expected = "Random has no rerank criterion")]
fn criterion_weight_panics_on_random() {
    criterion_weight(FeatureSelector::Random, false, 1.0, 1.0, 1.0, 1.0);
}

/// The weight formulas themselves, against hand computation — both the
/// GELU and SiLU activation arms of `ActXUpScoreXDownNorm`.
#[test]
fn criterion_weight_formulas_match_hand_computation() {
    let (g, u, un, dn) = (-0.5_f32, 2.0_f32, 3.0_f32, 4.0_f32);
    let cases = [
        (FeatureSelector::GateOnly, g.abs()),
        (FeatureSelector::GateXDownNorm, g.abs() * dn),
        (FeatureSelector::GateXUpDownNorm, g.abs() * dn * un),
        (FeatureSelector::GateXUpScore, g.abs() * u.abs()),
        (
            FeatureSelector::ActXUpScoreXDownNorm,
            (crate::ffn::gelu_tanh(g) * u).abs() * dn,
        ),
    ];
    for (kind, expected) in cases {
        assert_eq!(
            criterion_weight(kind, true, g, u, un, dn),
            expected,
            "{kind:?}"
        );
    }
    // SiLU arm of the activated criterion.
    assert_eq!(
        criterion_weight(FeatureSelector::ActXUpScoreXDownNorm, false, g, u, un, dn),
        (g * crate::ffn::sigmoid(g) * u).abs() * dn
    );
}

// ── (a) equivalence at M = num_features ─────────────────────────────────

/// With M = num_features the shortlist admits every feature, so the
/// two-stage selection must equal the full-projection `joint_gate_knn`
/// — same features, same (deterministic rerank) order — for every
/// scored selector, with no decline and no selector fallback.
#[test]
fn two_stage_at_full_m_equals_joint_gate_knn_for_every_scored_selector() {
    let weights = make_test_q4k_weights();
    let index = make_test_q4k_vindex_with_model_gate(&weights);
    let residual = residual_ramp(weights.hidden_size);
    let x_slice = residual.as_slice().unwrap();
    for kind in SCORED {
        let walk = WalkFfn::from_config(
            &weights,
            &index,
            WalkFfnConfig::sparse(weights.num_layers, K).with_selector(kind),
        );
        let n = index.num_features(0);
        let joint = walk.joint_gate_knn(0, &residual, K, kind);
        let two_stage = walk.shortlist_rerank(0, &residual, x_slice, K, n, kind);
        let joint_feats: Vec<usize> = joint.iter().map(|&(f, _)| f).collect();
        let two_feats: Vec<usize> = two_stage.iter().map(|&(f, _)| f).collect();
        assert_eq!(
            two_feats, joint_feats,
            "{kind:?}: two-stage at M=N must select the same features in the same rerank order"
        );
        // Both carry the RAW gate score for each feature. The two paths
        // score gates through the same bytes (batched gemm vs the
        // production chain's gemv), so the scores must agree to BLAS
        // reassociation noise at most.
        for (&(f, g2), &(_, gj)) in two_stage.iter().zip(joint.iter()) {
            assert!(
                (g2 - gj).abs() <= 1e-5 * gj.abs().max(1.0),
                "{kind:?}: raw gate score mismatch for feature {f}: {g2} vs {gj}"
            );
        }
        assert_eq!(
            walk.shortlist_decline_count(),
            0,
            "{kind:?}: must not decline"
        );
        assert_eq!(
            walk.selector_fallback_count(),
            0,
            "{kind:?}: must not fall back"
        );
    }
}

// ── (b) the shortlist actually restricts ────────────────────────────────

/// Hand-built index for the restriction pin: 8 features whose gate
/// scores descend, with the SMALLEST-gate feature carrying an enormous
/// `‖down_row‖` so the full-projection `GateXDownNorm` rerank picks it
/// first — but a gate shortlist of M = 4 never admits it.
struct RestrictionIndex {
    gate_scores: Vec<f32>,
    down: Arc<Vec<f32>>,
}

impl RestrictionIndex {
    fn top_k(&self, top_k: usize) -> Vec<(usize, f32)> {
        let mut idx: Vec<usize> = (0..self.gate_scores.len()).collect();
        idx.sort_by(|&a, &b| {
            self.gate_scores[b]
                .abs()
                .partial_cmp(&self.gate_scores[a].abs())
                .expect("NaN-free")
        });
        idx.truncate(top_k);
        idx.into_iter().map(|i| (i, self.gate_scores[i])).collect()
    }
}

impl GateLookup for RestrictionIndex {
    fn gate_knn(&self, _layer: usize, _r: &Array1<f32>, top_k: usize) -> Vec<(usize, f32)> {
        self.top_k(top_k)
    }
    fn gate_walk(
        &self,
        _layer: usize,
        _r: &Array1<f32>,
        top_k: usize,
    ) -> Option<Vec<(usize, f32)>> {
        Some(self.top_k(top_k))
    }
    fn gate_scores_batch(&self, _layer: usize, _x: &Array2<f32>) -> Option<Array2<f32>> {
        Array2::from_shape_vec((1, self.gate_scores.len()), self.gate_scores.clone()).ok()
    }
    fn feature_meta(&self, _layer: usize, _feature: usize) -> Option<FeatureMeta> {
        None
    }
    fn num_features(&self, _layer: usize) -> usize {
        self.gate_scores.len()
    }
}
impl QuantizedFfnAccess for RestrictionIndex {
    fn kquant_ffn_layer(&self, _layer: usize, component: usize) -> Option<Arc<Vec<f32>>> {
        (component == 2).then(|| Arc::clone(&self.down))
    }
}
impl PatchOverrides for RestrictionIndex {}
impl NativeFfnAccess for RestrictionIndex {}
impl Fp4FfnAccess for RestrictionIndex {}

/// THE cost-model pin: a feature the full-projection rerank WOULD pick
/// (huge `‖down‖`, tiny gate) sits outside the top-M gate shortlist —
/// two-stage must exclude it. This is the point of the feature: stage 2
/// only ever sees stage-1 candidates.
#[test]
fn shortlist_restricts_stage_two_to_stage_one_candidates() {
    /// `‖down_row‖` of the decoy feature — large enough that
    /// gate × norm dominates every honest candidate's weight.
    const DECOY_DOWN_NORM: f32 = 1000.0;
    const N_FEATURES: usize = 8;
    const M: usize = 4;

    let weights = make_test_weights();
    let hidden = weights.hidden_size;
    let decoy = N_FEATURES - 1;
    // Gates descend 0.9, 0.8, … with the decoy smallest at 0.05.
    let mut gate_scores: Vec<f32> = (0..N_FEATURES).map(|i| 0.9 - 0.1 * i as f32).collect();
    gate_scores[decoy] = 0.05;
    // Down rows are unit-norm e₀, except the decoy's huge row.
    let mut down = vec![0.0_f32; N_FEATURES * hidden];
    for (i, row) in down.chunks_mut(hidden).enumerate() {
        row[0] = if i == decoy { DECOY_DOWN_NORM } else { 1.0 };
    }
    let index = RestrictionIndex {
        gate_scores,
        down: Arc::new(down),
    };
    let walk = WalkFfn::from_config(
        &weights,
        &index,
        WalkFfnConfig::sparse(weights.num_layers, 2)
            .with_selector(FeatureSelector::GateXDownNorm)
            .with_shortlist_m(M),
    );
    let residual = residual_ramp(hidden);
    let x_slice = residual.as_slice().unwrap();

    // Full projection picks the decoy first: |0.05| × 1000 ≫ |0.9| × 1.
    let joint = walk.joint_gate_knn(0, &residual, 2, FeatureSelector::GateXDownNorm);
    assert_eq!(
        joint[0].0, decoy,
        "the full-projection rerank must pick the decoy — otherwise this test pins nothing"
    );

    // Two-stage: stage 1 admits only the top-M by gate = features 0..4;
    // stage 2 reranks within them. The decoy is excluded by structure.
    let two_stage =
        walk.shortlist_rerank(0, &residual, x_slice, 2, M, FeatureSelector::GateXDownNorm);
    let feats: Vec<usize> = two_stage.iter().map(|&(f, _)| f).collect();
    assert!(
        !feats.contains(&decoy),
        "two-stage must exclude the outside-shortlist decoy, got {feats:?}"
    );
    assert!(
        feats.iter().all(|&f| f < M),
        "every hit must come from the stage-1 top-{M} gate shortlist, got {feats:?}"
    );
    assert_eq!(
        feats,
        vec![0, 1],
        "rerank within {{0..4}} by |gate|×1 keeps the gate order"
    );
    assert_eq!(walk.shortlist_decline_count(), 0);
}

// ── (c) declines are observable ─────────────────────────────────────────

/// M < K cannot fill the requested top-K from the shortlist: the call
/// must decline to single-stage — same result as `joint_gate_knn` —
/// with the `shortlist:declined` trace entry and a counted decline,
/// never silence.
#[test]
fn shortlist_m_below_k_declines_observably_to_single_stage() {
    let weights = make_test_q4k_weights();
    let index = make_test_q4k_vindex_with_model_gate(&weights);
    let kind = FeatureSelector::GateXDownNorm;
    let walk = WalkFfn::from_config(
        &weights,
        &index,
        WalkFfnConfig::sparse(weights.num_layers, K)
            .with_selector(kind)
            .with_shortlist_m(K - 1),
    )
    .with_dispatch_trace();
    let residual = residual_ramp(weights.hidden_size);
    let x_slice = residual.as_slice().unwrap();

    let hits = walk.shortlist_rerank(0, &residual, x_slice, K, K - 1, kind);
    assert_eq!(
        walk.shortlist_decline_count(),
        1,
        "the decline must be counted"
    );
    let trace = walk.take_dispatch_trace();
    assert!(
        trace.iter().any(|e| e.path == PATH_SHORTLIST_DECLINED),
        "the decline must leave a dispatch-trace entry, got {trace:?}"
    );
    // The declined call still selects — via the single-stage path.
    assert_eq!(hits, walk.joint_gate_knn(0, &residual, K, kind));
}

/// `Random` has no rerank criterion, so a configured shortlist declines
/// observably and the single-stage random control still runs.
#[test]
fn shortlist_with_random_selector_declines_observably() {
    let weights = make_test_q4k_weights();
    let index = make_test_q4k_vindex_with_model_gate(&weights);
    let walk = WalkFfn::from_config(
        &weights,
        &index,
        WalkFfnConfig::sparse(weights.num_layers, K)
            .with_selector(FeatureSelector::Random)
            .with_shortlist_m(64),
    )
    .with_dispatch_trace();
    let residual = residual_ramp(weights.hidden_size);
    let x_slice = residual.as_slice().unwrap();

    let hits = walk.shortlist_rerank(0, &residual, x_slice, K, 64, FeatureSelector::Random);
    assert_eq!(hits.len(), K, "single-stage random control still selects K");
    assert_eq!(walk.shortlist_decline_count(), 1);
    let trace = walk.take_dispatch_trace();
    assert!(trace.iter().any(|e| e.path == PATH_SHORTLIST_DECLINED));
}

/// Stage-2 inputs unavailable (an index with gate vectors but no FFN
/// storage at all): the shortlist declines and the single-stage joint
/// path takes over — which then makes its own observable fallback.
#[test]
fn shortlist_missing_stage_two_inputs_declines_observably() {
    let weights = make_test_weights();
    let index = crate::test_utils::make_test_vindex(&weights); // no FFN data
    let kind = FeatureSelector::GateXDownNorm;
    let walk = WalkFfn::from_config(
        &weights,
        &index,
        WalkFfnConfig::sparse(weights.num_layers, K)
            .with_selector(kind)
            .with_shortlist_m(8),
    );
    let residual = residual_ramp(weights.hidden_size);
    let x_slice = residual.as_slice().unwrap();

    let hits = walk.shortlist_rerank(0, &residual, x_slice, K, 8, kind);
    assert_eq!(hits.len(), K, "selection still produces output");
    assert_eq!(
        walk.shortlist_decline_count(),
        1,
        "no down norms → the shortlist must decline, not fabricate weights"
    );
}

// ── (d) default-off pins current behaviour ──────────────────────────────

/// With `shortlist_m` unset, `select_route_hits` is bit-identical to
/// the single-stage dispatch for every selector — and no shortlist
/// signal fires.
#[test]
fn default_off_select_route_hits_is_single_stage_bit_identical() {
    let weights = make_test_q4k_weights();
    let index = make_test_q4k_vindex_with_model_gate(&weights);
    let residual = residual_ramp(weights.hidden_size);
    let x_slice = residual.as_slice().unwrap();
    for kind in SCORED {
        let cfg = WalkFfnConfig::sparse(weights.num_layers, K).with_selector(kind);
        assert_eq!(cfg.shortlist_m, None, "shortlist must be opt-in");
        let walk = WalkFfn::from_config(&weights, &index, cfg).with_dispatch_trace();
        let routed = walk.select_route_hits(0, &residual, x_slice, K);
        let single = walk.single_stage_hits(0, &residual, K, kind);
        assert_eq!(
            routed, single,
            "{kind:?}: default-off must be the single-stage path"
        );
        assert_eq!(walk.shortlist_decline_count(), 0);
        assert!(walk
            .take_dispatch_trace()
            .iter()
            .all(|e| e.path != PATH_SHORTLIST_DECLINED));
    }
}

// ── cost pin ────────────────────────────────────────────────────────────

/// Delegating wrapper around the Q4K fixture whose full-projection
/// entry points — `gate_scores_batch`/`gate_scores_batch_backend` (the
/// O(N·d) gate projection) and `kquant_matmul_transb` (the O(N·d)
/// batched up projection behind `compute_full_up_scores`) — count every
/// call and, when `panic_on_full_projection` is set, panic. Everything
/// per-row (`gate_walk`'s production chain, `kquant_ffn_row_*`, the
/// lazy norm caches' `kquant_ffn_layer`) delegates untouched.
struct ProjectionCountingIndex<'i> {
    inner: &'i VectorIndex,
    full_projection_calls: AtomicU64,
    panic_on_full_projection: bool,
}

impl<'i> ProjectionCountingIndex<'i> {
    fn new(inner: &'i VectorIndex, panic_on_full_projection: bool) -> Self {
        Self {
            inner,
            full_projection_calls: AtomicU64::new(0),
            panic_on_full_projection,
        }
    }

    fn record_full_projection(&self, what: &str) {
        self.full_projection_calls.fetch_add(1, Ordering::Relaxed);
        assert!(
            !self.panic_on_full_projection,
            "two-stage selection must never pay a full projection — {what} was called"
        );
    }
}

impl GateLookup for ProjectionCountingIndex<'_> {
    fn gate_knn(&self, layer: usize, r: &Array1<f32>, top_k: usize) -> Vec<(usize, f32)> {
        self.inner.gate_knn(layer, r, top_k)
    }
    fn feature_meta(&self, layer: usize, feature: usize) -> Option<FeatureMeta> {
        GateLookup::feature_meta(self.inner, layer, feature)
    }
    fn num_features(&self, layer: usize) -> usize {
        GateLookup::num_features(self.inner, layer)
    }
    fn gate_walk(&self, layer: usize, r: &Array1<f32>, top_k: usize) -> Option<Vec<(usize, f32)>> {
        GateLookup::gate_walk(self.inner, layer, r, top_k)
    }
    fn gate_knn_q4(
        &self,
        layer: usize,
        r: &Array1<f32>,
        top_k: usize,
        backend: &dyn larql_compute::ComputeBackend,
    ) -> Option<Vec<(usize, f32)>> {
        GateLookup::gate_knn_q4(self.inner, layer, r, top_k, backend)
    }
    fn gate_scores_batch(&self, layer: usize, x: &Array2<f32>) -> Option<Array2<f32>> {
        self.record_full_projection("gate_scores_batch");
        GateLookup::gate_scores_batch(self.inner, layer, x)
    }
    fn gate_scores_batch_backend(
        &self,
        layer: usize,
        x: &Array2<f32>,
        backend: Option<&dyn larql_compute::ComputeBackend>,
    ) -> Option<Array2<f32>> {
        self.record_full_projection("gate_scores_batch_backend");
        GateLookup::gate_scores_batch_backend(self.inner, layer, x, backend)
    }
}

impl QuantizedFfnAccess for ProjectionCountingIndex<'_> {
    fn has_interleaved_kquant(&self) -> bool {
        self.inner.has_interleaved_kquant()
    }
    fn interleaved_kquant_layer_data(&self, layer: usize) -> Option<[(&[u8], &str); 3]> {
        self.inner.interleaved_kquant_layer_data(layer)
    }
    fn kquant_ffn_layer(&self, layer: usize, component: usize) -> Option<Arc<Vec<f32>>> {
        self.inner.kquant_ffn_layer(layer, component)
    }
    fn kquant_ffn_row_dot(&self, l: usize, c: usize, f: usize, x: &[f32]) -> Option<f32> {
        self.inner.kquant_ffn_row_dot(l, c, f, x)
    }
    fn kquant_ffn_row_into(&self, l: usize, c: usize, f: usize, out: &mut [f32]) -> bool {
        self.inner.kquant_ffn_row_into(l, c, f, out)
    }
    fn kquant_ffn_row_scaled_add(
        &self,
        l: usize,
        c: usize,
        f: usize,
        a: f32,
        out: &mut [f32],
    ) -> bool {
        self.inner.kquant_ffn_row_scaled_add(l, c, f, a, out)
    }
    fn kquant_ffn_row_scaled_add_via_cache(
        &self,
        l: usize,
        c: usize,
        f: usize,
        a: f32,
        out: &mut [f32],
    ) -> bool {
        self.inner
            .kquant_ffn_row_scaled_add_via_cache(l, c, f, a, out)
    }
    fn kquant_down_feature_scaled_add(&self, l: usize, f: usize, a: f32, out: &mut [f32]) -> bool {
        self.inner.kquant_down_feature_scaled_add(l, f, a, out)
    }
    fn has_down_features_kquant(&self) -> bool {
        self.inner.has_down_features_kquant()
    }
    fn kquant_matmul_transb(
        &self,
        layer: usize,
        component: usize,
        x: &[f32],
        x_rows: usize,
        backend: Option<&dyn larql_compute::ComputeBackend>,
    ) -> Option<Vec<f32>> {
        self.record_full_projection("kquant_matmul_transb");
        self.inner
            .kquant_matmul_transb(layer, component, x, x_rows, backend)
    }
}
impl PatchOverrides for ProjectionCountingIndex<'_> {}
impl NativeFfnAccess for ProjectionCountingIndex<'_> {}
impl Fp4FfnAccess for ProjectionCountingIndex<'_> {}

/// THE cost pin: a full forward through the sparse walk with two-stage
/// selection configured (the heaviest criterion) never touches a
/// full-projection entry point — the wrapper PANICS if one is called.
/// Stage 2's inputs all come from per-row primitives.
#[test]
fn cost_pin_two_stage_forward_never_calls_a_full_projection() {
    let weights = make_test_q4k_weights();
    let inner = make_test_q4k_vindex_with_model_gate(&weights);
    let index = ProjectionCountingIndex::new(&inner, true);
    let walk = WalkFfn::from_config(
        &weights,
        &index,
        WalkFfnConfig::sparse(weights.num_layers, K)
            .with_selector(FeatureSelector::ActXUpScoreXDownNorm)
            .with_shortlist_m(4 * K),
    );
    let x = Array2::from_shape_vec(
        (1, weights.hidden_size),
        residual_ramp(weights.hidden_size).to_vec(),
    )
    .unwrap();
    use crate::ffn::FfnBackend;
    let out = walk.forward(0, &x); // panics inside the wrapper on any full projection
    assert_eq!(out.shape(), &[1, weights.hidden_size]);
    assert!(out.iter().all(|v| v.is_finite()));
    assert_eq!(
        walk.shortlist_decline_count(),
        0,
        "the two-stage path must actually have run (no silent decline)"
    );
    assert_eq!(index.full_projection_calls.load(Ordering::Relaxed), 0);
}

/// Counter-half of the cost pin: the SAME wrapper with panics disabled
/// shows the default-off joint path really does pay full projections —
/// proving the pin above discriminates.
#[test]
fn cost_pin_counterpart_single_stage_joint_does_call_full_projections() {
    let weights = make_test_q4k_weights();
    let inner = make_test_q4k_vindex_with_model_gate(&weights);
    let index = ProjectionCountingIndex::new(&inner, false);
    let walk = WalkFfn::from_config(
        &weights,
        &index,
        WalkFfnConfig::sparse(weights.num_layers, K)
            .with_selector(FeatureSelector::ActXUpScoreXDownNorm),
    );
    let residual = residual_ramp(weights.hidden_size);
    let hits = walk.joint_gate_knn(0, &residual, K, FeatureSelector::ActXUpScoreXDownNorm);
    assert_eq!(hits.len(), K);
    assert!(
        index.full_projection_calls.load(Ordering::Relaxed) >= 2,
        "single-stage joint selection pays the gate AND up full projections"
    );
}

// ── planner + trace integration ─────────────────────────────────────────

/// The Sparse plan's reason records the shortlist width: actual = M,
/// cutoff = the layer's K, satisfied = two-stage will actually run.
#[test]
fn plan_reason_records_shortlist_m() {
    let weights = make_test_q4k_weights();
    let index = make_test_q4k_vindex_with_model_gate(&weights);
    let x = Array2::from_shape_vec(
        (1, weights.hidden_size),
        residual_ramp(weights.hidden_size).to_vec(),
    )
    .unwrap();
    let cases = [
        (FeatureSelector::GateXDownNorm, 4 * K, true),  // runs
        (FeatureSelector::GateXDownNorm, K - 1, false), // M < K → declines
        (FeatureSelector::Random, 4 * K, false),        // unscored → declines
    ];
    for (kind, m, satisfied) in cases {
        let walk = WalkFfn::from_config(
            &weights,
            &index,
            WalkFfnConfig::sparse(weights.num_layers, K)
                .with_selector(kind)
                .with_shortlist_m(m),
        );
        let plan = walk.plan_for(0, &x);
        assert_eq!(plan.rung(), PlanRung::Sparse);
        let check = plan
            .reason()
            .thresholds
            .iter()
            .find(|t| t.name == THRESHOLD_SHORTLIST_M)
            .expect("shortlist width must be recorded in the plan reason");
        assert_eq!((check.actual, check.cutoff), (m, K), "{kind:?}");
        assert_eq!(check.satisfied, satisfied, "{kind:?} M={m}");
        assert!(plan.reason().summary().contains(THRESHOLD_SHORTLIST_M));
    }
    // Default-off: no shortlist entry in the plan reason.
    let walk = WalkFfn::from_config(
        &weights,
        &index,
        WalkFfnConfig::sparse(weights.num_layers, K),
    );
    assert!(walk
        .plan_for(0, &x)
        .reason()
        .thresholds
        .iter()
        .all(|t| t.name != THRESHOLD_SHORTLIST_M));
}

/// The runtime trace's `rank` field reflects the FINAL rerank order:
/// the traced feature sequence at each position equals the two-stage
/// selection's own output order, rank 0 first.
#[test]
fn runtime_trace_rank_is_the_final_rerank_order() {
    let weights = make_test_q4k_weights();
    let index = make_test_q4k_vindex_with_model_gate(&weights);
    let kind = FeatureSelector::ActXUpScoreXDownNorm;
    let m = 4 * K;
    let walk = WalkFfn::from_config(
        &weights,
        &index,
        WalkFfnConfig::sparse(weights.num_layers, K)
            .with_selector(kind)
            .with_shortlist_m(m),
    )
    .with_trace();
    let residual = residual_ramp(weights.hidden_size);
    let x = Array2::from_shape_vec((1, weights.hidden_size), residual.to_vec()).unwrap();
    use crate::ffn::FfnBackend;
    walk.forward(0, &x);

    let expected = walk.shortlist_rerank(0, &residual, residual.as_slice().unwrap(), K, m, kind);
    let records = walk.take_runtime_trace();
    let rec = records
        .iter()
        .find(|r| r.layer == 0 && !r.features.is_empty())
        .expect("layer 0 must emit a per-feature trace record");
    assert_eq!(rec.features.len(), K);
    for (i, (traced, &(feat, _))) in rec.features.iter().zip(expected.iter()).enumerate() {
        assert_eq!(traced.rank, i, "rank must be the visit index");
        assert_eq!(
            traced.feature, feat,
            "trace order must be the final rerank order, not gate order"
        );
    }
    assert_eq!(walk.shortlist_decline_count(), 0);
}
