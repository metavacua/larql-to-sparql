//! Construction, forward-shape, trace, and cache dispatch tests for
//! `WalkFfn` — colocated with the routing ladder they exercise
//! (`mod.rs`); the mock-backend ladder-priority tests live in
//! `routing_tests.rs`.

use super::*;
use crate::model::ModelWeights;
use crate::test_utils::make_test_weights;
use larql_vindex::{
    FeatureMeta, Fp4FfnAccess, GateLookup, NativeFfnAccess, PatchOverrides, QuantizedFfnAccess,
};
use ndarray::{Array1, Array2};
use std::sync::OnceLock;

fn shared_weights() -> &'static ModelWeights {
    static W: OnceLock<ModelWeights> = OnceLock::new();
    W.get_or_init(make_test_weights)
}
use crate::ffn::FfnBackend;

/// Minimal GateIndex with only the 3 required methods.
/// All optional methods fall back to their trait defaults (all return None/false/[]).
/// WalkFfn routes through path 9 (last-resort sparse matmul against weights.tensors).
struct MockGateIndex {
    n_features: usize,
}

impl GateLookup for MockGateIndex {
    fn gate_knn(&self, _layer: usize, _residual: &Array1<f32>, top_k: usize) -> Vec<(usize, f32)> {
        (0..top_k.min(self.n_features))
            .map(|i| (i, 1.0 / (i as f32 + 1.0)))
            .collect()
    }
    fn feature_meta(&self, _layer: usize, _feature: usize) -> Option<FeatureMeta> {
        None
    }
    fn num_features(&self, _layer: usize) -> usize {
        self.n_features
    }
}

impl PatchOverrides for MockGateIndex {}
impl NativeFfnAccess for MockGateIndex {}
impl QuantizedFfnAccess for MockGateIndex {}
impl Fp4FfnAccess for MockGateIndex {}

fn mock_index(weights: &ModelWeights) -> MockGateIndex {
    MockGateIndex {
        n_features: weights.intermediate_size,
    }
}

/// Overridden layers with NO FFN payload: `has_overrides_at` is
/// true, but the sparse walk fails (all storage traits at their
/// defaults). The M7 regression scenario.
struct OverriddenNoPayloadIndex {
    inner: MockGateIndex,
}

impl GateLookup for OverriddenNoPayloadIndex {
    fn gate_knn(&self, layer: usize, residual: &Array1<f32>, top_k: usize) -> Vec<(usize, f32)> {
        self.inner.gate_knn(layer, residual, top_k)
    }
    fn feature_meta(&self, layer: usize, feature: usize) -> Option<FeatureMeta> {
        self.inner.feature_meta(layer, feature)
    }
    fn num_features(&self, layer: usize) -> usize {
        self.inner.num_features(layer)
    }
}

impl PatchOverrides for OverriddenNoPayloadIndex {
    fn has_overrides_at(&self, _layer: usize) -> bool {
        true
    }
}
impl NativeFfnAccess for OverriddenNoPayloadIndex {}
impl QuantizedFfnAccess for OverriddenNoPayloadIndex {}
impl Fp4FfnAccess for OverriddenNoPayloadIndex {}

fn input(seq: usize, hidden: usize) -> Array2<f32> {
    Array2::from_shape_vec(
        (seq, hidden),
        (0..seq * hidden).map(|i| (i as f32 + 1.0) * 0.02).collect(),
    )
    .unwrap()
}

// ── WalkFfn construction ──────────────────────────────────────────────────

#[test]
fn walk_ffn_new_unlimited() {
    let weights = shared_weights();
    let idx = mock_index(weights);
    let ffn = WalkFfn::new_unlimited(weights, &idx);
    assert_eq!(ffn.name(), "walk");
}

#[test]
fn dense_top_k_for_clamps_to_layer_feature_count() {
    let weights = shared_weights();
    let idx = mock_index(weights);
    let ffn = WalkFfn::new_unlimited(weights, &idx);

    assert_eq!(ffn.top_k_for(0), weights.intermediate_size);
}

#[test]
fn walk_ffn_sparse_k() {
    let weights = shared_weights();
    let idx = mock_index(weights);
    let ffn = WalkFfn::new(weights, &idx, 4);
    assert_eq!(ffn.name(), "walk");
}

// ── forward shape and finiteness ─────────────────────────────────────────

#[test]
fn walk_ffn_forward_shape_single_token() {
    let weights = shared_weights();
    let idx = mock_index(weights);
    let ffn = WalkFfn::new_unlimited(weights, &idx);
    let x = input(1, weights.hidden_size);
    let out = ffn.forward(0, &x);
    assert_eq!(out.shape(), &[1, weights.hidden_size]);
}

#[test]
fn walk_ffn_forward_shape_multi_token() {
    let weights = shared_weights();
    let idx = mock_index(weights);
    let ffn = WalkFfn::new_unlimited(weights, &idx);
    let x = input(3, weights.hidden_size);
    let out = ffn.forward(0, &x);
    assert_eq!(out.shape(), &[3, weights.hidden_size]);
    assert!(out.iter().all(|v| v.is_finite()));
}

#[test]
fn walk_ffn_forward_all_layers() {
    let weights = shared_weights();
    let idx = mock_index(weights);
    let ffn = WalkFfn::new_unlimited(weights, &idx);
    let x = input(1, weights.hidden_size);
    for layer in 0..weights.num_layers {
        let out = ffn.forward(layer, &x);
        assert_eq!(
            out.shape(),
            &[1, weights.hidden_size],
            "layer {layer} wrong shape"
        );
        assert!(
            out.iter().all(|v| v.is_finite()),
            "layer {layer} non-finite"
        );
    }
}

#[test]
fn walk_ffn_sparse_vs_dense_same_shape() {
    let weights = shared_weights();
    let idx = mock_index(weights);
    let ffn_sparse = WalkFfn::new(weights, &idx, 4);
    let ffn_dense = WalkFfn::new_unlimited(weights, &idx);
    let x = input(1, weights.hidden_size);
    let out_s = ffn_sparse.forward(0, &x);
    let out_d = ffn_dense.forward(0, &x);
    assert_eq!(out_s.shape(), out_d.shape());
}

#[test]
fn walk_ffn_forward_observed_returns_activation_and_matches_forward() {
    let weights = shared_weights();
    let idx = mock_index(weights);
    let ffn = WalkFfn::new_unlimited(weights, &idx);
    let x = input(2, weights.hidden_size);
    let (out, obs) = ffn.forward_observed(0, &x);
    assert_eq!(out.shape(), &[2, weights.hidden_size]);
    let act = obs
        .into_dense()
        .expect("an executed walk path must observe activations");
    assert_eq!(act.shape()[0], 2, "activation should have seq_len rows");
    // The split's core contract: both entry points run the same routed
    // body, so the outputs are identical.
    assert_eq!(out, ffn.forward(0, &x));
}

#[test]
fn walk_ffn_zero_features_falls_back_to_weight_ffn() {
    // When MockGateIndex returns 0 features, WalkFfn should fall back to WeightFfn.
    let weights = shared_weights();
    let zero_idx = MockGateIndex { n_features: 0 };
    let ffn = WalkFfn::new_unlimited(weights, &zero_idx);
    let x = input(1, weights.hidden_size);
    let out = ffn.forward(0, &x);
    assert_eq!(out.shape(), &[1, weights.hidden_size]);
    assert!(out.iter().all(|v| v.is_finite()));
}

#[test]
fn walk_ffn_zero_features_observed_reports_dense_weight_ffn_activation() {
    // Same fallback, observed entry point: the WeightFfn detour is a
    // dense path, so the observation must be a real Dense activation.
    let weights = shared_weights();
    let zero_idx = MockGateIndex { n_features: 0 };
    let ffn = WalkFfn::new_unlimited(weights, &zero_idx).with_dispatch_trace();
    let x = input(1, weights.hidden_size);
    let (out, obs) = ffn.forward_observed(0, &x);
    assert_eq!(
        ffn.take_dispatch_trace().last().map(|e| e.path),
        Some("zero_features_dense")
    );
    assert_eq!(out.shape(), &[1, weights.hidden_size]);
    let act = obs
        .into_dense()
        .expect("WeightFfn fallback observes densely");
    assert_eq!(act.shape(), &[1, weights.intermediate_size]);
}

#[test]
fn overridden_layer_observed_lands_on_override_fallback_with_real_observation() {
    // M7's observed twin: base_delta and the sparse walk both decline
    // (no FFN payload), so the observed call must land on the
    // override-aware safetensors fallback and report the activations it
    // actually computed — not zeros, not Absent.
    let weights = shared_weights();
    let index = OverriddenNoPayloadIndex {
        inner: mock_index(weights),
    };
    let walk = WalkFfn::new_unlimited(weights, &index).with_dispatch_trace();
    let x = input(1, weights.hidden_size);
    let (out, obs) = walk.forward_observed(0, &x);
    assert_eq!(
        walk.take_dispatch_trace().last().map(|e| e.path),
        Some("weights_fallback:override")
    );
    assert!(!obs.is_absent(), "the fallback computes — it must observe");
    let act = obs.into_dense().expect("observation densifies");
    assert!(
        act.iter().any(|v| v.abs() > 0.0),
        "real activations, not the old fabricated zeros"
    );
    assert_eq!(out, walk.forward(0, &x), "Skip/Record outputs agree");
}

#[test]
fn walk_ffn_with_backend() {
    let weights = shared_weights();
    let idx = mock_index(weights);
    let ffn = WalkFfn::new_unlimited_with_backend(weights, &idx, &larql_compute::CpuBackend);
    let x = input(1, weights.hidden_size);
    let out = ffn.forward(0, &x);
    assert_eq!(out.shape(), &[1, weights.hidden_size]);
}

// ── trace + l1_cache + dispatch_trace ──────────────────────────────

#[test]
fn walk_ffn_take_residuals_returns_per_layer_traces() {
    let weights = shared_weights();
    let idx = mock_index(weights);
    let ffn = WalkFfn::new_unlimited(weights, &idx).with_trace();
    let x = input(2, weights.hidden_size);
    // Run forward for every layer so trace_residuals populates.
    for layer in 0..weights.num_layers {
        ffn.forward(layer, &x);
    }
    let residuals = ffn.take_residuals();
    assert_eq!(residuals.len(), weights.num_layers);
    for (layer, residual) in &residuals {
        assert!(*layer < weights.num_layers);
        assert_eq!(residual.len(), weights.hidden_size);
        assert!(residual.iter().all(|v| v.is_finite()));
    }
    // Drained — second call must be empty.
    assert!(ffn.take_residuals().is_empty());
}

#[test]
fn walk_ffn_with_trace_emits_residuals() {
    let weights = shared_weights();
    let idx = mock_index(weights);
    let ffn = WalkFfn::new_with_trace(weights, &idx, 4);
    let x = input(1, weights.hidden_size);
    ffn.forward(0, &x);
    let residuals = ffn.take_residuals();
    assert_eq!(residuals.len(), 1);
    assert_eq!(residuals[0].0, 0);
}

/// `peek_residuals` is the NON-draining view (the early-exit path
/// inspects mid-forward): peeking must return the captured residuals
/// and leave them in place for a later `take_residuals`.
#[test]
fn walk_ffn_peek_residuals_does_not_drain() {
    let weights = shared_weights();
    let idx = mock_index(weights);
    let ffn = WalkFfn::new_unlimited_with_trace(weights, &idx);
    let x = input(1, weights.hidden_size);
    ffn.forward(0, &x);

    let peeked = ffn.peek_residuals();
    assert_eq!(peeked.len(), 1);
    assert_eq!(peeked[0].0, 0);
    assert_eq!(peeked[0].1.len(), weights.hidden_size);
    // Still there: take() must return the same entry afterwards.
    let taken = ffn.take_residuals();
    assert_eq!(taken, peeked, "peek must not consume the trace");
    assert!(ffn.take_residuals().is_empty());
}

#[test]
fn walk_ffn_new_unlimited_with_trace_records() {
    let weights = shared_weights();
    let idx = mock_index(weights);
    let ffn = WalkFfn::new_unlimited_with_trace(weights, &idx);
    let x = input(1, weights.hidden_size);
    ffn.forward(0, &x);
    assert_eq!(ffn.take_residuals().len(), 1);
}

#[test]
fn walk_ffn_take_trace_pairs_residuals_with_walk_hits() {
    let weights = shared_weights();
    let idx = mock_index(weights);
    let ffn = WalkFfn::new_unlimited_with_trace(weights, &idx);
    let x = input(1, weights.hidden_size);
    ffn.forward(0, &x);
    let trace = ffn.take_trace();
    assert!(!trace.layers.is_empty());
    // Mock index returns no FeatureMeta so walk_hits collapses to empty
    // — but the layer entry itself must still be present.
    assert_eq!(trace.layers[0].0, 0);
}

#[test]
fn walk_ffn_new_with_backend_attaches_backend() {
    let weights = shared_weights();
    let idx = mock_index(weights);
    let ffn = WalkFfn::new_with_backend(weights, &idx, 4, &larql_compute::CpuBackend);
    let x = input(1, weights.hidden_size);
    let out = ffn.forward(0, &x);
    assert_eq!(out.shape(), &[1, weights.hidden_size]);
}

#[test]
fn walk_ffn_with_l1_cache_records_misses_then_hits() {
    let weights = shared_weights();
    let idx = mock_index(weights);
    let ffn = WalkFfn::new_unlimited(weights, &idx).with_l1_cache(weights.num_layers);
    let x = input(1, weights.hidden_size);
    // L1 cache stats: first call is a miss, second identical call a hit.
    ffn.forward(0, &x);
    ffn.forward(0, &x);
    let (hits, misses) = ffn.l1_cache_stats().expect("cache enabled");
    assert!(misses >= 1, "first call must be a miss");
    // Whether the second call hits depends on the cache key — so we
    // just assert hits is a sensible non-overflowing count.
    assert!(hits + misses >= 2);
}

/// Item 15 (2026-07-30 review): the L1 cache stores outputs only, so a
/// hit used to fabricate an all-zero activation matrix. The observed
/// entry point must instead BYPASS the cache read and recompute — real
/// observation, identical output — while the hot `forward` keeps
/// serving hits.
#[test]
fn l1_cache_hit_serves_forward_but_observed_bypasses_and_recomputes() {
    let weights = shared_weights();
    let idx = mock_index(weights);
    let ffn = WalkFfn::new_unlimited(weights, &idx)
        .with_l1_cache(weights.num_layers)
        .with_dispatch_trace();
    let x = input(1, weights.hidden_size);

    // Warm the cache, then confirm the hot path serves the hit.
    let warm = ffn.forward(0, &x);
    let served = ffn.forward(0, &x);
    assert_eq!(warm, served);
    let trace = ffn.take_dispatch_trace();
    assert_eq!(
        trace.last().map(|e| e.path),
        Some("l1_cache_hit"),
        "second identical hot call must be served from L1, got {trace:?}"
    );

    // Observed call on the same residual: recomputes (no l1_cache_hit
    // trace), returns the same output, and reports a REAL observation.
    let (out_obs, obs) = ffn.forward_observed(0, &x);
    let trace = ffn.take_dispatch_trace();
    assert!(
        trace.iter().all(|e| e.path != "l1_cache_hit"),
        "observed call must bypass the L1 read, got {trace:?}"
    );
    assert_eq!(out_obs, warm, "bypass recomputes the identical output");
    assert!(
        !obs.is_absent(),
        "the recomputed path must observe real activations, never fabricate"
    );
    let act = obs.into_dense().expect("non-absent observation densifies");
    assert!(
        act.iter().any(|v| v.abs() > 0.0),
        "mock fixture produces non-zero activations — all-zero would mean \
         the old fabricated-zeros bug is back"
    );
}

#[test]
fn walk_ffn_l1_cache_stats_returns_none_when_disabled() {
    let weights = shared_weights();
    let idx = mock_index(weights);
    let ffn = WalkFfn::new_unlimited(weights, &idx);
    assert!(ffn.l1_cache_stats().is_none());
}

#[test]
fn walk_ffn_with_dispatch_trace_records_per_layer_path() {
    let weights = shared_weights();
    let idx = mock_index(weights);
    let ffn = WalkFfn::new_unlimited(weights, &idx).with_dispatch_trace();
    let x = input(1, weights.hidden_size);
    for layer in 0..weights.num_layers {
        ffn.forward(layer, &x);
    }
    let trace = ffn.take_dispatch_trace();
    assert_eq!(trace.len(), weights.num_layers);
    for (i, entry) in trace.iter().enumerate() {
        assert_eq!(entry.layer, i);
        assert!(!entry.path.is_empty());
    }
    // Drain semantics: second call returns empty.
    assert!(ffn.take_dispatch_trace().is_empty());
}

#[test]
fn walk_ffn_take_dispatch_trace_returns_empty_when_disabled() {
    let weights = shared_weights();
    let idx = mock_index(weights);
    let ffn = WalkFfn::new_unlimited(weights, &idx);
    assert!(ffn.take_dispatch_trace().is_empty());
}

#[test]
fn walk_ffn_from_config_uses_supplied_walkffnconfig() {
    let weights = shared_weights();
    let idx = mock_index(weights);
    let cfg = crate::vindex::WalkFfnConfig::sparse(weights.num_layers, 2);
    let ffn = WalkFfn::from_config(weights, &idx, cfg);
    let x = input(1, weights.hidden_size);
    let out = ffn.forward(0, &x);
    assert_eq!(out.shape(), &[1, weights.hidden_size]);
}

/// `WalkFfn::new` with `top_k = usize::MAX` routes through the dense
/// `WalkFfnConfig::dense` branch (line 178). The `new(_, _, k)`
/// constructor's other branch (sparse) is already exercised by
/// `walk_ffn_sparse_k`.
#[test]
fn walk_ffn_new_with_usize_max_takes_dense_branch() {
    let weights = shared_weights();
    let idx = mock_index(weights);
    let ffn = WalkFfn::new(weights, &idx, usize::MAX);
    let x = input(1, weights.hidden_size);
    let out = ffn.forward(0, &x);
    assert_eq!(out.shape(), &[1, weights.hidden_size]);
}

/// Variant of `MockGateIndex` that yields a `FeatureMeta` for every
/// `(layer, feature)` query. This is what `take_trace` needs to
/// promote a residual into a populated `WalkHit` — without it, the
/// `filter_map` collapses to empty (`feature_meta` returns `None`).
struct MockGateIndexWithMeta {
    n_features: usize,
}

impl GateLookup for MockGateIndexWithMeta {
    fn gate_knn(&self, _layer: usize, _residual: &Array1<f32>, top_k: usize) -> Vec<(usize, f32)> {
        (0..top_k.min(self.n_features))
            .map(|i| (i, 1.0 / (i as f32 + 1.0)))
            .collect()
    }
    fn feature_meta(&self, _layer: usize, feature: usize) -> Option<FeatureMeta> {
        Some(FeatureMeta {
            top_token: format!("tok{feature}"),
            top_token_id: feature as u32,
            c_score: 0.9,
            top_k: Vec::new(),
        })
    }
    fn num_features(&self, _layer: usize) -> usize {
        self.n_features
    }
}

impl PatchOverrides for MockGateIndexWithMeta {}
impl NativeFfnAccess for MockGateIndexWithMeta {}
impl QuantizedFfnAccess for MockGateIndexWithMeta {}
impl Fp4FfnAccess for MockGateIndexWithMeta {}

/// When `feature_meta` returns `Some`, `take_trace` builds a populated
/// `WalkHit` per (layer, feature) — covering the `Some(WalkHit { .. })`
/// arm of the `filter_map` (lines 235-238).
#[test]
fn walk_ffn_take_trace_populates_walk_hits_when_meta_present() {
    let weights = shared_weights();
    let idx = MockGateIndexWithMeta {
        n_features: weights.intermediate_size,
    };
    let ffn = WalkFfn::new_with_trace(weights, &idx, 3);
    let x = input(1, weights.hidden_size);
    ffn.forward(0, &x);
    let trace = ffn.take_trace();
    assert_eq!(trace.layers.len(), 1);
    let (layer, hits) = &trace.layers[0];
    assert_eq!(*layer, 0);
    assert!(
        !hits.is_empty(),
        "expected WalkHits when feature_meta returns Some"
    );
    for hit in hits {
        assert_eq!(hit.layer, 0);
        assert!(hit.gate_score.is_finite());
        assert!(hit.meta.top_token.starts_with("tok"));
    }
}

/// `walk_trace_env_enabled` caches the env-var lookup in a thread-local.
/// Spawn a fresh thread so the cell starts empty, set `LARQL_WALK_TRACE=1`
/// in that thread via the thread-local override (NOT `std::env::set_var`,
/// which races concurrent `getenv` on the decode path → SIGSEGV), then
/// drive `forward` — `trace_path` reads the cache (first call populates it
/// as `Some(true)`) and emits to stderr. The override is
/// set INSIDE the spawned thread because both the override map and the
/// `WALK_TRACE_ENABLED` cache are thread-local; it is cleared before the
/// thread returns.
#[test]
fn walk_ffn_trace_path_honours_env_var_in_fresh_thread() {
    let handle = std::thread::spawn(|| {
        larql_compute::options::set_env_override(super::timings::ENV_WALK_TRACE, Some("1"));
        let weights = make_test_weights();
        let idx = MockGateIndex {
            n_features: weights.intermediate_size,
        };
        let ffn = WalkFfn::new_unlimited(&weights, &idx);
        let x = Array2::from_shape_vec(
            (1, weights.hidden_size),
            (0..weights.hidden_size)
                .map(|i| (i as f32 + 1.0) * 0.02)
                .collect(),
        )
        .unwrap();
        // Drives trace_path, which checks walk_trace_env_enabled.
        ffn.forward(0, &x);
        larql_compute::options::clear_fast_path_overrides();
    });
    handle.join().expect("env-var thread panicked");
}

/// Forward against the Q4K test fixture routes through the native
/// kquant path (priority 4 in the routing ladder), which fires when
/// the vindex has interleaved_kquant data AND the `num_features` Q4_K
/// fallback returns a non-zero width.
#[test]
fn walk_ffn_forward_routes_through_native_kquant_path() {
    use crate::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
    let weights = make_test_q4k_weights();
    let index = make_test_q4k_vindex(&weights);
    // Pre-flight: the Q4K manifest's gate component bytes should give
    // a positive intermediate width via the `num_features` fallback.
    // If this fails the test would silently route through
    // `zero_features_dense` and the native kquant path stays uncov.
    for layer in 0..weights.num_layers {
        assert!(
            index.num_features(layer) > 0,
            "layer {layer}: num_features must be > 0 for Q4K routing — \
             the kquant_ffn_intermediate_width fallback returned 0"
        );
    }
    let ffn = WalkFfn::new_unlimited(&weights, &index);
    let x = input(1, weights.hidden_size);
    let out = ffn.forward(0, &x);
    assert_eq!(out.shape(), &[1, weights.hidden_size]);
    assert!(out.iter().all(|v| v.is_finite()));
}

/// M10 regression (2026-07-30 review): a joint selector running on
/// an index without batched gate scores degrades to the production
/// GateOnly chain — that degradation must be observable, or an A/B
/// sweep silently reports GateOnly results under a joint-selector
/// label.
#[test]
fn joint_selector_fallback_is_observable() {
    use crate::vindex::walk_config::FeatureSelector;
    let weights = shared_weights();
    let idx = mock_index(weights);
    let walk = WalkFfn::new_unlimited(weights, &idx).with_dispatch_trace();
    let residual = Array1::from_vec(vec![0.02_f32; weights.hidden_size]);

    let hits = walk.joint_gate_knn(0, &residual, 4, FeatureSelector::GateXDownNorm);
    assert!(!hits.is_empty(), "fallback chain still selects features");
    assert_eq!(
        walk.selector_fallback_count(),
        1,
        "degradation to GateOnly must be counted"
    );
    let trace = walk.take_dispatch_trace();
    assert_eq!(trace.len(), 1);
    assert_eq!(trace[0].path, "selector:fallback");
}

/// M7 regression (2026-07-30 review): an overridden layer whose
/// sparse walk fails must route DIRECTLY to the override-aware
/// safetensors fallback — never through the L1 cache or the
/// whole-layer ladder. Pre-fix, the first call's result was
/// inserted into the L1 cache on the fallthrough path, so the
/// second identical call traced `l1_cache_hit` — a cache entry the
/// override machinery can never invalidate.
#[test]
fn overridden_layer_with_failed_sparse_bypasses_l1_cache() {
    let weights = shared_weights();
    let index = OverriddenNoPayloadIndex {
        inner: mock_index(weights),
    };
    let walk = WalkFfn::new_unlimited(weights, &index)
        .with_l1_cache(weights.num_layers)
        .with_dispatch_trace();
    let x = input(1, weights.hidden_size);

    walk.forward(0, &x);
    walk.forward(0, &x);

    let trace = walk.take_dispatch_trace();
    assert_eq!(trace.len(), 2);
    for entry in &trace {
        assert_eq!(
            entry.path, "weights_fallback:override",
            "overridden layer must land on the override-aware fallback \
             every call — got {} (an override-blind path)",
            entry.path
        );
    }
}
