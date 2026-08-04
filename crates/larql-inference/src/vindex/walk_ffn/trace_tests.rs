//! Tests for `trace.rs` — runtime trace emission (2026-07-30 review,
//! item 17).
//!
//! The load-bearing test is
//! [`take_trace_reports_executed_route_not_gate_knn`]: it configures a
//! non-default route (precomputed pool) and asserts the trace carries
//! the EXECUTED features — the pre-item-17 `take_trace`, which re-ran
//! plain `gate_knn` post-hoc, reports the decoy KNN set instead and
//! fails this test.

use std::sync::Arc;

use ndarray::{Array1, Array2};

use super::sparse::{PATH_SPARSE_GATHER, PATH_SPARSE_SERIAL};
use crate::ffn::FfnBackend;
use crate::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
use crate::vindex::{WalkFfn, WalkFfnConfig};
use larql_vindex::{
    FeatureMeta, Fp4FfnAccess, GateLookup, NativeFfnAccess, PatchOverrides, QuantizedFfnAccess,
};

/// The precomputed route the walk EXECUTES in these tests.
const EXECUTED_POOL: [usize; 4] = [0, 1, 2, 3];
/// The decoy feature set the wrapper's `gate_knn` returns — what the
/// old post-hoc `take_trace` would have reported. Disjoint from
/// [`EXECUTED_POOL`] so the two answers can never be confused.
const GATE_KNN_DECOY: [usize; 4] = [10, 11, 12, 13];

/// Q4K-fixture wrapper with synthetic `FeatureMeta` and a deliberately
/// WRONG `gate_knn` (the decoy set). Storage access delegates to the
/// inner `VectorIndex`, so the executed walk is the real Q4K serial
/// path.
struct DecoyKnnIndex<'a> {
    inner: &'a larql_vindex::VectorIndex,
}

impl GateLookup for DecoyKnnIndex<'_> {
    fn gate_knn(&self, _layer: usize, _residual: &Array1<f32>, top_k: usize) -> Vec<(usize, f32)> {
        GATE_KNN_DECOY
            .iter()
            .take(top_k)
            .map(|&f| (f, 9.9))
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
    fn num_features(&self, layer: usize) -> usize {
        self.inner.num_features(layer)
    }
}
impl PatchOverrides for DecoyKnnIndex<'_> {}
impl NativeFfnAccess for DecoyKnnIndex<'_> {}
impl QuantizedFfnAccess for DecoyKnnIndex<'_> {
    fn has_interleaved_kquant(&self) -> bool {
        self.inner.has_interleaved_kquant()
    }
    fn interleaved_kquant_layer_data(&self, layer: usize) -> Option<[(&[u8], &str); 3]> {
        self.inner.interleaved_kquant_layer_data(layer)
    }
    fn kquant_ffn_layer(&self, layer: usize, component: usize) -> Option<Arc<Vec<f32>>> {
        self.inner.kquant_ffn_layer(layer, component)
    }
    fn kquant_ffn_row_dot(
        &self,
        layer: usize,
        component: usize,
        feat: usize,
        x: &[f32],
    ) -> Option<f32> {
        self.inner.kquant_ffn_row_dot(layer, component, feat, x)
    }
    fn kquant_ffn_row_scaled_add_via_cache(
        &self,
        layer: usize,
        component: usize,
        feat: usize,
        alpha: f32,
        out: &mut [f32],
    ) -> bool {
        self.inner
            .kquant_ffn_row_scaled_add_via_cache(layer, component, feat, alpha, out)
    }
    fn has_down_features_kquant(&self) -> bool {
        self.inner.has_down_features_kquant()
    }
    fn down_features_q4k_layer_data(&self, layer: usize) -> Option<(&[u8], &str, usize)> {
        self.inner.down_features_q4k_layer_data(layer)
    }
}
impl Fp4FfnAccess for DecoyKnnIndex<'_> {}

fn x(seq: usize, hidden: usize) -> Array2<f32> {
    Array2::from_shape_vec(
        (seq, hidden),
        (0..seq * hidden).map(|i| (i as f32 + 1.0) * 0.02).collect(),
    )
    .unwrap()
}

/// Precomputed-pool config: the executed route is [`EXECUTED_POOL`],
/// which plain `gate_knn` (the decoy) would never return.
fn pool_cfg(num_layers: usize) -> WalkFfnConfig {
    WalkFfnConfig::sparse(num_layers, EXECUTED_POOL.len())
        .with_pool_per_layer(Arc::new(vec![EXECUTED_POOL.to_vec(); num_layers]))
        .with_precomputed_routing(true)
}

/// THE item-17 test: with a non-default route configured, the trace's
/// features must equal the EXECUTED route. The old `take_trace`
/// re-ran `gate_knn` post-hoc and would report [`GATE_KNN_DECOY`].
#[test]
fn take_trace_reports_executed_route_not_gate_knn() {
    let weights = make_test_q4k_weights();
    let inner = make_test_q4k_vindex(&weights);
    let index = DecoyKnnIndex { inner: &inner };
    let ffn = WalkFfn::from_config(&weights, &index, pool_cfg(weights.num_layers)).with_trace();
    ffn.forward(0, &x(1, weights.hidden_size));

    let trace = ffn.take_trace();
    assert_eq!(trace.layers.len(), 1);
    let (layer, hits) = &trace.layers[0];
    assert_eq!(*layer, 0);
    let traced: Vec<usize> = hits.iter().map(|h| h.feature).collect();
    assert_eq!(
        traced,
        EXECUTED_POOL.to_vec(),
        "trace must carry the EXECUTED route, in executed order"
    );
    // The decoy is what the pre-item-17 post-hoc re-run reports.
    let decoy: Vec<usize> = index
        .gate_knn(0, &Array1::zeros(weights.hidden_size), EXECUTED_POOL.len())
        .iter()
        .map(|&(f, _)| f)
        .collect();
    assert_eq!(decoy, GATE_KNN_DECOY.to_vec());
    assert!(
        traced.iter().all(|f| !decoy.contains(f)),
        "executed route and gate_knn answer must be disjoint for this test to bite"
    );
    // Extended WalkHit fields carry the executed values.
    for (rank, hit) in hits.iter().enumerate() {
        assert_eq!(hit.rank, Some(rank));
        assert!(hit.activation.is_some(), "executed activation recorded");
        assert!(hit.up_score.is_some(), "gated arch records the up dot");
        assert!(hit.gate_score.is_finite());
        assert_eq!(hit.meta.top_token, format!("tok{}", hit.feature));
    }
}

/// The runtime records carry the executed kernel's own values: path
/// name, per-feature scores in visit order, and `‖out_row‖`.
#[test]
fn runtime_records_carry_scores_path_and_residual_norm() {
    let weights = make_test_q4k_weights();
    let inner = make_test_q4k_vindex(&weights);
    let index = DecoyKnnIndex { inner: &inner };
    let input = x(1, weights.hidden_size);

    let ffn = WalkFfn::from_config(&weights, &index, pool_cfg(weights.num_layers)).with_trace();
    let out = ffn.forward(0, &input);
    // Reference: an ordinary observed call on an untraced twin runs the
    // identical Record-mode ladder — bit-equal activations.
    let twin = WalkFfn::from_config(&weights, &index, pool_cfg(weights.num_layers));
    let (twin_out, twin_obs) = twin.forward_observed(0, &input);
    assert_eq!(out, twin_out, "tracing must not change the executed math");
    let twin_dense = twin_obs.into_dense().expect("sparse observation densifies");

    let records = ffn.take_runtime_trace();
    assert_eq!(records.len(), 1, "one position, one layer, one record");
    let rec = &records[0];
    assert_eq!((rec.layer, rec.position), (0, 0));
    assert_eq!(rec.path, PATH_SPARSE_SERIAL);
    let expected_norm = out.row(0).iter().map(|v| v * v).sum::<f32>().sqrt();
    assert_eq!(rec.residual_delta_norm, expected_norm);
    assert_eq!(rec.features.len(), EXECUTED_POOL.len());
    for (i, f) in rec.features.iter().enumerate() {
        assert_eq!(f.rank, i);
        assert_eq!(f.feature, EXECUTED_POOL[i]);
        assert_eq!(f.activation, twin_dense[[0, f.feature]]);
        assert!(f.gate_score.is_some() && f.up_score.is_some());
        assert_eq!(
            f.down_row_norm, None,
            "no selector built the norm cache — tracing must not build it"
        );
    }
    // Drained: a second take is empty.
    assert!(ffn.take_runtime_trace().is_empty());
}

/// `‖down_row‖` is served from the selector's lazy cache when (and
/// only when) it was already built — tracing never builds it.
#[test]
fn down_row_norm_served_from_prebuilt_cache_only() {
    let weights = make_test_q4k_weights();
    let inner = make_test_q4k_vindex(&weights);
    let index = DecoyKnnIndex { inner: &inner };
    let ffn = WalkFfn::from_config(&weights, &index, pool_cfg(weights.num_layers)).with_trace();
    // Build the cache the way a joint selector would.
    let norms = ffn
        .down_row_norms_pub(0)
        .expect("Q4K fixture yields down norms");
    ffn.forward(0, &x(1, weights.hidden_size));
    let records = ffn.take_runtime_trace();
    for f in &records[0].features {
        assert_eq!(
            f.down_row_norm,
            Some(norms[f.feature]),
            "prebuilt cache must serve the traced norm for feature {}",
            f.feature
        );
    }
}

/// Dense whole-layer paths have no per-feature selection data — the
/// trace emits the layer summary (path + residual-delta norm) with an
/// empty feature list rather than fabricating records.
#[test]
fn dense_paths_emit_summary_without_per_feature_records() {
    let weights = make_test_q4k_weights();
    let index = make_test_q4k_vindex(&weights);
    let cfg = WalkFfnConfig::dense(weights.num_layers);
    let ffn = WalkFfn::from_config(&weights, &index, cfg).with_trace();
    let out = ffn.forward(0, &x(2, weights.hidden_size));

    let records = ffn.take_runtime_trace();
    assert_eq!(records.len(), 2, "one record per position");
    for (s, rec) in records.iter().enumerate() {
        assert_eq!(rec.position, s);
        assert_eq!(rec.path, "interleaved_kquant:native");
        assert!(rec.features.is_empty(), "dense path declines per-feature");
        let expected = out.row(s).iter().map(|v| v * v).sum::<f32>().sqrt();
        assert_eq!(rec.residual_delta_norm, expected);
    }
    // The WalkTrace view keeps the layer entry with empty hits.
    let ffn = WalkFfn::from_config(&weights, &index, WalkFfnConfig::dense(weights.num_layers))
        .with_trace();
    ffn.forward(0, &x(1, weights.hidden_size));
    let trace = ffn.take_trace();
    assert_eq!(trace.layers.len(), 1);
    assert!(trace.layers[0].1.is_empty());
}

/// `take_trace` reports the LAST position per layer call (historic
/// last-row semantics) and keeps one entry per call across steps.
#[test]
fn take_trace_uses_last_position_and_groups_calls() {
    let weights = make_test_q4k_weights();
    let inner = make_test_q4k_vindex(&weights);
    let index = DecoyKnnIndex { inner: &inner };
    let ffn = WalkFfn::from_config(&weights, &index, pool_cfg(weights.num_layers)).with_trace();
    let input = x(2, weights.hidden_size);
    ffn.forward(0, &input);

    // Reference activations for BOTH positions.
    let twin = WalkFfn::from_config(&weights, &index, pool_cfg(weights.num_layers));
    let dense = twin
        .forward_observed(0, &input)
        .1
        .into_dense()
        .expect("sparse observation densifies");

    let trace = ffn.take_trace();
    assert_eq!(trace.layers.len(), 1, "one layer call, one entry");
    let hits = &trace.layers[0].1;
    assert!(!hits.is_empty());
    for hit in hits {
        assert_eq!(
            hit.activation,
            Some(dense[[1, hit.feature]]),
            "WalkTrace view must report the LAST position's execution"
        );
    }

    // Two decode steps on the same layer → two per-call entries.
    let step = x(1, weights.hidden_size);
    ffn.forward(0, &step);
    ffn.forward(0, &step);
    let trace = ffn.take_trace();
    assert_eq!(trace.layers.len(), 2);
    assert!(trace.layers.iter().all(|(l, _)| *l == 0));
}

/// Runtime records keep every executed feature; the `WalkTrace` view
/// (whose `WalkHit.meta` is mandatory) drops meta-less features.
#[test]
fn walktrace_view_drops_metaless_features_records_keep_them() {
    let weights = make_test_q4k_weights();
    // Plain fixture: `feature_meta` is None for every feature.
    let index = make_test_q4k_vindex(&weights);
    let ffn = WalkFfn::from_config(&weights, &index, pool_cfg(weights.num_layers)).with_trace();
    ffn.forward(0, &x(1, weights.hidden_size));
    let records = ffn.take_runtime_trace();
    assert_eq!(records[0].features.len(), EXECUTED_POOL.len());

    let ffn = WalkFfn::from_config(&weights, &index, pool_cfg(weights.num_layers)).with_trace();
    ffn.forward(0, &x(1, weights.hidden_size));
    let trace = ffn.take_trace();
    assert_eq!(trace.layers.len(), 1);
    assert!(
        trace.layers[0].1.is_empty(),
        "meta-less features drop from the compat view only"
    );
}

/// Without `with_trace` nothing is recorded and the observe upgrade
/// never happens (the hot path stays Skip).
#[test]
fn disabled_trace_records_nothing() {
    let weights = make_test_q4k_weights();
    let inner = make_test_q4k_vindex(&weights);
    let index = DecoyKnnIndex { inner: &inner };
    let ffn = WalkFfn::from_config(&weights, &index, pool_cfg(weights.num_layers));
    ffn.forward(0, &x(1, weights.hidden_size));
    assert!(ffn.take_runtime_trace().is_empty());
    assert!(ffn.take_trace().layers.is_empty());
}

/// The gather kernel labels its positions: on a sidecar-equipped route
/// the traced path is `sparse:gather_q4k` and the per-feature scores
/// are the fused kernels' own dots.
#[test]
fn gather_positions_are_labelled_with_the_gather_kernel() {
    use crate::test_utils::attach_down_features_q4k_to_test_vindex;
    let weights = make_test_q4k_weights();
    let mut inner = make_test_q4k_vindex(&weights);
    attach_down_features_q4k_to_test_vindex(&weights, &mut inner);
    let index = DecoyKnnIndex { inner: &inner };
    // Route = whole layer (== GATHER_MIN_FEATURES on this fixture).
    let pool: Vec<usize> = (0..weights.intermediate_size).collect();
    let cfg = WalkFfnConfig::sparse(weights.num_layers, pool.len())
        .with_pool_per_layer(Arc::new(vec![pool.clone(); weights.num_layers]))
        .with_precomputed_routing(true);
    let ffn = WalkFfn::from_config(&weights, &index, cfg).with_trace();
    ffn.forward(0, &x(1, weights.hidden_size));
    let records = ffn.take_runtime_trace();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].path, PATH_SPARSE_GATHER);
    assert_eq!(records[0].features.len(), pool.len());
    assert!(records[0]
        .features
        .iter()
        .all(|f| f.gate_score.is_some() && f.up_score.is_some()));
}
