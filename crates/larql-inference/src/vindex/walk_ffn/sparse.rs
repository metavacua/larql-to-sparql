//! Sparse walk path — zero matrix multiplications.
//!
//! The hot path for FFN inference on the LARQL vindex. For each position:
//!
//!   1. `gate_knn` → top-K features (HNSW / batched brute-force / gate-walk)
//!   2. For each feature:
//!      - `up_score  = dot(up_row(feat), x)`         via unified ffn_row_dot
//!      - `activated = silu(gate_score) * up_score`   (GEGLU)
//!      - `out      += activated * down_row(feat)`   via unified ffn_row_scaled_add
//!
//! The "unified" accessors in the `GateIndex` trait dispatch through
//! FP4 → native f32 → Q4K backends in priority order, so this single
//! function is **format-blind** — the same code path serves FP4, Q4K,
//! and native f32 vindexes. Adding a new storage format doesn't touch
//! this file.
//!
//! Four specialisations are layered on top, each in its own module:
//!
//! - **Full-K gemv fast path** (`sparse_gemv.rs`): when K ≥ 80% of
//!   num_features, the per-feature loop is mathematically equivalent to
//!   three dense matmuls — route through BLAS gemm / Q4K direct matmul.
//! - **Route selection** (`sparse_route.rs`): the per-position decision
//!   tree picking WHICH features the walk visits (cell router / pools /
//!   selector dispatch).
//! - **Gather-contiguous Q4K kernel** (`sparse_gather.rs`): known-pool
//!   routes gather gate/up/down bytes contiguous and run fused kernels.
//! - **Parallel Q4K down-cache path** (`sparse_parallel.rs`): for
//!   medium-K on Q4K-only vindexes, cache the dequantised down layer
//!   and parallelise feature chunks over rayon.
//!
//! This file keeps the entry point (`walk_ffn_sparse`), its preflight,
//! and the serial per-feature loop — the canonical correctness
//! baseline; it always works because `ffn_row_*` always has *some*
//! backend.

use ndarray::Array2;

use super::helpers::hits_len_ge_intermediate;
use super::observe::Observe;
use super::thresholds::GATHER_MIN_FEATURES;
use super::WalkFfn;
use crate::ffn::{FfnActivations, SparseActivations};
use crate::vindex::walk_config::FeatureSelector;
use larql_vindex::{FFN_DOWN, FFN_UP};

/// Dispatch-trace / per-position kernel label for the serial
/// per-feature loop — single source for `trace_path` and the runtime
/// trace's observation labels (2026-07-30 review, item 17).
pub(super) const PATH_SPARSE_SERIAL: &str = "sparse:serial";
/// Same, for the gather-contiguous Q4K kernel (`sparse_gather.rs`).
pub(super) const PATH_SPARSE_GATHER: &str = "sparse:gather_q4k";

impl<'a> WalkFfn<'a> {
    /// Sparse walk FFN — see module docs.
    ///
    /// In `Observe::Record` mode the walk emits exactly the `(feature,
    /// activation)` pairs it computes (full-K gemv reports its intrinsic
    /// dense matrix); in `Skip` mode no activation buffer exists at all
    /// — the returned observation is `None`.
    pub(super) fn walk_ffn_sparse(
        &self,
        layer: usize,
        x: &Array2<f32>,
        observe: Observe,
    ) -> Option<(Array2<f32>, Option<FfnActivations>)> {
        let hidden = x.shape()[1];
        let seq_len = x.shape()[0];
        let intermediate = self.index.num_features(layer);

        // Prefer native f32 mmap (zero-copy). When no native mmap is
        // available we still run — the inner loops dispatch per-row
        // through `ffn_row_dot` / `ffn_row_scaled_add`, which the
        // GateIndex trait routes to FP4 or Q4K or last-resort native
        // as appropriate. The only thing we can't do with neither
        // native f32 mmap, Q4K storage, nor FP4 storage is the serial
        // per-feature loop — those all fail and bail.
        let up_native = self.index.up_layer_matrix(layer);
        let down_native = self.index.down_layer_matrix(layer);
        let row_fallback = up_native.is_none() || down_native.is_none();
        if row_fallback
            && self.index.interleaved_kquant_layer_data(layer).is_none()
            && !self.index.has_fp4_storage()
        {
            return None;
        }

        let arch = &*self.weights.arch;
        let is_gated = arch.ffn_type() == larql_models::FfnType::Gated;
        let use_gelu = arch.activation().uses_gelu_tanh_gate_up();

        // Hint the kernel to start streaming layer N+1's Q4_K/Q6_K bytes
        // into the page cache while we work on N. No-op when there's no
        // Q4_K mmap, no manifest, or `layer+1` is out of range.
        self.index.prefetch_interleaved_kquant_layer(layer + 1);

        let mut out = Array2::<f32>::zeros((seq_len, hidden));
        // Observation record — only in Record mode; the per-feature
        // loops below push exactly what they compute into it. The old
        // dense `seq_len × intermediate` zero-fill is gone.
        let mut obs = observe
            .recording()
            .then(|| SparseActivations::new(seq_len, intermediate));

        let layer_has_overrides = self.index.has_overrides_at(layer);
        let up_bias_for_layer = if !is_gated {
            arch.ffn_up_bias_key(layer)
                .and_then(|bk| self.weights.vectors.get(&bk).cloned())
        } else {
            None
        };
        let activation_floor = self.config.effective_activation_floor();

        // ── Full-K gemv fast path (`sparse_gemv.rs`) ─────────────────────
        // Skipped when a non-default selector is configured, a per-layer
        // pool restriction is set, or a two-stage shortlist is requested:
        // in all three cases gemv would bypass the alternative selection
        // structure, so we force the walk.
        let selector_forces_walk = !matches!(self.config.selector, FeatureSelector::GateOnly)
            || self.config.pool_per_layer.is_some()
            || self.config.cell_router.is_some()
            || self.config.shortlist_m.is_some();
        let k_is_full =
            !selector_forces_walk && hits_len_ge_intermediate(&self.config, layer, intermediate);
        if !layer_has_overrides && is_gated && k_is_full {
            if let Some((gemv_out, act)) =
                self.sparse_full_k_gemv(layer, x, up_native, down_native, use_gelu, intermediate)
            {
                // The gemv computes every feature; its activation matrix
                // is intrinsic, so the observation is honestly Dense.
                return Some((gemv_out, observe.dense(act)));
            }
        }

        // ── Per-position sparse loop ─────────────────────────────────────
        for s in 0..seq_len {
            let x_row = x.row(s);
            let x_owned = x_row.to_owned();
            let x_slice_owned: Vec<f32>;
            let x_slice: &[f32] = if let Some(sl) = x_row.as_slice() {
                sl
            } else {
                x_slice_owned = x_owned.as_slice().unwrap().to_vec();
                &x_slice_owned
            };

            let top_k = self.top_k_for(layer);

            // ── Gather-contiguous Q4K fast path (`sparse_gather.rs`) ─────
            // For a KNOWN-pool route (precomputed pool or cell-router, no
            // within-pool ranking) the active feature set is decided without
            // gate scores, so we skip the scattered `local_pool_gate_knn` and
            // gather gate+up+down (down from the feature-major sidecar)
            // contiguous, running the fused kernel in one cache-friendly pass.
            // Fixes the ~4× per-row overhead at faithful K; re-gathers every
            // position (the content-addressed pool moves per token). Declines
            // (→ scalar paths) unless gated, no overrides, Q4K up, the down
            // sidecar is loaded, and the route has ≥ GATHER_MIN_FEATURES.
            if is_gated
                && !layer_has_overrides
                && up_native.is_none()
                && !self.config.rank_within_pool
                && self.index.has_down_features_kquant()
            {
                if let Some(feats) = self.gather_route_feats(layer, x_slice, top_k) {
                    if feats.len() >= GATHER_MIN_FEATURES {
                        if let Some(g) =
                            self.gather_q4k_accumulate(layer, &feats, x_slice, use_gelu, hidden)
                        {
                            let mut out_row = out.row_mut(s);
                            out_row.as_slice_mut().unwrap().copy_from_slice(&g.out);
                            if let Some(o) = obs.as_mut() {
                                // The gate/up dots the fused kernels
                                // actually computed ride into the record.
                                for (i, &feat) in feats.iter().enumerate() {
                                    o.record_scored(
                                        s,
                                        feat,
                                        g.acts[i],
                                        Some(g.gate_scores[i]),
                                        Some(g.up_scores[i]),
                                    );
                                }
                                o.set_kernel(s, PATH_SPARSE_GATHER);
                            }
                            self.trace_path(layer, PATH_SPARSE_GATHER);
                            continue;
                        }
                    }
                }
            }

            let t_gate = std::time::Instant::now();
            let hits = self.select_route_hits(layer, &x_owned, x_slice, top_k);
            let gate_knn_ns = t_gate.elapsed().as_nanos() as u64;

            let mut out_row = out.row_mut(s);

            // Parallel Q4K-down-cache path (`sparse_parallel.rs`).
            if self.try_parallel_q4k_down(
                layer,
                &hits,
                x_row,
                x_slice,
                up_native,
                use_gelu,
                down_native.is_none(),
                is_gated,
                layer_has_overrides,
                gate_knn_ns,
                &mut out_row,
                s,
                obs.as_mut(),
            ) {
                continue;
            }

            // Serial per-feature loop — the correctness baseline.
            for (feat, gate_score) in hits {
                // `(activation, gate-position score, up score)` — the
                // scores are recorded alongside the activation so the
                // runtime trace reports the EXECUTED projections
                // (2026-07-30 review, item 17). Non-gated archs have a
                // single projection: its post-bias value is the
                // gate-position score, and there is no up score.
                let (act, gate_obs, up_obs) = if is_gated {
                    let up_ov = if layer_has_overrides {
                        self.index.up_override(layer, feat)
                    } else {
                        None
                    };
                    let up_score = if let Some(up_ov) = up_ov.filter(|o| o.len() == hidden) {
                        ndarray::ArrayView1::from(up_ov).dot(&x_row)
                    } else if let Some(ref up_view) = up_native {
                        up_view.row(feat).dot(&x_row)
                    } else {
                        // Unified dispatch: FP4 → native → Q4K, per GateIndex.
                        self.index.ffn_row_dot(layer, FFN_UP, feat, x_slice)?
                    };
                    let activated_gate = if use_gelu {
                        crate::ffn::gelu_tanh(gate_score)
                    } else {
                        gate_score * crate::ffn::sigmoid(gate_score)
                    };
                    (activated_gate * up_score, gate_score, Some(up_score))
                } else {
                    let mut v = gate_score;
                    if let Some(ref bias) = up_bias_for_layer {
                        if feat < bias.len() {
                            v += bias[feat];
                        }
                    }
                    let act = if use_gelu {
                        crate::ffn::gelu_tanh(v)
                    } else {
                        v * crate::ffn::sigmoid(v)
                    };
                    (act, v, None)
                };

                if let Some(o) = obs.as_mut() {
                    // Recorded regardless of the floor: the floor gates
                    // the down accumulate, not activation observation.
                    o.record_scored(s, feat, act, Some(gate_obs), up_obs);
                }

                if act.abs() > activation_floor {
                    let down_ov = if layer_has_overrides {
                        self.index.down_override(layer, feat)
                    } else {
                        None
                    };
                    if let Some(override_down) = down_ov.filter(|o| o.len() == hidden) {
                        out_row.scaled_add(act, &ndarray::ArrayView1::from(override_down));
                        continue;
                    }
                    if let Some(ref down_view) = down_native {
                        out_row.scaled_add(act, &down_view.row(feat));
                    } else {
                        let out_slice = out_row.as_slice_mut().unwrap();
                        // Unified dispatch: FP4 → native → Q4K-via-cache, per GateIndex.
                        if !self
                            .index
                            .ffn_row_scaled_add(layer, FFN_DOWN, feat, act, out_slice)
                        {
                            return None;
                        }
                    }
                }
            }

            if let Some(o) = obs.as_mut() {
                o.set_kernel(s, PATH_SPARSE_SERIAL);
            }
        }

        // Down bias
        if let Some(bias) = arch
            .ffn_down_bias_key(layer)
            .and_then(|k| self.weights.vectors.get(&k))
        {
            crate::forward::add_bias(&mut out, bias);
        }

        self.trace_path(layer, PATH_SPARSE_SERIAL);
        Some((out, obs.map(FfnActivations::Sparse)))
    }
}

#[cfg(test)]
mod tests {
    use super::super::observe::Observe;
    use crate::ffn::FfnActivations;
    use crate::test_utils::{
        make_test_q4k_vindex, make_test_q4k_weights, make_test_vindex, make_test_weights,
    };
    use crate::vindex::{WalkFfn, WalkFfnConfig};
    use ndarray::Array2;

    /// Densify a Record-mode observation for assertions.
    fn obs_dense(obs: Option<FfnActivations>) -> Array2<f32> {
        obs.expect("Record mode observes")
            .into_dense()
            .expect("walk observations materialise densely")
    }

    fn x(seq: usize, hidden: usize) -> Array2<f32> {
        Array2::from_shape_vec(
            (seq, hidden),
            (0..seq * hidden).map(|i| (i as f32 + 1.0) * 0.02).collect(),
        )
        .unwrap()
    }

    /// Sparse walk over the Q4K fixture — `up_layer_matrix`/`down_layer_matrix`
    /// both return None (Q4K storage is byte-only) so the function
    /// routes through the row-fallback ladder dispatching via
    /// `ffn_row_dot` / `ffn_row_scaled_add`.
    #[test]
    fn walk_ffn_sparse_routes_through_q4k_fixture() {
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let cfg = WalkFfnConfig::sparse(weights.num_layers, 8);
        let ffn = WalkFfn::from_config(&weights, &index, cfg);
        let result = ffn.walk_ffn_sparse(0, &x(1, weights.hidden_size), Observe::Record);
        if let Some((out, obs)) = result {
            assert_eq!(out.shape(), &[1, weights.hidden_size]);
            assert_eq!(obs_dense(obs).shape()[0], 1);
        }
    }

    /// M3 regression (2026-07-30 review): `activation_floor` was
    /// documented and settable but read by nothing — the skip
    /// threshold was a hardcoded 1e-10. A floor above every
    /// activation's magnitude must now suppress every down
    /// contribution (zero output) while still recording the
    /// activations themselves.
    #[test]
    fn walk_ffn_sparse_honours_activation_floor() {
        use std::sync::Arc;
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        // The fixture has no gate vectors, so plain gate-KNN selects
        // nothing — route through a precomputed pool to get real hits.
        let pool: Vec<usize> = vec![0, 1, 2, 3];
        let cfg = |floor: f32| {
            let mut c = WalkFfnConfig::sparse(weights.num_layers, 4)
                .with_pool_per_layer(Arc::new(vec![pool.clone(); weights.num_layers]))
                .with_precomputed_routing(true);
            c.activation_floor = floor;
            c
        };

        let (out_low, act_low) = WalkFfn::from_config(&weights, &index, cfg(0.0))
            .walk_ffn_sparse(0, &x(1, weights.hidden_size), Observe::Record)
            .expect("sparse walk runs on the Q4K fixture");
        assert!(
            out_low.iter().any(|v| v.abs() > 0.0),
            "fixture must produce a non-zero FFN output at floor 0, \
             or this test can't distinguish anything"
        );

        let (out_high, act_high) = WalkFfn::from_config(&weights, &index, cfg(f32::MAX))
            .walk_ffn_sparse(0, &x(1, weights.hidden_size), Observe::Record)
            .expect("sparse walk runs on the Q4K fixture");
        assert!(
            out_high.iter().all(|v| *v == 0.0),
            "a floor above every |activation| must suppress all down \
             contributions — the configured floor is being ignored"
        );
        // The floor gates the down accumulate, not activation capture.
        assert_eq!(act_low, act_high);
    }

    /// Sparse walk over the feature-major f32 fixture — `up_layer_matrix`
    /// + `down_layer_matrix` both return Some so the function bypasses
    ///   the row-fallback and goes through the BLAS gemm fast path.
    #[test]
    fn walk_ffn_sparse_routes_through_feature_major_f32_fixture() {
        use crate::test_utils::attach_feature_major_f32_to_test_vindex;
        let weights = make_test_weights();
        let mut index = make_test_vindex(&weights);
        attach_feature_major_f32_to_test_vindex(&weights, &mut index);
        let cfg = WalkFfnConfig::sparse(weights.num_layers, 4);
        let ffn = WalkFfn::from_config(&weights, &index, cfg);
        let result = ffn
            .walk_ffn_sparse(0, &x(2, weights.hidden_size), Observe::Skip)
            .expect("feature-major f32 fixture should produce output");
        let (out, _obs) = result;
        assert_eq!(out.shape(), &[2, weights.hidden_size]);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    /// Sparse walk against a bare vindex (no FFN data) returns None —
    /// no native f32, no Q4K, no FP4 → the `row_fallback` guard fires.
    #[test]
    fn walk_ffn_sparse_returns_none_when_no_ffn_data() {
        let weights = make_test_weights();
        let index = make_test_vindex(&weights);
        let cfg = WalkFfnConfig::sparse(weights.num_layers, 4);
        let ffn = WalkFfn::from_config(&weights, &index, cfg);
        let result = ffn.walk_ffn_sparse(0, &x(1, weights.hidden_size), Observe::Skip);
        assert!(result.is_none());
    }

    /// Sparse walk against a StarCoder2-shaped arch (Standard FFN +
    /// up_bias) on a feature-major f32 fixture drives the
    /// `up_bias_for_layer = Some(...)` branch AND the non-gated
    /// activation arm of the serial loop.
    #[test]
    fn walk_ffn_sparse_non_gated_arch_uses_up_bias() {
        use crate::test_utils::{
            attach_feature_major_f32_to_test_vindex, make_starcoder2_test_weights,
        };
        let weights = make_starcoder2_test_weights();
        let mut index = make_test_vindex(&weights);
        attach_feature_major_f32_to_test_vindex(&weights, &mut index);
        let cfg = WalkFfnConfig::sparse(weights.num_layers, 4);
        let ffn = WalkFfn::from_config(&weights, &index, cfg);
        let out = ffn
            .walk_ffn_sparse(0, &x(1, weights.hidden_size), Observe::Skip)
            .expect("starcoder2 + feature-major fixture should produce output");
        assert_eq!(out.0.shape(), &[1, weights.hidden_size]);
        assert!(out.0.iter().all(|v| v.is_finite()));
    }

    // ── FP4 unified-dispatch stub ───────────────────────────────────────

    use larql_vindex::{
        FeatureMeta, Fp4FfnAccess, GateLookup, NativeFfnAccess, PatchOverrides, QuantizedFfnAccess,
    };

    /// The constant up-score every `fp4_ffn_row_dot` call returns —
    /// makes the serial loop's expected output hand-computable.
    const FP4_STUB_UP_DOT: f32 = 0.5;

    /// FP4-storage stub: gate hits with known scores, a constant up dot,
    /// and a scaled-add that writes `out[feat] += alpha` — so the serial
    /// loop's output is exactly `out[feat] = act(feat)`. The two `fail_*`
    /// flags turn each row op off to drive the walk's abort paths.
    struct Fp4StubIndex {
        n_features: usize,
        fail_dot: bool,
        fail_scaled_add: bool,
    }

    impl Fp4StubIndex {
        fn ok(n_features: usize) -> Self {
            Self {
                n_features,
                fail_dot: false,
                fail_scaled_add: false,
            }
        }
        /// Gate score for feature `i` — must match `gate_knn` below.
        fn gate_score(i: usize) -> f32 {
            (i as f32 + 1.0) * 0.1
        }
    }

    impl GateLookup for Fp4StubIndex {
        fn gate_knn(
            &self,
            _layer: usize,
            _residual: &ndarray::Array1<f32>,
            top_k: usize,
        ) -> Vec<(usize, f32)> {
            (0..top_k.min(self.n_features))
                .map(|i| (i, Self::gate_score(i)))
                .collect()
        }
        fn feature_meta(&self, _layer: usize, _feature: usize) -> Option<FeatureMeta> {
            None
        }
        fn num_features(&self, _layer: usize) -> usize {
            self.n_features
        }
    }
    impl PatchOverrides for Fp4StubIndex {}
    impl NativeFfnAccess for Fp4StubIndex {}
    impl QuantizedFfnAccess for Fp4StubIndex {}
    impl Fp4FfnAccess for Fp4StubIndex {
        fn has_fp4_storage(&self) -> bool {
            true
        }
        fn fp4_ffn_row_dot(
            &self,
            _layer: usize,
            component: usize,
            _feat: usize,
            _x: &[f32],
        ) -> Option<f32> {
            if self.fail_dot || component != 1 {
                None
            } else {
                Some(FP4_STUB_UP_DOT)
            }
        }
        fn fp4_ffn_row_scaled_add(
            &self,
            _layer: usize,
            component: usize,
            feat: usize,
            alpha: f32,
            out: &mut [f32],
        ) -> bool {
            if self.fail_scaled_add || component != 2 || feat >= out.len() {
                return false;
            }
            out[feat] += alpha;
            true
        }
    }

    /// FP4 storage routes through ladder step 3 into the serial sparse
    /// walk (the point of the trait refactor: zero FP4-specific kernel
    /// code), and the output is EXACTLY the hand-computed serial math:
    /// `out[feat] = silu(gate) * up_dot` under the stub's unit down row.
    #[test]
    fn fp4_storage_routes_through_serial_sparse_with_exact_math() {
        use crate::ffn::FfnBackend;
        let weights = make_test_weights(); // tinymodel → SiLU, gated
                                           // Fewer features than hidden so the stub's `out[feat] += alpha`
                                           // down-write stays in range.
        let n_features = 8;
        assert!(n_features <= weights.hidden_size);
        let index = Fp4StubIndex::ok(n_features);
        let ffn = WalkFfn::new_unlimited(&weights, &index).with_dispatch_trace();

        let input = x(1, weights.hidden_size);
        let (out, obs) = ffn.forward_observed(0, &input);
        let act = obs.into_dense().expect("sparse observation densifies");

        let trace = ffn.take_dispatch_trace();
        assert_eq!(trace.len(), 1);
        assert_eq!(
            trace[0].path, "sparse:serial",
            "FP4 must be served by the format-blind serial walk"
        );

        assert_eq!(act.shape(), &[1, n_features]);
        for feat in 0..n_features {
            let g = Fp4StubIndex::gate_score(feat);
            // Same expression the serial loop evaluates (SiLU arm).
            let expected = (g * crate::ffn::sigmoid(g)) * FP4_STUB_UP_DOT;
            assert_eq!(act[[0, feat]], expected, "activation for feature {feat}");
            assert_eq!(
                out[[0, feat]],
                expected,
                "stub down adds act at out[feat] — must be bit-exact"
            );
        }
        for h in n_features..weights.hidden_size {
            assert_eq!(out[[0, h]], 0.0, "untouched hidden dim {h}");
        }
    }

    /// M7-adjacent contract: when the down accumulate fails mid-walk
    /// (`ffn_row_scaled_add` → false), the sparse walk must ABORT with
    /// `None` — and the ladder must then land on the safetensors
    /// fallback, never a half-accumulated sparse result.
    #[test]
    fn sparse_aborts_to_weights_fallback_when_scaled_add_fails() {
        use crate::ffn::FfnBackend;
        let weights = make_test_weights();
        let index = Fp4StubIndex {
            n_features: 8,
            fail_dot: false,
            fail_scaled_add: true,
        };
        let ffn = WalkFfn::new_unlimited(&weights, &index).with_dispatch_trace();

        // Direct: the walk itself must decline.
        assert!(ffn
            .walk_ffn_sparse(0, &x(1, weights.hidden_size), Observe::Skip)
            .is_none());

        // Ladder: FP4 step fails → fall through to weights_fallback.
        let out = ffn.forward(0, &x(1, weights.hidden_size));
        assert_eq!(out.shape(), &[1, weights.hidden_size]);
        let trace = ffn.take_dispatch_trace();
        assert_eq!(
            trace.last().map(|e| e.path),
            Some("weights_fallback:sparse"),
            "failed sparse walk must fall through to the safetensors path"
        );
    }

    /// Same abort contract for the up dot: `ffn_row_dot` returning
    /// `None` for a hit aborts the walk (`?` in the serial loop).
    #[test]
    fn sparse_aborts_when_row_dot_fails() {
        let weights = make_test_weights();
        let index = Fp4StubIndex {
            n_features: 8,
            fail_dot: true,
            fail_scaled_add: false,
        };
        let ffn = WalkFfn::new_unlimited(&weights, &index);
        assert!(ffn
            .walk_ffn_sparse(0, &x(1, weights.hidden_size), Observe::Record)
            .is_none());
    }

    // ── Override arms of the serial loop ────────────────────────────────

    /// Q4K fixture wrapper with per-feature up/down overrides installed.
    /// Delegates all storage access to the inner `VectorIndex`.
    struct OverrideQ4kIndex<'a> {
        inner: &'a larql_vindex::VectorIndex,
        up_zero_feat: usize,
        down_zero_feat: usize,
        zeros: Vec<f32>,
    }

    impl GateLookup for OverrideQ4kIndex<'_> {
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
    impl PatchOverrides for OverrideQ4kIndex<'_> {
        fn has_overrides_at(&self, _layer: usize) -> bool {
            true
        }
        fn up_override(&self, _layer: usize, feature: usize) -> Option<&[f32]> {
            (feature == self.up_zero_feat).then_some(self.zeros.as_slice())
        }
        fn down_override(&self, _layer: usize, feature: usize) -> Option<&[f32]> {
            (feature == self.down_zero_feat).then_some(self.zeros.as_slice())
        }
    }
    impl NativeFfnAccess for OverrideQ4kIndex<'_> {}
    impl QuantizedFfnAccess for OverrideQ4kIndex<'_> {
        fn has_interleaved_kquant(&self) -> bool {
            self.inner.has_interleaved_kquant()
        }
        fn interleaved_kquant_layer_data(&self, layer: usize) -> Option<[(&[u8], &str); 3]> {
            self.inner.interleaved_kquant_layer_data(layer)
        }
        fn kquant_ffn_layer(
            &self,
            layer: usize,
            component: usize,
        ) -> Option<std::sync::Arc<Vec<f32>>> {
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
    }
    impl Fp4FfnAccess for OverrideQ4kIndex<'_> {}

    /// Override arms of the serial loop, asserted against the
    /// no-override baseline on the same route:
    /// - a zero **up** override kills exactly that feature's activation
    ///   (and hence its down contribution), all others bit-equal;
    /// - a zero **down** override removes exactly `act · down_row` from
    ///   the output (down_row read from the same dequant cache the
    ///   un-overridden accumulate uses).
    #[test]
    fn walk_ffn_sparse_override_arms_match_baseline_minus_contributions() {
        use larql_vindex::FfnRowAccess as _;
        use std::sync::Arc;
        let weights = make_test_q4k_weights();
        let inner = make_test_q4k_vindex(&weights);
        let hidden = weights.hidden_size;
        let up_zero_feat = 1usize;
        let down_zero_feat = 2usize;
        let pool: Vec<usize> = vec![0, 1, 2, 3];
        let cfg = || {
            WalkFfnConfig::sparse(weights.num_layers, pool.len())
                .with_pool_per_layer(Arc::new(vec![pool.clone(); weights.num_layers]))
                .with_precomputed_routing(true)
        };
        let input = x(1, hidden);

        let (out_base, obs_base) = WalkFfn::from_config(&weights, &inner, cfg())
            .walk_ffn_sparse(0, &input, Observe::Record)
            .expect("baseline walk runs");
        let act_base = obs_dense(obs_base);

        let overridden = OverrideQ4kIndex {
            inner: &inner,
            up_zero_feat,
            down_zero_feat,
            zeros: vec![0.0; hidden],
        };
        let (out_ov, obs_ov) = WalkFfn::from_config(&weights, &overridden, cfg())
            .walk_ffn_sparse(0, &input, Observe::Record)
            .expect("override walk runs");
        let act_ov = obs_dense(obs_ov);

        // Up override → that feature's activation is exactly 0.
        assert!(
            act_base[[0, up_zero_feat]].abs() > 0.0,
            "baseline must activate the up-overridden feature"
        );
        assert_eq!(act_ov[[0, up_zero_feat]], 0.0);
        for &f in &pool {
            if f != up_zero_feat {
                assert_eq!(act_ov[[0, f]], act_base[[0, f]], "feature {f} activation");
            }
        }

        // Output: baseline minus the two features' down contributions.
        let down_cache = inner.kquant_ffn_layer(0, 2).expect("down dequant cache");
        let mut expected = out_base.clone();
        for (feat, act) in [
            (up_zero_feat, act_base[[0, up_zero_feat]]),
            (down_zero_feat, act_base[[0, down_zero_feat]]),
        ] {
            let row = &down_cache[feat * hidden..(feat + 1) * hidden];
            for h in 0..hidden {
                expected[[0, h]] -= act * row[h];
            }
        }
        // Sanity: the unified dispatch would have used the same cache row.
        assert!(inner
            .ffn_row_dot(0, 1, up_zero_feat, input.row(0).as_slice().unwrap())
            .is_some());
        for h in 0..hidden {
            assert!(
                (out_ov[[0, h]] - expected[[0, h]]).abs() <= 1e-5,
                "hidden {h}: override output {} vs baseline-minus-contribs {}",
                out_ov[[0, h]],
                expected[[0, h]]
            );
        }
    }

    /// The override arm of the full routing ladder, observed: this
    /// index answers only point queries (`override_slots_at` keeps its
    /// `None` default) so base+delta declines, and the override-aware
    /// sparse walk serves the observed call with its real activations
    /// (mod.rs arm 1b in `Observe::Record` mode).
    #[test]
    fn override_arm_forward_observed_routes_sparse_with_real_observation() {
        use crate::ffn::FfnBackend;
        use std::sync::Arc;
        let weights = make_test_q4k_weights();
        let inner = make_test_q4k_vindex(&weights);
        let hidden = weights.hidden_size;
        let overridden = OverrideQ4kIndex {
            inner: &inner,
            up_zero_feat: 1,
            down_zero_feat: 2,
            zeros: vec![0.0; hidden],
        };
        let pool: Vec<usize> = vec![0, 1, 2, 3];
        let cfg = WalkFfnConfig::sparse(weights.num_layers, pool.len())
            .with_pool_per_layer(Arc::new(vec![pool; weights.num_layers]))
            .with_precomputed_routing(true);
        let ffn = WalkFfn::from_config(&weights, &overridden, cfg).with_dispatch_trace();
        let input = x(1, hidden);
        let (out, obs) = ffn.forward_observed(0, &input);
        assert_eq!(
            ffn.take_dispatch_trace().last().map(|e| e.path),
            Some("sparse:serial"),
            "point-query-only overrides must route the sparse walk"
        );
        let act = obs.into_dense().expect("sparse observation densifies");
        assert!(act.iter().any(|v| v.abs() > 0.0), "real activations");
        assert_eq!(
            act[[0, 1]],
            0.0,
            "zero up override kills that slot's observed activation"
        );
        assert_eq!(out, ffn.forward(0, &input), "Skip/Record outputs agree");
    }

    /// A non-contiguous input (column-major storage) must produce the
    /// same output as the same values in standard layout — pins the
    /// owned-slice fallbacks in the per-position preamble.
    #[test]
    fn walk_ffn_sparse_non_contiguous_input_matches_contiguous() {
        use std::sync::Arc;
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let hidden = weights.hidden_size;
        let seq = 2usize;
        let pool: Vec<usize> = (0..8).collect();
        let cfg = || {
            WalkFfnConfig::sparse(weights.num_layers, pool.len())
                .with_pool_per_layer(Arc::new(vec![pool.clone(); weights.num_layers]))
                .with_precomputed_routing(true)
        };

        let x_std = x(seq, hidden);
        // Same values, column-major: rows are strided, `as_slice` is None.
        let mut transposed_data = Vec::with_capacity(seq * hidden);
        for h in 0..hidden {
            for s in 0..seq {
                transposed_data.push(x_std[[s, h]]);
            }
        }
        let x_cm = Array2::from_shape_vec((hidden, seq), transposed_data)
            .unwrap()
            .reversed_axes();
        assert_eq!(x_cm, x_std);
        assert!(
            x_cm.row(0).as_slice().is_none(),
            "fixture must actually be non-contiguous or the test is vacuous"
        );

        let (out_std, act_std) = WalkFfn::from_config(&weights, &index, cfg())
            .walk_ffn_sparse(0, &x_std, Observe::Record)
            .expect("contiguous walk runs");
        let (out_cm, act_cm) = WalkFfn::from_config(&weights, &index, cfg())
            .walk_ffn_sparse(0, &x_cm, Observe::Record)
            .expect("column-major walk runs");
        assert_eq!(out_cm, out_std);
        assert_eq!(act_cm, act_std);
    }
}
