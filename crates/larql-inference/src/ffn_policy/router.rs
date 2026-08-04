//! [`BoundFfnRouter`], [`BuildError`], and the [`ValidatedFfnLayerPolicy::build_router`]
//! method that ties a validated policy to live model handles.

use larql_vindex::VectorIndex;
use ndarray::Array2;

use crate::ffn::{FfnBackend, NullFfn, WeightFfn};
use crate::model::ModelWeights;
use crate::vindex::WalkFfn;

use super::backend_kind::FfnBackendKind;
use super::validated::ValidatedFfnLayerPolicy;

/// Owned per-layer FFN dispatcher built from a
/// [`ValidatedFfnLayerPolicy`].
///
/// Owns its backend instances (`Vec<Box<dyn FfnBackend + 'a>>`) so
/// callers don't have to manage backend lifetimes alongside the
/// router's. Lifetime `'a` ties to the [`ModelWeights`] and
/// [`VectorIndex`] borrows passed at build time — the router is
/// valid as long as those handles are.
///
/// Distinct from [`crate::ffn::LayerFfnRouter`], which takes
/// `Vec<&'a dyn FfnBackend>` (caller-owned). `BoundFfnRouter` trades
/// the slight per-layer Box overhead for a self-contained handle
/// that doesn't force callers to construct + hold backend instances
/// on their own stack. See
/// [`docs/ffn-build-router.md`](../../docs/ffn-build-router.md) §3
/// for the design
/// rationale and the deferred question of unifying the two router
/// types.
pub struct BoundFfnRouter<'a> {
    backends: Vec<Box<dyn FfnBackend + 'a>>,
}

impl<'a> BoundFfnRouter<'a> {
    /// FFN backend for a specific layer. Panics on out-of-range
    /// layer; callers should only pass `layer < num_layers()`.
    pub fn get(&self, layer: usize) -> &dyn FfnBackend {
        self.backends[layer].as_ref()
    }

    /// Number of layers this router dispatches over.
    pub fn num_layers(&self) -> usize {
        self.backends.len()
    }
}

impl<'a> std::fmt::Debug for BoundFfnRouter<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BoundFfnRouter")
            .field("num_layers", &self.backends.len())
            .field(
                "backend_names",
                &self.backends.iter().map(|b| b.name()).collect::<Vec<_>>(),
            )
            .finish()
    }
}

// `BoundFfnRouter` itself implements [`FfnBackend`] by delegating
// per-layer to the underlying backend selected by the policy. This is
// the integration trick that lets a single `&dyn FfnBackend` parameter
// (which today's engine trait surface and accuracy/bench harnesses
// already take) carry per-layer routing without changing any of the
// downstream signatures.
//
// The trait already takes `layer: usize` as a parameter on every
// dispatch method, so the delegation is straightforward:
//
//   router.forward(L, x) → router.get(L).forward(L, x)
//
// Callers that want the historical "one backend everywhere" behaviour
// still get it via a uniform policy (`dense`, `walk:k=100`); per-layer
// behaviour comes from the braced spec form. No engine has to know
// the difference.
impl<'a> FfnBackend for BoundFfnRouter<'a> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        self.get(layer).forward(layer, x)
    }

    fn forward_observed(
        &self,
        layer: usize,
        x: &Array2<f32>,
    ) -> (Array2<f32>, crate::ffn::FfnActivations) {
        self.get(layer).forward_observed(layer, x)
    }

    fn name(&self) -> &str {
        // Stable identifier so log lines / accuracy JSON have
        // something to anchor on. The per-layer composition is
        // captured in Debug; downstream tooling can inspect the
        // policy via `ValidatedFfnLayerPolicy::expand_to_layers`
        // for the full per-layer plan.
        "bound-router"
    }

    fn forward_moe_full_layer(
        &self,
        layer: usize,
        h_post_attn: &Array2<f32>,
    ) -> Result<Option<Array2<f32>>, larql_execution::BoxRefusal> {
        self.get(layer).forward_moe_full_layer(layer, h_post_attn)
    }
}

/// Errors from [`ValidatedFfnLayerPolicy::build_router`].
#[derive(Debug)]
pub enum BuildError {
    /// A `Walk { k }` binding requires a [`VectorIndex`], but `None`
    /// was passed. The `layer` is the index of the first layer that
    /// needed an index — useful for diagnosing which binding in a
    /// multi-binding spec was the problem.
    VectorIndexRequired { layer: usize },
    /// Remote backend construction is not yet wired in `build_router`
    /// v0. The variant exists so callers get a typed error rather
    /// than a panic; the build path lands in a follow-up slice.
    RemoteWalkNotYetWired { endpoint: String },
    /// Internal invariant violation: a [`ValidatedFfnLayerPolicy`]
    /// references a layer outside the model's `num_layers`. **Should
    /// be unreachable** for a [`ValidatedFfnLayerPolicy`] built via
    /// `validate_for(num_layers)` with a matching `weights`; presence
    /// indicates `weights` was swapped between validate and build,
    /// or a bug in `validate_for`. Surfaces as a typed error rather
    /// than a panic so callers can log and recover.
    LayerOutOfRange { layer: usize, num_layers: usize },
}

impl std::fmt::Display for BuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BuildError::VectorIndexRequired { layer } => write!(
                f,
                "layer {layer} requires a VectorIndex (Walk{{k}} binding), \
                 but build_router was called with index=None"
            ),
            BuildError::RemoteWalkNotYetWired { endpoint } => write!(
                f,
                "RemoteWalk backend (endpoint={endpoint:?}) is not yet wired \
                 in build_router v0 — wiring deferred to a follow-up slice"
            ),
            BuildError::LayerOutOfRange { layer, num_layers } => write!(
                f,
                "layer {layer} is out of range for model with {num_layers} layers \
                 (validate/build mismatch — likely a bug)"
            ),
        }
    }
}

impl std::error::Error for BuildError {}

impl ValidatedFfnLayerPolicy {
    /// Build a live [`BoundFfnRouter`] by binding each layer's
    /// [`FfnBackendKind`] to a concrete `&dyn FfnBackend`
    /// constructed from the model handles.
    ///
    /// Constructs one backend per layer (no deduplication across
    /// layers that share the same kind). The per-layer overhead is
    /// small for cheap backends (`Dense` → [`WeightFfn`] is a single
    /// pointer; `Null` → [`NullFfn`] is zero-sized); for
    /// `Walk { k }`, each instance carries its own caches, but those
    /// caches are lazily populated per-layer so the duplication is
    /// bounded. Dedup is a v1 optimization if profiling surfaces a
    /// hotspot.
    ///
    /// `index` is `Option<&VectorIndex>` because `Dense`, `Null`, and
    /// `RemoteWalk` don't need one — only `Walk { k }` does. Passing
    /// `None` and having any `Walk { k }` binding errors with
    /// [`BuildError::VectorIndexRequired`].
    ///
    /// `RemoteWalk` returns [`BuildError::RemoteWalkNotYetWired`] in
    /// v0 — the variant is parseable so the spec language stays
    /// stable, but the connection-pool wiring is a separate slice.
    pub fn build_router<'a>(
        &self,
        weights: &'a ModelWeights,
        index: Option<&'a VectorIndex>,
    ) -> Result<BoundFfnRouter<'a>, BuildError> {
        // Defensive: weights' layer count must match the count this
        // policy was validated against. ValidatedFfnLayerPolicy's
        // invariant should make this unreachable when the caller
        // passes the same weights handle to validate and build, but
        // surface a typed error rather than panicking if the handles
        // got crossed.
        if weights.num_layers != self.num_layers() {
            return Err(BuildError::LayerOutOfRange {
                layer: self.num_layers().saturating_sub(1),
                num_layers: weights.num_layers,
            });
        }

        let layer_kinds = self.expand_to_layers();
        let mut backends: Vec<Box<dyn FfnBackend + 'a>> = Vec::with_capacity(layer_kinds.len());

        for (layer, kind) in layer_kinds.iter().enumerate() {
            let backend: Box<dyn FfnBackend + 'a> = match kind {
                FfnBackendKind::Dense => Box::new(WeightFfn { weights }),
                FfnBackendKind::Walk { k } => {
                    let idx = index.ok_or(BuildError::VectorIndexRequired { layer })?;
                    match k {
                        None => Box::new(WalkFfn::new_unlimited(weights, idx)),
                        Some(n) => Box::new(WalkFfn::new(weights, idx, *n)),
                    }
                }
                FfnBackendKind::RemoteWalk { endpoint, .. } => {
                    return Err(BuildError::RemoteWalkNotYetWired {
                        endpoint: endpoint.clone(),
                    });
                }
                FfnBackendKind::Null => Box::new(NullFfn),
            };
            backends.push(backend);
        }

        Ok(BoundFfnRouter { backends })
    }
}

#[cfg(test)]
mod tests {
    use super::super::policy::FfnLayerPolicy;
    use super::*;
    use crate::test_utils::make_test_weights;

    #[test]
    fn build_router_dense_uniform_constructs_one_backend_per_layer() {
        let weights = make_test_weights();
        let v = FfnLayerPolicy::from_spec("dense")
            .unwrap()
            .validate_for(weights.num_layers)
            .unwrap();
        let router = v.build_router(&weights, None).unwrap();
        assert_eq!(router.num_layers(), weights.num_layers);
        for layer in 0..weights.num_layers {
            assert_eq!(router.get(layer).name(), "weights");
        }
    }

    #[test]
    fn build_router_null_uniform_works_without_index() {
        let weights = make_test_weights();
        let v = FfnLayerPolicy::from_spec("null")
            .unwrap()
            .validate_for(weights.num_layers)
            .unwrap();
        let router = v.build_router(&weights, None).unwrap();
        assert_eq!(router.num_layers(), weights.num_layers);
        // NullFfn's name is documented in larql_compute::ffn::weight.
        // We don't assert the exact string — the contract is just
        // "name() returns *some* identifier."
        assert!(!router.get(0).name().is_empty());
    }

    #[test]
    fn build_router_walk_without_index_errors_with_vector_index_required() {
        let weights = make_test_weights();
        let v = FfnLayerPolicy::from_spec("walk:k=100")
            .unwrap()
            .validate_for(weights.num_layers)
            .unwrap();
        // Bind the result before the match so the temporary holding
        // a borrow of `weights` drops before `weights` itself, even
        // when the `other => panic!(... {other:?})` arm formats the
        // Result with Debug.
        let result = v.build_router(&weights, None);
        match result {
            Err(BuildError::VectorIndexRequired { layer }) => {
                assert_eq!(layer, 0);
            }
            other => panic!("expected VectorIndexRequired, got {other:?}"),
        }
    }

    #[test]
    fn build_router_hybrid_walk_without_index_errors_at_first_walk_layer() {
        let weights = make_test_weights();
        let total = weights.num_layers;
        if total < 4 {
            return;
        }
        let spec = format!("{{walk:k=10}}@layers=2-{};{{dense}}@otherwise", total - 1);
        let v = FfnLayerPolicy::from_spec(&spec)
            .unwrap()
            .validate_for(total)
            .unwrap();
        let result = v.build_router(&weights, None);
        match result {
            Err(BuildError::VectorIndexRequired { layer }) => {
                assert_eq!(layer, 2);
            }
            other => panic!("expected VectorIndexRequired, got {other:?}"),
        }
    }

    #[test]
    fn build_router_remote_walk_errors_with_not_yet_wired() {
        let weights = make_test_weights();
        let v = FfnLayerPolicy::from_spec("remote-walk:endpoint=http://shard:8080")
            .unwrap()
            .validate_for(weights.num_layers)
            .unwrap();
        let result = v.build_router(&weights, None);
        match result {
            Err(BuildError::RemoteWalkNotYetWired { endpoint }) => {
                assert_eq!(endpoint, "http://shard:8080");
            }
            other => panic!("expected RemoteWalkNotYetWired, got {other:?}"),
        }
    }

    #[test]
    fn build_error_display_formats_each_variant() {
        let msg = format!("{}", BuildError::VectorIndexRequired { layer: 14 });
        assert!(msg.contains("14"));
        assert!(msg.contains("VectorIndex"));

        let msg = format!(
            "{}",
            BuildError::RemoteWalkNotYetWired {
                endpoint: "http://x".into(),
            }
        );
        assert!(msg.contains("http://x"));
        assert!(msg.contains("not yet wired"));

        let msg = format!(
            "{}",
            BuildError::LayerOutOfRange {
                layer: 99,
                num_layers: 34,
            }
        );
        assert!(msg.contains("99"));
        assert!(msg.contains("34"));
    }

    #[test]
    fn bound_router_debug_includes_layer_count_and_backend_names() {
        let weights = make_test_weights();
        let v = FfnLayerPolicy::from_spec("dense")
            .unwrap()
            .validate_for(weights.num_layers)
            .unwrap();
        let router = v.build_router(&weights, None).unwrap();
        let dbg = format!("{router:?}");
        assert!(dbg.contains("BoundFfnRouter"));
        assert!(dbg.contains(&format!("{}", weights.num_layers)));
    }

    // ── BoundFfnRouter as FfnBackend (delegation impl) ──────────────────────

    #[test]
    fn ffn_backend_impl_name_is_stable() {
        let weights = make_test_weights();
        let router = FfnLayerPolicy::from_spec("dense")
            .unwrap()
            .validate_for(weights.num_layers)
            .unwrap()
            .build_router(&weights, None)
            .unwrap();
        // The stable name is the load-bearing property — accuracy
        // JSON's `ffn_backend` column anchors on this when the CLI
        // wiring slice lands.
        let n: &dyn FfnBackend = &router;
        assert_eq!(n.name(), "bound-router");
    }

    #[test]
    fn ffn_backend_impl_forward_matches_underlying_per_layer_dispatch() {
        // The trait impl's contract: `router.forward(L, x)` produces
        // the same output as `router.get(L).forward(L, x)`. If this
        // breaks, every accuracy/bench harness that gets a router via
        // `&dyn FfnBackend` silently dispatches to the wrong backend.
        use ndarray::Array2;
        let weights = make_test_weights();
        let router = FfnLayerPolicy::from_spec("dense")
            .unwrap()
            .validate_for(weights.num_layers)
            .unwrap()
            .build_router(&weights, None)
            .unwrap();

        // Build a small synthetic input matching the weights' hidden
        // size. The exact values don't matter — we only need the
        // two dispatch paths to produce identical outputs.
        let hidden = weights.hidden_size;
        let x = Array2::<f32>::from_shape_fn((1, hidden), |(_, j)| 0.01 * (j as f32));

        for layer in 0..weights.num_layers {
            let via_router = (&router as &dyn FfnBackend).forward(layer, &x);
            let via_direct = router.get(layer).forward(layer, &x);
            assert_eq!(
                via_router.shape(),
                via_direct.shape(),
                "shape mismatch at layer {layer}"
            );
            for (a, b) in via_router.iter().zip(via_direct.iter()) {
                assert!(
                    (a - b).abs() < 1e-9,
                    "layer {layer}: router-dispatch and direct-dispatch \
                     produced different outputs ({a} vs {b})"
                );
            }
        }
    }

    #[test]
    fn ffn_backend_impl_forward_observed_matches_direct_dispatch() {
        // Same contract for the observed variant — output AND
        // observation must match the direct dispatch.
        use ndarray::Array2;
        let weights = make_test_weights();
        let router = FfnLayerPolicy::from_spec("dense")
            .unwrap()
            .validate_for(weights.num_layers)
            .unwrap()
            .build_router(&weights, None)
            .unwrap();

        let hidden = weights.hidden_size;
        let x = Array2::<f32>::from_shape_fn((1, hidden), |(_, j)| 0.02 * (j as f32));

        let layer = 0;
        let (via_router_out, via_router_obs) =
            (&router as &dyn FfnBackend).forward_observed(layer, &x);
        let (via_direct_out, via_direct_obs) = router.get(layer).forward_observed(layer, &x);
        assert_eq!(via_router_out, via_direct_out);
        assert_eq!(via_router_obs, via_direct_obs);
    }

    #[test]
    fn ffn_backend_impl_moe_full_layer_delegates_and_defaults_to_none() {
        // None of the v0 backends (Dense/Walk/Null) implement
        // `forward_moe_full_layer`; the trait's default impl returns
        // `None`. The router's delegating impl must preserve that.
        use ndarray::Array2;
        let weights = make_test_weights();
        let router = FfnLayerPolicy::from_spec("dense")
            .unwrap()
            .validate_for(weights.num_layers)
            .unwrap()
            .build_router(&weights, None)
            .unwrap();
        let x = Array2::<f32>::zeros((1, weights.hidden_size));
        let result = (&router as &dyn FfnBackend).forward_moe_full_layer(0, &x);
        assert!(
            matches!(result, Ok(None)),
            "v0 backends don't implement moe_full_layer; router delegation must \
             preserve the not-applicable default, not turn it into a refusal"
        );
    }

    #[test]
    fn ffn_backend_impl_works_as_dyn_object_in_existing_callsite_shape() {
        // Sanity check: a `&BoundFfnRouter` coerces to `&dyn FfnBackend`
        // and round-trips through a function that takes the trait
        // object directly. Mirrors how `evaluate_parametric` / bench
        // engines consume `&dyn FfnBackend` today — confirms drop-in
        // compatibility for the CLI wiring slice.
        fn takes_dyn(ffn: &dyn FfnBackend) -> &str {
            ffn.name()
        }
        let weights = make_test_weights();
        let router = FfnLayerPolicy::from_spec("dense")
            .unwrap()
            .validate_for(weights.num_layers)
            .unwrap()
            .build_router(&weights, None)
            .unwrap();
        assert_eq!(takes_dyn(&router), "bound-router");
    }
}
