//! `impl GateLookup for VectorIndex`.
//!
//! Thin delegation shim over the inherent `gate_*` methods that live
//! on `VectorIndex` itself (defined in `index::compute::gate_knn` and
//! `index::storage::gate_accessors`). Keeping the trait impl separate
//! from the inherent impl makes the capability surface easy to read
//! without scrolling through the storage implementation.

use ndarray::{Array1, Array2};

use super::VectorIndex;
use crate::index::types::{FeatureMeta, GateLookup};

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

impl GateLookup for VectorIndex {
    fn gate_knn(&self, layer: usize, residual: &Array1<f32>, top_k: usize) -> Vec<(usize, f32)> {
        self.gate_knn(layer, residual, top_k)
    }

    fn feature_meta(&self, layer: usize, feature: usize) -> Option<FeatureMeta> {
        self.feature_meta(layer, feature)
    }

    fn num_features(&self, layer: usize) -> usize {
        self.num_features(layer)
    }

    fn gate_knn_batch(&self, layer: usize, x: &Array2<f32>, top_k: usize) -> Vec<usize> {
        self.gate_knn_batch(layer, x, top_k)
    }

    fn gate_knn_q4(
        &self,
        layer: usize,
        residual: &ndarray::Array1<f32>,
        top_k: usize,
        backend: &dyn larql_compute::ComputeBackend,
    ) -> Option<Vec<(usize, f32)>> {
        // Delegate to VectorIndex's existing gate_knn_q4 method
        VectorIndex::gate_knn_q4(self, layer, residual, top_k, backend)
    }

    fn gate_scores_batch(&self, layer: usize, x: &Array2<f32>) -> Option<Array2<f32>> {
        self.gate_scores_batch(layer, x)
    }

    fn gate_scores_batch_backend(
        &self,
        layer: usize,
        x: &Array2<f32>,
        backend: Option<&dyn larql_compute::ComputeBackend>,
    ) -> Option<Array2<f32>> {
        self.gate_scores_batch_backend(layer, x, backend)
    }

    fn gate_walk(
        &self,
        layer: usize,
        residual: &Array1<f32>,
        top_k: usize,
    ) -> Option<Vec<(usize, f32)>> {
        // Delegate to the inherent exact batched-gemv implementation.
        // This shim was missing from day one (the trait default's
        // "Override in VectorIndex" comment dates to 2026-04-04): every
        // `&dyn GateIndex` caller — the sparse walk's whole selection
        // chain — silently got the `None` default and fell through to
        // `gate_knn`, which let the approximate HNSW serving path leak
        // into walk numerics whenever `enable_hnsw()` was on
        // (2026-07-30 review, item #13). Returns `None` only when no
        // dense-resolvable gate exists (e.g. Q4K-interleaved-only
        // storage), where the `gate_knn_q4`/`gate_knn` fallbacks are
        // the right tools.
        VectorIndex::gate_walk(self, layer, residual, top_k)
    }
}

#[cfg(test)]
mod tests {
    //! These tests pin the trait-impl shims by calling each one against
    //! a freshly constructed `VectorIndex::empty()` so the delegation
    //! lines run under coverage. The inherent methods themselves are
    //! covered by the storage-engine and walk integration tests.
    use super::*;

    fn fresh() -> VectorIndex {
        VectorIndex::empty(2, 8)
    }

    #[test]
    fn gate_knn_delegates_to_inherent_on_empty() {
        let v = fresh();
        let r = Array1::<f32>::zeros(8);
        // Empty vindex returns no features; the trait-impl line still runs.
        let hits = <VectorIndex as GateLookup>::gate_knn(&v, 0, &r, 4);
        assert!(hits.is_empty());
    }

    #[test]
    fn feature_meta_delegates_to_inherent() {
        let v = fresh();
        assert!(<VectorIndex as GateLookup>::feature_meta(&v, 0, 0).is_none());
    }

    /// `gate_walk` must reach the inherent exact gemv, not the trait's
    /// `None` default.
    ///
    /// This needs a **populated** index, and that is the whole point: on
    /// `VectorIndex::empty()` the inherent method also returns `None`
    /// (no features), so the delegating and non-delegating impls are
    /// indistinguishable — which is exactly how the missing override
    /// survived while every other shim here had a passing test. An
    /// empty-index assertion pins that the line compiles, not that it
    /// delegates.
    #[test]
    fn gate_walk_delegates_to_inherent_on_a_populated_index() {
        const LAYERS: usize = 1;
        const HIDDEN: usize = 4;
        const FEATURES: usize = 3;

        // Feature f is the f-th basis vector, so a one-hot residual has an
        // unambiguous nearest feature and the result is easy to state.
        let gate =
            Array2::from_shape_fn((FEATURES, HIDDEN), |(f, h)| if f == h { 1.0 } else { 0.0 });
        let v = VectorIndex::new(vec![Some(gate)], vec![None], LAYERS, HIDDEN);

        let mut residual = Array1::<f32>::zeros(HIDDEN);
        residual[1] = 1.0;

        let hits = <VectorIndex as GateLookup>::gate_walk(&v, 0, &residual, 1)
            .expect("trait gate_walk must delegate, not return the default None");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, 1, "feature 1 is the nearest to a one-hot at 1");
    }

    #[test]
    fn num_features_delegates_to_inherent() {
        let v = fresh();
        // Empty index → no features at any layer.
        assert_eq!(<VectorIndex as GateLookup>::num_features(&v, 0), 0);
        assert_eq!(<VectorIndex as GateLookup>::num_features(&v, 1), 0);
    }

    #[test]
    fn gate_knn_batch_delegates_to_inherent() {
        let v = fresh();
        let x = Array2::<f32>::zeros((3, 8));
        let out = <VectorIndex as GateLookup>::gate_knn_batch(&v, 0, &x, 4);
        assert!(out.is_empty(), "no features → empty union");
    }

    #[test]
    fn gate_knn_q4_returns_none_on_empty_vindex() {
        let v = fresh();
        let r = Array1::<f32>::zeros(8);
        let backend = larql_compute::CpuBackend;
        // No Q4 gate data loaded → None.
        assert!(<VectorIndex as GateLookup>::gate_knn_q4(&v, 0, &r, 4, &backend).is_none());
    }

    #[test]
    fn gate_scores_batch_returns_none_on_empty_vindex() {
        let v = fresh();
        let x = Array2::<f32>::zeros((1, 8));
        assert!(<VectorIndex as GateLookup>::gate_scores_batch(&v, 0, &x).is_none());
    }

    #[test]
    fn gate_scores_batch_backend_returns_none_on_empty_vindex() {
        let v = fresh();
        let x = Array2::<f32>::zeros((1, 8));
        let backend = larql_compute::CpuBackend;
        assert!(
            <VectorIndex as GateLookup>::gate_scores_batch_backend(&v, 0, &x, Some(&backend))
                .is_none()
        );
    }

    #[test]
    fn gate_scores_batch_backend_with_no_backend_falls_through() {
        let v = fresh();
        let x = Array2::<f32>::zeros((1, 8));
        assert!(<VectorIndex as GateLookup>::gate_scores_batch_backend(&v, 0, &x, None).is_none());
    }
}
