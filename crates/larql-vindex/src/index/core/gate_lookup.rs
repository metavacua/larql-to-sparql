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
