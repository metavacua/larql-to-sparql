//! `impl GateIndex for VectorIndex` — the trait implementation that
//! lets `VectorIndex` plug into the `GateIndex` abstraction (also
//! implemented by `PatchedVindex`). Pulled out of `core.rs` so the
//! struct definition + constructors stay focused.

use ndarray::{Array1, Array2};

use super::core::VectorIndex;
use super::types::*;

impl GateIndex for VectorIndex {
    fn gate_knn(&self, layer: usize, residual: &Array1<f32>, top_k: usize) -> Vec<(usize, f32)> {
        self.gate_knn(layer, residual, top_k)
    }

    fn feature_meta(&self, layer: usize, feature: usize) -> Option<FeatureMeta> {
        self.feature_meta(layer, feature)
    }

    fn num_features(&self, layer: usize) -> usize {
        self.num_features(layer)
    }

    fn down_override(&self, layer: usize, feature: usize) -> Option<&[f32]> {
        self.down_overrides.get(&(layer, feature)).map(|v| v.as_slice())
    }

    fn up_override(&self, layer: usize, feature: usize) -> Option<&[f32]> {
        self.up_overrides.get(&(layer, feature)).map(|v| v.as_slice())
    }

    fn has_overrides_at(&self, layer: usize) -> bool {
        self.down_overrides.keys().any(|(l, _)| *l == layer)
            || self.up_overrides.keys().any(|(l, _)| *l == layer)
    }

    fn gate_knn_batch(&self, layer: usize, x: &Array2<f32>, top_k: usize) -> Vec<usize> {
        self.gate_knn_batch(layer, x, top_k)
    }

    fn down_feature_vector(&self, layer: usize, feature: usize) -> Option<&[f32]> {
        self.down_feature_vector(layer, feature)
    }

    fn has_down_features(&self) -> bool {
        self.down_features_mmap.is_some()
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

    fn down_layer_matrix(&self, layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        self.down_layer_matrix(layer)
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

    fn up_layer_matrix(&self, layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        self.up_layer_matrix(layer)
    }

    fn has_full_mmap_ffn(&self) -> bool {
        self.has_full_mmap_ffn()
    }

    fn has_interleaved(&self) -> bool {
        self.has_interleaved()
    }

    fn interleaved_gate(&self, layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        self.interleaved_gate(layer)
    }

    fn interleaved_up(&self, layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        self.interleaved_up(layer)
    }

    fn interleaved_down(&self, layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        self.interleaved_down(layer)
    }

    fn prefetch_interleaved_layer(&self, layer: usize) {
        self.prefetch_interleaved_layer(layer)
    }

    fn has_interleaved_q4(&self) -> bool {
        self.has_interleaved_q4()
    }

    fn interleaved_q4_gate(&self, layer: usize) -> Option<ndarray::Array2<f32>> {
        self.interleaved_q4_gate(layer)
    }

    fn interleaved_q4_up(&self, layer: usize) -> Option<ndarray::Array2<f32>> {
        self.interleaved_q4_up(layer)
    }

    fn interleaved_q4_down(&self, layer: usize) -> Option<ndarray::Array2<f32>> {
        self.interleaved_q4_down(layer)
    }

    fn prefetch_interleaved_q4_layer(&self, layer: usize) {
        self.prefetch_interleaved_q4_layer(layer)
    }

    fn interleaved_q4_mmap_ref(&self) -> Option<&[u8]> {
        self.interleaved_q4_mmap.as_ref().map(|m| m.as_ref() as &[u8])
    }

    fn has_interleaved_q4k(&self) -> bool {
        self.has_interleaved_q4k()
    }

    fn interleaved_q4k_mmap_ref(&self) -> Option<&[u8]> {
        self.interleaved_q4k_mmap.as_ref().map(|m| m.as_ref() as &[u8])
    }

    fn interleaved_q4k_layer_data(&self, layer: usize) -> Option<[(&[u8], &str); 3]> {
        VectorIndex::interleaved_q4k_layer_data(self, layer)
    }

    fn q4k_ffn_layer(&self, layer: usize, component: usize)
        -> Option<std::sync::Arc<Vec<f32>>>
    {
        VectorIndex::q4k_ffn_layer(self, layer, component)
    }

    fn q4k_ffn_row_into(&self, layer: usize, component: usize, feat: usize, out: &mut [f32]) -> bool {
        VectorIndex::q4k_ffn_row_into(self, layer, component, feat, out)
    }

    fn q4k_ffn_row_dot(&self, layer: usize, component: usize, feat: usize, x: &[f32]) -> Option<f32> {
        VectorIndex::q4k_ffn_row_dot(self, layer, component, feat, x)
    }

    fn q4k_ffn_row_dot_via_cache(&self, layer: usize, component: usize, feat: usize, x: &[f32]) -> Option<f32> {
        VectorIndex::q4k_ffn_row_dot_via_cache(self, layer, component, feat, x)
    }
    fn q4k_ffn_row_scaled_add_via_cache(&self, layer: usize, component: usize, feat: usize, alpha: f32, out: &mut [f32]) -> bool {
        VectorIndex::q4k_ffn_row_scaled_add_via_cache(self, layer, component, feat, alpha, out)
    }

    fn q4k_ffn_row_scaled_add(&self, layer: usize, component: usize, feat: usize, alpha: f32, out: &mut [f32]) -> bool {
        VectorIndex::q4k_ffn_row_scaled_add(self, layer, component, feat, alpha, out)
    }

    fn q4k_matmul_transb(
        &self,
        layer: usize,
        component: usize,
        x: &[f32],
        x_rows: usize,
        backend: Option<&dyn larql_compute::ComputeBackend>,
    ) -> Option<Vec<f32>> {
        VectorIndex::q4k_matmul_transb(self, layer, component, x, x_rows, backend)
    }
}
