//! `impl GateIndex for PatchedVindex` — the trait conformance that
//! lets the patch overlay slot in wherever a `GateIndex` is expected
//! (also implemented by `VectorIndex`). Pulled out of `overlay.rs` so
//! the file holding `PatchedVindex`'s own API stays focused.
//!
//! Most methods forward to the inherent `PatchedVindex` impl;
//! `gate_override` reads from the patch overlay (not the base) and
//! `gate_knn_batch` re-ranks per-row to surface inserted slots that
//! the base path would miss.

use ndarray::Array1;

use crate::index::{FeatureMeta, GateIndex};

use super::overlay::PatchedVindex;

impl GateIndex for PatchedVindex {
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
        self.base.down_override(layer, feature)
    }

    fn up_override(&self, layer: usize, feature: usize) -> Option<&[f32]> {
        self.base.up_override(layer, feature)
    }

    fn gate_override(&self, layer: usize, feature: usize) -> Option<&[f32]> {
        // Gate overrides live on the patch overlay (not the base
        // index). Surface them through the trait so the sparse
        // inference fallback can read the strong installed gate.
        self.overrides_gate.get(&(layer, feature)).map(|v| v.as_slice())
    }

    fn has_overrides_at(&self, layer: usize) -> bool {
        self.overrides_gate.keys().any(|(l, _)| *l == layer)
            || self.base.has_overrides_at(layer)
    }

    fn down_feature_vector(&self, layer: usize, feature: usize) -> Option<&[f32]> {
        self.base.down_feature_vector(layer, feature)
    }

    fn has_down_features(&self) -> bool {
        self.base.has_down_features()
    }

    fn down_layer_matrix(&self, layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        self.base.down_layer_matrix(layer)
    }

    fn gate_scores_batch(&self, layer: usize, x: &ndarray::Array2<f32>) -> Option<ndarray::Array2<f32>> {
        self.base.gate_scores_batch(layer, x)
    }

    fn up_layer_matrix(&self, layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        self.base.up_layer_matrix(layer)
    }

    fn has_full_mmap_ffn(&self) -> bool {
        self.base.has_full_mmap_ffn()
    }

    fn has_interleaved(&self) -> bool {
        self.base.has_interleaved()
    }

    fn interleaved_gate(&self, layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        self.base.interleaved_gate(layer)
    }

    fn interleaved_up(&self, layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        self.base.interleaved_up(layer)
    }

    fn interleaved_down(&self, layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        self.base.interleaved_down(layer)
    }

    fn has_interleaved_q4(&self) -> bool {
        self.base.has_interleaved_q4()
    }

    fn interleaved_q4_mmap_ref(&self) -> Option<&[u8]> {
        self.base.interleaved_q4_mmap_ref()
    }

    fn has_interleaved_q4k(&self) -> bool {
        self.base.has_interleaved_q4k()
    }

    fn interleaved_q4k_mmap_ref(&self) -> Option<&[u8]> {
        self.base.interleaved_q4k_mmap_ref()
    }

    fn interleaved_q4k_layer_data(&self, layer: usize) -> Option<[(&[u8], &str); 3]> {
        self.base.interleaved_q4k_layer_data(layer)
    }

    fn q4k_ffn_layer(&self, layer: usize, component: usize)
        -> Option<std::sync::Arc<Vec<f32>>>
    {
        self.base.q4k_ffn_layer(layer, component)
    }

    fn q4k_ffn_row_into(&self, layer: usize, component: usize, feat: usize, out: &mut [f32]) -> bool {
        self.base.q4k_ffn_row_into(layer, component, feat, out)
    }

    fn q4k_ffn_row_dot(&self, layer: usize, component: usize, feat: usize, x: &[f32]) -> Option<f32> {
        self.base.q4k_ffn_row_dot(layer, component, feat, x)
    }

    fn q4k_ffn_row_dot_via_cache(&self, layer: usize, component: usize, feat: usize, x: &[f32]) -> Option<f32> {
        self.base.q4k_ffn_row_dot_via_cache(layer, component, feat, x)
    }
    fn q4k_ffn_row_scaled_add_via_cache(&self, layer: usize, component: usize, feat: usize, alpha: f32, out: &mut [f32]) -> bool {
        self.base.q4k_ffn_row_scaled_add_via_cache(layer, component, feat, alpha, out)
    }

    fn q4k_ffn_row_scaled_add(&self, layer: usize, component: usize, feat: usize, alpha: f32, out: &mut [f32]) -> bool {
        self.base.q4k_ffn_row_scaled_add(layer, component, feat, alpha, out)
    }

    fn q4k_matmul_transb(
        &self,
        layer: usize,
        component: usize,
        x: &[f32],
        x_rows: usize,
        backend: Option<&dyn larql_compute::ComputeBackend>,
    ) -> Option<Vec<f32>> {
        self.base.q4k_matmul_transb(layer, component, x, x_rows, backend)
    }

    fn gate_knn_batch(&self, layer: usize, x: &ndarray::Array2<f32>, top_k: usize) -> Vec<usize> {
        // The base impl runs a BLAS gemm against the disk-side gate
        // matrix and ignores the patch overlay — so any feature with
        // an overridden gate (e.g. an INSERT slot) wouldn't be in the
        // candidate set. Re-rank per row using the per-row `gate_knn`
        // path, which `PatchedVindex::gate_knn` overrides correctly.
        // Returns the union of selected feature indices across all
        // rows, deduplicated.
        if self.overrides_gate.iter().all(|((l, _), _)| *l != layer) {
            // No overrides at this layer — base path is correct.
            return self.base.gate_knn_batch(layer, x, top_k);
        }
        let mut selected = std::collections::BTreeSet::<usize>::new();
        for s in 0..x.shape()[0] {
            let row = x.row(s).to_owned();
            let hits = self.gate_knn(layer, &row, top_k);
            for (feat, _) in hits {
                selected.insert(feat);
            }
        }
        selected.into_iter().collect()
    }
}
