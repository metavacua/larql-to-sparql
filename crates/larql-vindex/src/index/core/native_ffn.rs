//! `impl NativeFfnAccess for VectorIndex`.
//!
//! Delegation shim for f32-native FFN access — `down`/`up` row reads,
//! interleaved layer matrices, and the `prefetch_interleaved_layer`
//! madvise hint. Inherent methods live in `index::storage::ffn_store`.

use super::VectorIndex;
use crate::index::types::NativeFfnAccess;

impl NativeFfnAccess for VectorIndex {
    fn down_feature_vector(&self, layer: usize, feature: usize) -> Option<&[f32]> {
        self.down_feature_vector(layer, feature)
    }

    fn has_down_features(&self) -> bool {
        self.has_down_features()
    }

    fn down_layer_matrix(&self, layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        self.down_layer_matrix(layer)
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
}

#[cfg(test)]
mod tests {
    //! Trait-impl shim coverage. Inherent methods are exercised by
    //! `index/storage/ffn_store/*` integration tests; here we just pin
    //! that the trait dispatch lines run.
    use super::*;

    fn fresh() -> VectorIndex {
        VectorIndex::empty(2, 8)
    }

    #[test]
    fn down_feature_vector_returns_none_on_empty_vindex() {
        let v = fresh();
        assert!(<VectorIndex as NativeFfnAccess>::down_feature_vector(&v, 0, 0).is_none());
    }

    #[test]
    fn has_down_features_false_on_empty_vindex() {
        let v = fresh();
        assert!(!<VectorIndex as NativeFfnAccess>::has_down_features(&v));
    }

    #[test]
    fn down_layer_matrix_none_on_empty() {
        let v = fresh();
        assert!(<VectorIndex as NativeFfnAccess>::down_layer_matrix(&v, 0).is_none());
    }

    #[test]
    fn up_layer_matrix_none_on_empty() {
        let v = fresh();
        assert!(<VectorIndex as NativeFfnAccess>::up_layer_matrix(&v, 0).is_none());
    }

    #[test]
    fn has_full_mmap_ffn_false_on_empty() {
        let v = fresh();
        assert!(!<VectorIndex as NativeFfnAccess>::has_full_mmap_ffn(&v));
    }

    #[test]
    fn has_interleaved_false_on_empty() {
        let v = fresh();
        assert!(!<VectorIndex as NativeFfnAccess>::has_interleaved(&v));
    }

    #[test]
    fn interleaved_gate_up_down_none_on_empty() {
        let v = fresh();
        assert!(<VectorIndex as NativeFfnAccess>::interleaved_gate(&v, 0).is_none());
        assert!(<VectorIndex as NativeFfnAccess>::interleaved_up(&v, 0).is_none());
        assert!(<VectorIndex as NativeFfnAccess>::interleaved_down(&v, 0).is_none());
    }

    #[test]
    fn prefetch_interleaved_layer_is_safe_on_empty() {
        let v = fresh();
        // No mmap → no madvise; must not panic.
        <VectorIndex as NativeFfnAccess>::prefetch_interleaved_layer(&v, 0);
    }
}
