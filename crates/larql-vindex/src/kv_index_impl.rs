//! `KvIndex` impl for `VectorIndex` (ADR-0022 Step 3a).
//!
//! Thin delegation — five methods forward to existing inherent methods
//! on `VectorIndex`; the no-arg `interleaved_kquant_mmap_ref` forwards
//! through the existing `QuantizedFfnAccess` trait impl. Defined in
//! `larql-vindex` so the trait (in `larql-compute`) stays free of
//! vindex-internal types.

use std::sync::Arc;

use larql_compute::{KvIndex, FFN_COMPONENTS_PER_LAYER as COMPUTE_FFN_COMPONENTS_PER_LAYER};

use crate::index::storage::ffn_store::FFN_COMPONENTS_PER_LAYER as VINDEX_FFN_COMPONENTS_PER_LAYER;
use crate::index::types::QuantizedFfnAccess;
use crate::VectorIndex;

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

const _: () = {
    // Pin that the trait's component constants match the wire format's.
    // Mismatch would silently slice fewer/more components (or the wrong
    // one) and corrupt FFN dispatch — fail at compile time instead.
    assert!(COMPUTE_FFN_COMPONENTS_PER_LAYER == VINDEX_FFN_COMPONENTS_PER_LAYER);
    assert!(larql_compute::FFN_GATE == crate::index::storage::ffn_store::FFN_GATE);
    assert!(larql_compute::FFN_UP == crate::index::storage::ffn_store::FFN_UP);
    assert!(larql_compute::FFN_DOWN == crate::index::storage::ffn_store::FFN_DOWN);
};

impl KvIndex for VectorIndex {
    // `#[inline]` on every method: each is a pure delegator to an
    // inherent method (or a trait method we know the impl of).
    // When the call site has a concrete `&VectorIndex` (rather than
    // `&dyn KvIndex`), inlining lets the compiler devirtualize through
    // the trait call and emit the underlying VectorIndex method
    // directly. That recovers the ~6% standard-engine gap that the
    // post-ADR-0022 Step 7 bench measured.

    #[inline]
    fn num_features(&self, layer: usize) -> usize {
        // Inherent on VectorIndex (storage/gate_accessors/mod.rs).
        VectorIndex::num_features(self, layer)
    }

    #[inline]
    fn attn_kquant_layer_data(&self, layer: usize) -> Option<[(&[u8], &str); 4]> {
        // Inherent on VectorIndex (storage/attn.rs).
        VectorIndex::attn_kquant_layer_data(self, layer)
    }

    #[inline]
    fn attn_q8_layer_data(&self, layer: usize) -> Option<[(&[u8], &[f32]); 4]> {
        VectorIndex::attn_q8_layer_data(self, layer)
    }

    #[inline]
    fn interleaved_kquant_layer_data(
        &self,
        layer: usize,
    ) -> Option<[(&[u8], &str); COMPUTE_FFN_COMPONENTS_PER_LAYER]> {
        // Inherent on VectorIndex (storage/ffn_store/interleaved_kquant.rs).
        VectorIndex::interleaved_kquant_layer_data(self, layer)
    }

    #[inline]
    fn interleaved_kquant_mmap_ref(&self) -> Option<&[u8]> {
        // No inherent equivalent — only on the QuantizedFfnAccess trait
        // (whose impl for VectorIndex lives in index/core/quantized_ffn.rs).
        <Self as QuantizedFfnAccess>::interleaved_kquant_mmap_ref(self)
    }

    #[inline]
    fn interleaved_q4_mmap_ref(&self) -> Option<&[u8]> {
        // Legacy Q4_0 sibling — also on QuantizedFfnAccess.
        <Self as QuantizedFfnAccess>::interleaved_q4_mmap_ref(self)
    }

    #[inline]
    fn kquant_ffn_layer_once(&self, layer: usize, component: usize) -> Option<Arc<Vec<f32>>> {
        // Inherent on VectorIndex (storage/ffn_store/kquant_cache.rs).
        VectorIndex::kquant_ffn_layer_once(self, layer, component)
    }

    #[inline]
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_vector_index_satisfies_kv_index_trait() {
        // Sanity check: a fresh VectorIndex (no kquant data loaded)
        // delegates cleanly — every kquant accessor returns None,
        // num_features returns the default (0).
        let v = VectorIndex::empty(2, 16);
        let idx: &dyn KvIndex = &v;
        assert_eq!(idx.vocab_size(), 0);
        assert_eq!(idx.num_features(0), 0);
        assert!(idx.attn_kquant_layer_data(0).is_none());
        assert!(idx.interleaved_kquant_layer_data(0).is_none());
        assert!(idx.interleaved_kquant_mmap_ref().is_none());
        assert!(idx.kquant_ffn_layer_once(0, 0).is_none());
    }

    #[test]
    fn vocab_size_field_passes_through() {
        let mut v = VectorIndex::empty(1, 8);
        v.vocab_size = 32;
        let idx: &dyn KvIndex = &v;
        assert_eq!(idx.vocab_size(), 32);
    }
}
