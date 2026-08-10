//! `KvIndex` — substrate abstraction over `larql_vindex::VectorIndex`.
//!
//! Defined in `larql-compute` so the `KvDispatch` and
//! `AsyncComputeBackend` trait method signatures (Steps 3 and 4 of
//! ADR-0022) — plus the moved-down `kquant_forward` helpers — can take
//! `Option<&dyn KvIndex>` without depending on `larql-vindex` (which
//! sits above compute in the dep chain).
//!
//! `larql-vindex` implements this trait on `VectorIndex` as a thin
//! delegation; no behaviour changes vs. the pre-ADR direct-VectorIndex
//! call sites.

#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc;
#[cfg(target_arch = "wasm32")]
use alloc::sync::Arc;

/// Number of FFN components per layer (gate / up / down).
///
/// Mirrors `larql_vindex`'s wire-format constant. Defined here so
/// callers in this crate (kv-dispatch, kquant_forward) don't have to
/// reach up to `larql-vindex`. A `const _: () = { assert!(...) };`
/// pin in larql-vindex's `kv_index_impl.rs` keeps the two in sync.
pub const FFN_COMPONENTS_PER_LAYER: usize = 3;

/// Component index of the gate projection inside per-layer FFN
/// manifests / caches (`layer * FFN_COMPONENTS_PER_LAYER + component`).
pub const FFN_GATE: usize = 0;
/// Component index of the up projection.
pub const FFN_UP: usize = 1;
/// Component index of the down projection. Special-cased throughout the
/// codebase because down is stored row-major `[hidden, intermediate]`
/// (the native `nn.Linear(intermediate, hidden)` orientation) and needs
/// a transpose to become feature-major like gate / up.
pub const FFN_DOWN: usize = 2;

/// Abstract surface that the kv-dispatch + Q4_K direct-decode paths
/// need from a vindex. Implemented by `larql_vindex::VectorIndex`.
///
/// All returns are primitives, borrowed slices, or `Arc`'d data — no
/// vindex-internal types escape the abstraction.
///
/// **Inlining note:** every method has `#[inline]` because impls are
/// expected to be thin delegators to inherent methods on the underlying
/// vindex type. `#[inline]` on the trait method propagates the hint to
/// impl bodies; when the compiler sees a concrete `&VectorIndex` (not
/// a trait object), it can devirtualize + inline the whole chain,
/// erasing the vtable indirection introduced by the trait. Recovers
/// the ~6% standard-engine gap measured post-ADR-0022 Step 7.
pub trait KvIndex: Send + Sync {
    /// Number of FFN features (intermediate-dim) for a given layer.
    #[inline]
    fn num_features(&self, layer: usize) -> usize {
        let _ = layer;
        0
    }

    /// Per-layer (Q, K, V, O) bytes for the Q4_K-quantised attention
    /// projections, with a per-tensor format tag (`"Q4_K"`, `"Q6_K"`, …).
    /// Returns `None` if the vindex doesn't carry kquant attention data.
    #[inline]
    fn attn_kquant_layer_data(&self, layer: usize) -> Option<[(&[u8], &str); 4]> {
        let _ = layer;
        None
    }

    /// Per-layer (Q, K, V, O) bytes for Q8-quantised attention
    /// projections, with associated per-element scale tables (one
    /// `&[f32]` per tensor). Returns `None` if the vindex doesn't
    /// carry Q8 attention data. Used by Metal's fused dispatch when
    /// `attn_kquant_layer_data` is absent.
    #[inline]
    fn attn_q8_layer_data(&self, layer: usize) -> Option<[(&[u8], &[f32]); 4]> {
        let _ = layer;
        None
    }

    /// Per-layer FFN bytes (gate / up / down) for Q4_K/Q6_K interleaved
    /// storage, format-tagged. Returns `None` if the vindex doesn't
    /// carry interleaved-kquant FFN data.
    #[inline]
    fn interleaved_kquant_layer_data(
        &self,
        layer: usize,
    ) -> Option<[(&[u8], &str); FFN_COMPONENTS_PER_LAYER]> {
        let _ = layer;
        None
    }

    /// Direct mmap reference to the full interleaved-kquant FFN byte
    /// range (used by the direct-matvec decode path that dispatches
    /// kernels straight against mmap'd weights). No layer arg — the
    /// returned slice spans the whole vindex's interleaved FFN region.
    #[inline]
    fn interleaved_kquant_mmap_ref(&self) -> Option<&[u8]> {
        None
    }

    /// Legacy Q4_0 sibling of [`Self::interleaved_kquant_mmap_ref`].
    /// Returns the whole-vindex interleaved Q4_0 byte range when the
    /// vindex stores FFN weights as Q4_0 rather than Q4_K/Q6_K.
    #[inline]
    fn interleaved_q4_mmap_ref(&self) -> Option<&[u8]> {
        None
    }

    /// Cached dequantised FFN component (`0 = gate`, `1 = up`, `2 = down`)
    /// for a single (layer, component) pair. Returns `None` if the
    /// component hasn't been dequantised + cached yet.
    #[inline]
    fn kquant_ffn_layer_once(&self, layer: usize, component: usize) -> Option<Arc<Vec<f32>>> {
        let _ = (layer, component);
        None
    }

    /// Vocabulary size — needed for the lm_head projection in the
    /// kquant_forward path.
    #[inline]
    fn vocab_size(&self) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal stub that returns `None` everywhere. Pins the trait's
    /// "empty vindex" contract — every kquant accessor returns `None`,
    /// `num_features` returns 0. Downstream callers (the moved-down
    /// kquant_forward + KvDispatch impl) must fall back to f32 paths
    /// when this happens.
    struct EmptyKvIndex;
    impl KvIndex for EmptyKvIndex {
        fn num_features(&self, _layer: usize) -> usize {
            0
        }
        fn attn_kquant_layer_data(&self, _layer: usize) -> Option<[(&[u8], &str); 4]> {
            None
        }
        fn interleaved_kquant_layer_data(
            &self,
            _layer: usize,
        ) -> Option<[(&[u8], &str); FFN_COMPONENTS_PER_LAYER]> {
            None
        }
        fn interleaved_kquant_mmap_ref(&self) -> Option<&[u8]> {
            None
        }
        fn interleaved_q4_mmap_ref(&self) -> Option<&[u8]> {
            None
        }
        fn attn_q8_layer_data(&self, _layer: usize) -> Option<[(&[u8], &[f32]); 4]> {
            None
        }
        fn kquant_ffn_layer_once(&self, _layer: usize, _component: usize) -> Option<Arc<Vec<f32>>> {
            None
        }
        fn vocab_size(&self) -> usize {
            0
        }
    }

    #[test]
    fn empty_kv_index_returns_none_everywhere() {
        let idx: &dyn KvIndex = &EmptyKvIndex;
        assert_eq!(idx.num_features(0), 0);
        assert!(idx.attn_kquant_layer_data(0).is_none());
        assert!(idx.interleaved_kquant_layer_data(0).is_none());
        assert!(idx.interleaved_kquant_mmap_ref().is_none());
        assert!(idx.kquant_ffn_layer_once(0, 0).is_none());
        assert_eq!(idx.vocab_size(), 0);
    }

    #[test]
    fn ffn_components_per_layer_pinned_to_three() {
        // gate / up / down — pin the constant so a value drift breaks
        // the build instead of producing silent off-by-one slicing.
        // The larql-vindex side has a `const _: () = assert!(...)`
        // tying this to the wire-format constant.
        assert_eq!(FFN_COMPONENTS_PER_LAYER, 3);
    }

    /// Stub that returns canned data — verifies the borrowed-slice
    /// lifetime threading works as expected for callers that pass
    /// `&dyn KvIndex` and consume the returned `&[u8]` immediately.
    struct CannedKvIndex {
        attn: Vec<u8>,
        ffn: Vec<u8>,
        ffn_cache: Arc<Vec<f32>>,
    }
    impl KvIndex for CannedKvIndex {
        fn num_features(&self, _layer: usize) -> usize {
            32
        }
        fn attn_kquant_layer_data(&self, _layer: usize) -> Option<[(&[u8], &str); 4]> {
            Some([
                (self.attn.as_slice(), "Q4_K"),
                (self.attn.as_slice(), "Q4_K"),
                (self.attn.as_slice(), "Q4_K"),
                (self.attn.as_slice(), "Q6_K"),
            ])
        }
        fn interleaved_kquant_layer_data(
            &self,
            _layer: usize,
        ) -> Option<[(&[u8], &str); FFN_COMPONENTS_PER_LAYER]> {
            Some([
                (self.ffn.as_slice(), "Q4_K"),
                (self.ffn.as_slice(), "Q4_K"),
                (self.ffn.as_slice(), "Q6_K"),
            ])
        }
        fn interleaved_kquant_mmap_ref(&self) -> Option<&[u8]> {
            Some(self.ffn.as_slice())
        }
        fn interleaved_q4_mmap_ref(&self) -> Option<&[u8]> {
            None
        }
        fn attn_q8_layer_data(&self, _layer: usize) -> Option<[(&[u8], &[f32]); 4]> {
            None
        }
        fn kquant_ffn_layer_once(&self, _layer: usize, _component: usize) -> Option<Arc<Vec<f32>>> {
            Some(Arc::clone(&self.ffn_cache))
        }
        fn vocab_size(&self) -> usize {
            128
        }
    }

    #[test]
    fn canned_kv_index_returns_borrowed_slices_tied_to_self() {
        let idx = CannedKvIndex {
            attn: vec![0u8; 16],
            ffn: vec![1u8; 16],
            ffn_cache: Arc::new(vec![0.5f32; 8]),
        };
        let dyn_idx: &dyn KvIndex = &idx;
        let attn = dyn_idx.attn_kquant_layer_data(0).unwrap();
        assert_eq!(attn[0].1, "Q4_K");
        assert_eq!(attn[3].1, "Q6_K");
        assert_eq!(attn[0].0.len(), 16);
        let ffn = dyn_idx.interleaved_kquant_layer_data(0).unwrap();
        assert_eq!(ffn.len(), FFN_COMPONENTS_PER_LAYER);
        let mmap = dyn_idx.interleaved_kquant_mmap_ref().unwrap();
        assert_eq!(mmap.len(), 16);
        let cached = dyn_idx.kquant_ffn_layer_once(0, 0).unwrap();
        assert_eq!(cached.len(), 8);
        assert_eq!(dyn_idx.num_features(0), 32);
        assert_eq!(dyn_idx.vocab_size(), 128);
    }

    /// Minimal `KvIndex` that overrides nothing — exercises every trait
    /// default. `EmptyKvIndex` above overrides each method explicitly;
    /// this stub uses the trait bodies themselves so they show as
    /// covered in the report.
    struct DefaultKvIndex;
    impl KvIndex for DefaultKvIndex {}

    #[test]
    fn kv_index_trait_defaults_all_return_none_or_zero() {
        let idx: &dyn KvIndex = &DefaultKvIndex;
        assert_eq!(idx.num_features(0), 0);
        assert_eq!(idx.num_features(42), 0);
        assert!(idx.attn_kquant_layer_data(0).is_none());
        assert!(idx.attn_q8_layer_data(0).is_none());
        assert!(idx.interleaved_kquant_layer_data(0).is_none());
        assert!(idx.interleaved_kquant_mmap_ref().is_none());
        assert!(idx.interleaved_q4_mmap_ref().is_none());
        assert!(idx.kquant_ffn_layer_once(0, 0).is_none());
        assert!(idx.kquant_ffn_layer_once(3, 2).is_none());
        assert_eq!(idx.vocab_size(), 0);
    }

    /// `EmptyKvIndex::attn_q8_layer_data` / `interleaved_q4_mmap_ref`
    /// stay covered (they were already declared override stubs but the
    /// callers in this test module had only exercised the kquant
    /// accessors). Pinning them here keeps the override surface live.
    #[test]
    fn empty_kv_index_overrides_q8_and_legacy_q4_to_none() {
        let idx = EmptyKvIndex;
        let dyn_idx: &dyn KvIndex = &idx;
        assert!(dyn_idx.attn_q8_layer_data(0).is_none());
        assert!(dyn_idx.interleaved_q4_mmap_ref().is_none());
    }

    /// Same for `CannedKvIndex`'s `attn_q8_layer_data` /
    /// `interleaved_q4_mmap_ref` — they're declared overrides returning
    /// None but weren't exercised by the round-trip test above.
    #[test]
    fn canned_kv_index_q8_and_legacy_q4_paths_return_none() {
        let idx = CannedKvIndex {
            attn: vec![0u8; 16],
            ffn: vec![1u8; 16],
            ffn_cache: Arc::new(vec![0.5f32; 8]),
        };
        let dyn_idx: &dyn KvIndex = &idx;
        assert!(dyn_idx.attn_q8_layer_data(0).is_none());
        assert!(dyn_idx.interleaved_q4_mmap_ref().is_none());
    }
}
