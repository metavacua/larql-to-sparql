//! W1-GPU dispatch path for `TurboQuantEngine`.
//!
//! Routes prefill + decode through the backend's
//! `coarse_prefill_with_state` / `coarse_decode_step_with_state`
//! surface. The state-dump payload is per-layer (K_new, V_new),
//! which the engine compresses through its WHT + Lloyd-Max codec
//! into per-layer `CompressedLayer` slots.
//!
//! On decode the path is **append-only** (2026-05-19): each layer's
//! new K/V row is encoded head-by-head and appended onto the
//! existing compressed buffer. This avoids the O(N) decompress +
//! re-compress cycle that earlier flamegraphs surfaced as the
//! per-step bottleneck.

use larql_inference::model::ModelWeights;
use larql_inference::PerLayerDecodeState;
use larql_vindex::VectorIndex;
use ndarray::Array2;

use crate::engines::turbo_quant::engine::{detect_head_dim, CompressedLayer, TurboQuantEngine};

impl TurboQuantEngine {
    /// W1-GPU step 6: prefill via `coarse_prefill_with_state`.
    /// Captured per-layer K/V is compressed into `CompressedLayer`
    /// entries (one per model layer) for the engine's contract.
    pub(super) fn try_prefill_via_dispatch(
        &mut self,
        weights: &ModelWeights,
        index: &VectorIndex,
        token_ids: &[u32],
    ) -> Option<Array2<f32>> {
        if !larql_inference::vindex::supports_cached_decode(weights)
            || !larql_inference::vindex::supports_direct_matvec_decode(weights, index)
        {
            return None;
        }
        let num_layers = weights.num_layers;
        let mut state = PerLayerDecodeState::with_capacity(num_layers);
        let (hidden, handle) = self.backend.as_ref().coarse_prefill_with_state(
            weights,
            token_ids,
            Some(index),
            Some(&mut state),
        )?;
        if !state.is_complete_for(num_layers) {
            return None;
        }
        // W10 Phase A: drain handle vecs and consume each layer's K/V
        // via into_array() — zero-copy move on the CPU happy path.
        self.layers.clear();
        let k_handles = std::mem::take(&mut state.k_new_per_layer);
        let v_handles = std::mem::take(&mut state.v_new_per_layer);
        for (k, v) in k_handles.into_iter().zip(v_handles) {
            let k_arr = k.into_array();
            let v_arr = v.into_array();
            self.layers
                .push(CompressedLayer::compress(&(k_arr, v_arr), &self.tq));
        }
        self.kv_handle = Some(handle);
        self.abs_position = token_ids.len();
        Some(hidden)
    }

    /// W1-GPU step 6: decode through dispatch. State capture gives
    /// us the new K/V row per layer; we encode head-by-head and
    /// append onto the existing `CompressedLayer` slot's compressed
    /// buffer (append-only — 2026-05-19 perf fix).
    pub(super) fn decode_step_via_dispatch(
        &mut self,
        weights: &ModelWeights,
        index: &VectorIndex,
        token_id: u32,
    ) -> Option<Array2<f32>> {
        let t_total = std::time::Instant::now();
        let num_layers = weights.num_layers;
        let mut state = PerLayerDecodeState::with_capacity(num_layers);
        let handle = self.kv_handle.as_mut()?;
        let t_capture = std::time::Instant::now();
        let hidden = self.backend.as_ref().coarse_decode_step_with_state(
            weights,
            token_id,
            Some(index),
            handle,
            self.abs_position,
            Some(&mut state),
        )?;
        if self.profiling {
            self.profile.state_capture.record(t_capture);
        }
        if !state.is_complete_for(num_layers) {
            self.kv_handle = None;
            return None;
        }
        let k_handles = std::mem::take(&mut state.k_new_per_layer);
        let v_handles = std::mem::take(&mut state.v_new_per_layer);
        let t_codec = std::time::Instant::now();
        let mut scratch_f32: Vec<f32> = Vec::new();
        let mut scratch_u8: Vec<u8> = Vec::new();
        for (layer, (k_handle, v_handle)) in k_handles.into_iter().zip(v_handles).enumerate() {
            let k_new_row = k_handle.into_array();
            let v_new_row = v_handle.into_array();
            let arch = &*weights.arch;
            let kv_dim = arch.num_kv_heads_for_layer(layer) * arch.head_dim_for_layer(layer);
            let head_dim = detect_head_dim(kv_dim);
            let layer_slot = &mut self.layers[layer];
            let heads_per_row = kv_dim / head_dim;
            let bytes_per_head = self.tq.bytes_per_vector(head_dim);
            debug_assert_eq!(
                layer_slot.compressed_k.len(),
                layer_slot.num_vecs * heads_per_row * bytes_per_head,
                "compressed_k length out of sync with num_vecs on layer {layer}"
            );
            debug_assert_eq!(layer_slot.kv_dim, kv_dim);
            debug_assert_eq!(layer_slot.head_dim, head_dim);
            layer_slot.append_row(
                k_new_row.as_slice().expect("non-contiguous K row"),
                v_new_row.as_slice().expect("non-contiguous V row"),
                &self.tq,
                &mut scratch_f32,
                &mut scratch_u8,
            );
        }
        if self.profiling {
            self.profile.recompute_hot.record(t_codec);
        }
        self.abs_position += 1;
        if self.profiling {
            self.profile.decode_total.record(t_total);
        }
        Some(hidden)
    }
}

#[cfg(test)]
mod tests {
    //! Coverage for the W1-GPU dispatch path. Drives `CpuBackend` via
    //! the synthetic Q4K fixture (`make_test_q4k_weights` +
    //! `make_test_q4k_vindex`) — the CPU `coarse_*_with_state` impl
    //! populates per-layer state, which the dispatch helpers then
    //! consume into the engine's `CompressedLayer` slots.

    use larql_inference::cpu_engine_backend;
    use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};

    use super::*;
    use crate::engines::turbo_quant::engine::TurboQuantEngine;

    fn fixture() -> (TurboQuantEngine, ModelWeights, VectorIndex) {
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let engine = TurboQuantEngine::with_backend(4, cpu_engine_backend());
        (engine, weights, index)
    }

    #[test]
    fn prefill_via_dispatch_compresses_per_layer_kv() {
        let (mut engine, weights, index) = fixture();
        let h = engine
            .try_prefill_via_dispatch(&weights, &index, &[0u32, 1, 2])
            .expect("prefill via dispatch");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert_eq!(engine.layers.len(), weights.num_layers);
        assert!(engine.kv_handle.is_some());
        assert_eq!(engine.abs_position, 3);
        // Each layer's compressed_k must hold prefill_len heads worth of
        // bytes — non-zero proves the compress() path ran.
        for layer in &engine.layers {
            assert!(!layer.compressed_k.is_empty());
            assert!(!layer.compressed_v.is_empty());
        }
    }

    #[test]
    fn prefill_via_dispatch_returns_none_when_index_lacks_direct_matvec() {
        // An empty / fresh VectorIndex (no Q4K attn payload) fails the
        // `supports_direct_matvec_decode` guard at the top of dispatch.
        let weights = make_test_q4k_weights();
        let empty_index = larql_vindex::VectorIndex::new(
            vec![None; weights.num_layers],
            vec![None; weights.num_layers],
            weights.num_layers,
            weights.hidden_size,
        );
        let mut engine = TurboQuantEngine::with_backend(4, cpu_engine_backend());
        let w = weights;
        assert!(engine
            .try_prefill_via_dispatch(&w, &empty_index, &[0u32, 1])
            .is_none());
        assert!(engine.kv_handle.is_none());
        assert!(engine.layers.is_empty());
    }

    #[test]
    fn prefill_via_dispatch_returns_none_on_empty_prompt() {
        // Empty token slice routes through CpuBackend's empty-tokens
        // guard in `coarse_prefill_with_state` → None.
        let (mut engine, weights, index) = fixture();
        assert!(engine
            .try_prefill_via_dispatch(&weights, &index, &[])
            .is_none());
        assert!(engine.kv_handle.is_none());
    }

    #[test]
    fn decode_step_via_dispatch_appends_one_position() {
        let (mut engine, weights, index) = fixture();
        engine
            .try_prefill_via_dispatch(&weights, &index, &[0u32, 1])
            .expect("prefill");
        let bytes_before: usize = engine.layers.iter().map(|l| l.compressed_k.len()).sum();
        let h = engine
            .decode_step_via_dispatch(&weights, &index, 2)
            .expect("decode");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert_eq!(engine.abs_position, 3);
        let bytes_after: usize = engine.layers.iter().map(|l| l.compressed_k.len()).sum();
        // Append-only encode: one new row per layer per step.
        assert!(
            bytes_after > bytes_before,
            "compressed_k must grow after decode_step (was {bytes_before}, now {bytes_after})"
        );
        // num_vecs counter should advance by exactly 1 on every layer.
        for layer in &engine.layers {
            assert_eq!(layer.num_vecs, 3, "num_vecs should be prefill_len + 1");
        }
    }

    #[test]
    fn decode_step_via_dispatch_without_prefill_returns_none() {
        let (mut engine, weights, index) = fixture();
        // kv_handle is None → early return at `self.kv_handle.as_mut()?`.
        assert!(engine
            .decode_step_via_dispatch(&weights, &index, 0)
            .is_none());
    }

    #[test]
    fn decode_step_via_dispatch_with_profiling_records_stages() {
        let (engine, weights, index) = fixture();
        let mut engine = engine.with_profiling(true);
        engine
            .try_prefill_via_dispatch(&weights, &index, &[0u32, 1])
            .expect("prefill");
        engine
            .decode_step_via_dispatch(&weights, &index, 2)
            .expect("decode");
        // The decode_total / state_capture / recompute_hot timers
        // should all have at least one observation.
        assert!(engine.profile.decode_total.count >= 1);
        assert!(engine.profile.state_capture.count >= 1);
        assert!(engine.profile.recompute_hot.count >= 1);
    }
}
