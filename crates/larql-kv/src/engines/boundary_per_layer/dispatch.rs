//! W1-GPU dispatch path for `BoundaryPerLayerEngine`.
//!
//! Mirrors `markov_residual`'s dispatch path. The two free functions
//! ([`try_prefill_via_dispatch`] and [`decode_step_via_dispatch`])
//! route through the backend's `coarse_prefill_with_state` /
//! `coarse_decode_step_with_state_masked` surface — on Metal this
//! runs the prompt through the fused per-layer kernel and dumps
//! per-layer `h_in` for the engine to pull into its residual store.
//!
//! Returns `None` (engine should fall back to the dense walk in
//! `super::walk`) when the backend / vindex doesn't support the
//! cached + direct-matvec decode path.
//!
//! **W10 mask cascade** — `boundary_per_layer` never shadows hot
//! K/V (it's recomputed at extend-cold-kv time on overflow), so
//! `LARQL_W10_HONLY=1` is always at least HOnly-safe. When
//! `window_size = None` the residual `stored` is also unused (no
//! cold-tier eviction can fire), so the engine additionally drops
//! it and requests the None mask. Bench (Gemma 3 4B Q4K, M3 Max,
//! 2026-05-21) closes the 13% gap to `standard`'s ~100 tok/s
//! ceiling.

use larql_inference::model::ModelWeights;
use larql_inference::{EngineBackend, KvHandle, PerLayerDecodeState};
use ndarray::Array2;

use crate::engines::boundary_per_layer::cold_tier::{extend_cold_kv_with_overflow, roundtrip};
use crate::engines::boundary_per_layer::policy::BoundaryLayerPolicy;
use crate::engines::boundary_per_layer::store::{PerLayerEncodedColdLayer, RsStorePerLayer};
use crate::engines::markov_residual::recompute_kv;

use crate::engines::w10_enabled as w10_env_on;

/// Run prefill through the W1-GPU dispatch path. Returns
/// `(last_hidden, new_store, kv_handle)` on success; `None` when the
/// backend / vindex lacks the required support (caller falls back to
/// `walk::run_prefill`).
///
/// `dequant_scratch` is the engine-owned f32 scratch: the cold-tier K/V
/// recompute on prefill overflow resolves attention weights through it
/// (real Q4K models carry no dense f32 attention tensors — see
/// [`cold_tier_recompute_view`]).
pub(super) fn try_prefill_via_dispatch(
    weights: &ModelWeights,
    backend: &dyn EngineBackend,
    policy: &BoundaryLayerPolicy,
    window_size: Option<usize>,
    index: &larql_inference::larql_vindex::VectorIndex,
    token_ids: &[u32],
    dequant_scratch: &mut larql_inference::DequantScratch,
) -> Option<(Array2<f32>, RsStorePerLayer, KvHandle)> {
    if !larql_inference::vindex::supports_cached_decode(weights)
        || !larql_inference::vindex::supports_direct_matvec_decode(weights, index)
    {
        return None;
    }
    let num_layers = weights.num_layers;
    let mut state = PerLayerDecodeState::with_capacity(num_layers);
    let (hidden, handle) =
        backend.coarse_prefill_with_state(weights, token_ids, Some(index), Some(&mut state))?;
    if !state.is_complete_for(num_layers) {
        return None;
    }
    let prompt_len = token_ids.len();

    // W10 Phase C: when LARQL_W10_HONLY=1 + window=None, no
    // cold-tier eviction can fire and `rs.stored` is dead weight.
    // Drop it; decode steps will request the None mask, eliminating
    // both K/V and h_in readback. (HOnly without dropping stored is
    // always safe — boundary_per_layer has no hot K/V shadow — but
    // dropping stored is what enables the None-mask path.)
    let drop_stored_shadow = w10_env_on() && window_size.is_none();
    let stored: Vec<Array2<f32>> = if drop_stored_shadow {
        let hidden_size = weights.hidden_size;
        (0..num_layers)
            .map(|_| Array2::<f32>::zeros((0, hidden_size)))
            .collect()
    } else {
        state
            .h_in_per_layer
            .into_iter()
            .map(|h| h.into_array())
            .collect()
    };

    let mut rs = RsStorePerLayer {
        stored,
        cold_encoded: None,
        cold_kv: None,
        hot_kv: None,
        cold_abs_start: 0,
        next_position: prompt_len,
        max_window: window_size,
        policy_codecs: policy.entries.clone(),
    };

    // Prefill-time clip only when we have a non-empty stored. With
    // drop_stored_shadow the stored is empty and clip is a no-op,
    // but we'd panic on indexing `stored[layer]` so just skip.
    if !drop_stored_shadow {
        let mut overflow_per_layer: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            overflow_per_layer.push(rs.clip_layer_overflow(layer));
        }
        if overflow_per_layer.first().map_or(0, |c| c.shape()[0]) > 0 {
            // Cold-tier recompute needs resolvable attention weights; on
            // real Q4K models only the vindex carries them, so dequantise
            // into the engine scratch and thread `index` — mirroring the
            // dense-walk fallback path in `engine::prefill_quant`. A
            // recompute failure aborts the dispatch route (return None →
            // engine falls back to the dense walk) instead of panicking.
            let view = cold_tier_recompute_view(dequant_scratch, weights, index);
            let mut encoded_layers: Vec<PerLayerEncodedColdLayer> = Vec::with_capacity(num_layers);
            let mut cold_kv: Vec<larql_inference::attention::SharedKV> =
                Vec::with_capacity(num_layers);
            for (layer, overflow) in overflow_per_layer.iter().enumerate() {
                let codec = policy.codec_for(layer);
                let decoded_overflow = roundtrip(overflow, codec);
                let (k, v) = recompute_kv(view, &decoded_overflow, layer, 0, backend, Some(index))?;
                cold_kv.push((k, v));
                let mut enc = PerLayerEncodedColdLayer::empty(codec, weights.hidden_size);
                enc.append(overflow);
                encoded_layers.push(enc);
            }
            rs.cold_encoded = Some(encoded_layers);
            rs.cold_kv = Some(cold_kv);
            rs.cold_abs_start = 0;
        }
    }
    Some((hidden, rs, handle))
}

/// Build the `WeightsView` the cold-tier K/V recompute resolves attention
/// weights through: Q4K attention tensors dequantised into the engine
/// scratch (idempotent), overlaid on canonical `weights`. `WeightsView::
/// dense` is NOT sufficient here — real Q4K models have no dense f32
/// attention tensors, so the dense view made `recompute_kv` fail (panic
/// at prefill, silent cold_kv desync at decode) on exactly the models the
/// dispatch path exists for.
fn cold_tier_recompute_view<'a>(
    dequant_scratch: &'a mut larql_inference::DequantScratch,
    weights: &'a ModelWeights,
    index: &larql_inference::larql_vindex::VectorIndex,
) -> larql_inference::WeightsView<'a> {
    larql_inference::vindex::dequant::ensure_attn_tensors_dequantised(
        dequant_scratch,
        weights,
        index,
    );
    larql_inference::WeightsView::with_scratch(weights, dequant_scratch)
}

/// One decode step through the W1-GPU dispatch path. Mutates the
/// supplied `KvHandle` in place (backend appends K/V) and the store in
/// place. `None` signals a state-dump failure — caller should clear its
/// `kv_handle` and fall back to the dense walk.
///
/// Failure invariant: the fallible backend call happens BEFORE any store
/// mutation, so on a `None` return `rs` is untouched — the engine keeps
/// the store and the documented dense-walk fallback stays reachable.
#[allow(clippy::too_many_arguments)]
pub(super) fn decode_step_via_dispatch(
    weights: &ModelWeights,
    backend: &dyn EngineBackend,
    policy: &BoundaryLayerPolicy,
    handle: &mut KvHandle,
    rs: &mut RsStorePerLayer,
    index: &larql_inference::larql_vindex::VectorIndex,
    token_id: u32,
    dequant_scratch: &mut larql_inference::DequantScratch,
) -> Option<Array2<f32>> {
    let num_layers = weights.num_layers;
    let mut state = PerLayerDecodeState::with_capacity(num_layers);
    let abs_position = rs.next_position;

    // W10 mask cascade. boundary_per_layer never shadows hot K/V,
    // so K/V readback is always wasted overhead → drop_hot_kv is
    // unconditionally true when env_on. stored is droppable only
    // when env_on + windowless (the prefill arranged that).
    let env_on = w10_env_on();
    let drop_stored = rs
        .stored
        .first()
        .map(|a| a.shape()[0] == 0)
        .unwrap_or(false)
        && env_on;
    let mask = if drop_stored {
        larql_compute::StateDumpMask::None
    } else if env_on {
        larql_compute::StateDumpMask::HOnly
    } else {
        larql_compute::StateDumpMask::Full
    };

    let hidden = backend.coarse_decode_step_with_state_masked(
        weights,
        token_id,
        Some(index),
        handle,
        abs_position,
        Some(&mut state),
        mask,
    )?;
    if !state.is_complete_under(num_layers, mask) {
        return None;
    }

    // Append h_in to each layer's stored slab (amortised O(m) via
    // push_row). Under None mask, h_in is empty — skip the loop;
    // stored stays the empty Vec from prefill.
    if !matches!(mask, larql_compute::StateDumpMask::None) {
        for (layer, h) in state.h_in_per_layer.into_iter().enumerate() {
            let h_arr = h.into_array();
            rs.stored[layer]
                .push_row(h_arr.row(0))
                .expect("push_row shape mismatch");
        }
    }
    rs.next_position = abs_position + 1;

    // Cold-tier eviction + cold_kv extension. Under None mask there's
    // no stored to evict from; skip.
    if matches!(mask, larql_compute::StateDumpMask::None) {
        return Some(hidden);
    }
    let mut overflow_per_layer: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        overflow_per_layer.push(rs.clip_layer_overflow(layer));
    }
    if overflow_per_layer.first().map_or(0, |c| c.shape()[0]) > 0 {
        let cold_abs_pos =
            rs.cold_abs_start + rs.cold_encoded.as_ref().map_or(0, |l| l[0].n_positions);
        match rs.cold_encoded.as_mut() {
            Some(layers) => {
                for (layer, overflow) in overflow_per_layer.iter().enumerate() {
                    layers[layer].append(overflow);
                }
            }
            None => {
                let hidden_size = weights.hidden_size;
                let mut layers: Vec<PerLayerEncodedColdLayer> = Vec::with_capacity(num_layers);
                for (layer, overflow) in overflow_per_layer.iter().enumerate() {
                    let codec = policy.codec_for(layer);
                    let mut enc = PerLayerEncodedColdLayer::empty(codec, hidden_size);
                    enc.append(overflow);
                    layers.push(enc);
                }
                rs.cold_encoded = Some(layers);
            }
        }
        // Same scratch-backed view as the prefill overflow path: dense
        // resolution fails on real Q4K models (no dense attn tensors).
        let view = cold_tier_recompute_view(dequant_scratch, weights, index);
        if extend_cold_kv_with_overflow(
            view,
            backend,
            policy,
            rs,
            &overflow_per_layer,
            cold_abs_pos,
        )
        .is_none()
        {
            // Per-layer K/V recompute failed: the helper dropped `cold_kv`
            // wholesale (atomicity — no layer desync possible), so the next
            // decode step rebuilds cold K/V from `cold_encoded`.
            debug_assert!(rs.cold_kv.is_none(), "failed extend must drop cold_kv");
        }
    }
    Some(hidden)
}

#[cfg(test)]
mod tests {
    //! Coverage for the W1-GPU dispatch free functions. Drives
    //! `CpuBackend` via the synthetic Q4K fixture so the per-layer
    //! `coarse_*_with_state` populates a `PerLayerDecodeState` the
    //! helpers then consume into `RsStorePerLayer`.
    //!
    //! The W10 mask cascade (`drop_stored_shadow` /
    //! `StateDumpMask::None`) is on by default; tests exercise both
    //! the windowed (`HOnly`) and windowless (`None`) shapes.

    use larql_inference::cpu_engine_backend;
    use larql_inference::test_utils::{
        make_test_q4k_vindex, make_test_q4k_weights, Q4K_TEST_NUM_LAYERS,
    };

    use super::*;
    use crate::engines::boundary_per_layer::policy::BoundaryLayerPolicy;

    fn bf16_policy() -> BoundaryLayerPolicy {
        BoundaryLayerPolicy::bf16_uniform("test", Q4K_TEST_NUM_LAYERS)
    }

    /// Clear the per-thread W10 cascade override so the engine reads
    /// the (unset, default-on) env. Tests call this at the start to
    /// neutralise overrides leaked by earlier tests on the same thread.
    fn clear_w10_override() {
        crate::engines::set_w10_disabled_override(None);
    }

    #[test]
    fn try_prefill_via_dispatch_returns_none_when_index_lacks_direct_matvec() {
        clear_w10_override();
        let weights = make_test_q4k_weights();
        let empty_index = larql_vindex::VectorIndex::new(
            vec![None; weights.num_layers],
            vec![None; weights.num_layers],
            weights.num_layers,
            weights.hidden_size,
        );
        let backend = cpu_engine_backend();
        let w = weights;
        assert!(try_prefill_via_dispatch(
            &w,
            backend.as_ref(),
            &bf16_policy(),
            Some(4),
            &empty_index,
            &[0u32, 1],
            &mut larql_inference::DequantScratch::new(),
        )
        .is_none());
    }

    #[test]
    fn try_prefill_via_dispatch_windowed_populates_store_under_w10_honly() {
        clear_w10_override();
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = cpu_engine_backend();
        let (h, rs, _handle) = try_prefill_via_dispatch(
            &weights,
            backend.as_ref(),
            &bf16_policy(),
            Some(4),
            &index,
            &[0u32, 1, 2],
            &mut larql_inference::DequantScratch::new(),
        )
        .expect("prefill via dispatch");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        // W10 default: HOnly mask is selected (drop_hot_kv unconditional,
        // drop_stored_shadow only when window_size = None). Windowed
        // configuration keeps stored populated.
        assert_eq!(rs.stored.len(), weights.num_layers);
        assert_eq!(rs.stored[0].shape()[0], 3);
        assert_eq!(rs.next_position, 3);
    }

    #[test]
    fn try_prefill_via_dispatch_windowless_drops_stored_under_w10_none_mask() {
        clear_w10_override();
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = cpu_engine_backend();
        let (_h, rs, _handle) = try_prefill_via_dispatch(
            &weights,
            backend.as_ref(),
            &bf16_policy(),
            None,
            &index,
            &[0u32, 1, 2],
            &mut larql_inference::DequantScratch::new(),
        )
        .expect("prefill via dispatch (windowless)");
        // W10 + window=None: drop_stored_shadow is true → empty stored
        // per layer.
        for slab in &rs.stored {
            assert_eq!(slab.shape()[0], 0, "stored should be empty under None mask");
        }
        assert!(rs.cold_encoded.is_none());
    }

    #[test]
    fn decode_step_via_dispatch_appends_h_in_under_honly() {
        clear_w10_override();
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = cpu_engine_backend();
        let mut scratch = larql_inference::DequantScratch::new();
        let (_h, mut rs, mut handle) = try_prefill_via_dispatch(
            &weights,
            backend.as_ref(),
            &bf16_policy(),
            Some(4),
            &index,
            &[0u32, 1],
            &mut scratch,
        )
        .expect("prefill");
        let rows_before = rs.stored[0].shape()[0];
        let h = decode_step_via_dispatch(
            &weights,
            backend.as_ref(),
            &bf16_policy(),
            &mut handle,
            &mut rs,
            &index,
            2,
            &mut scratch,
        )
        .expect("decode");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        // HOnly mask appends one h_in row per layer per step.
        assert_eq!(rs.stored[0].shape()[0], rows_before + 1);
        assert_eq!(rs.next_position, 3);
    }

    #[test]
    fn decode_step_via_dispatch_windowless_takes_none_mask_path() {
        clear_w10_override();
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = cpu_engine_backend();
        let mut scratch = larql_inference::DequantScratch::new();
        let (_h, mut rs, mut handle) = try_prefill_via_dispatch(
            &weights,
            backend.as_ref(),
            &bf16_policy(),
            None,
            &index,
            &[0u32, 1],
            &mut scratch,
        )
        .expect("prefill (windowless)");
        decode_step_via_dispatch(
            &weights,
            backend.as_ref(),
            &bf16_policy(),
            &mut handle,
            &mut rs,
            &index,
            2,
            &mut scratch,
        )
        .expect("decode (None mask)");
        // None mask: stored stays empty, but next_position still advances.
        for slab in &rs.stored {
            assert_eq!(slab.shape()[0], 0);
        }
        assert_eq!(rs.next_position, 3);
    }

    #[test]
    fn cold_tier_recompute_survives_missing_dense_attn_tensors() {
        // Production Q4K models carry NO dense f32 attention tensors — only
        // the vindex's Q4K bytes. The cold-tier K/V recompute must therefore
        // resolve attention weights via the dequant scratch / Q4K-direct
        // route, not via `WeightsView::dense` (which panicked pre-fix).
        clear_w10_override();
        let mut weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights); // built before stripping
        let attn_keys: Vec<String> = (0..weights.num_layers)
            .flat_map(|l| {
                [
                    weights.arch.attn_q_key(l),
                    weights.arch.attn_k_key(l),
                    weights.arch.attn_v_key(l),
                    weights.arch.attn_o_key(l),
                ]
            })
            .collect();
        for k in attn_keys {
            weights.tensors.remove(&k);
        }
        let backend = cpu_engine_backend();
        let mut scratch = larql_inference::DequantScratch::new();
        // window=2 + 3-token prompt → prefill-time overflow → cold K/V
        // recompute fires during try_prefill_via_dispatch.
        let (h, mut rs, mut handle) = try_prefill_via_dispatch(
            &weights,
            backend.as_ref(),
            &bf16_policy(),
            Some(2),
            &index,
            &[0u32, 1, 2],
            &mut scratch,
        )
        .expect("prefill via dispatch must survive without dense attn tensors");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        let cold_kv = rs.cold_kv.as_ref().expect("prefill overflow → cold_kv");
        for (k, v) in cold_kv {
            assert_eq!(k.shape()[0], 1, "1 overflow row per layer");
            assert_eq!(v.shape()[0], 1);
        }
        // Decode-time overflow: the decode-path cold K/V extension must also
        // survive (pre-fix it failed silently, desyncing cold_kv).
        decode_step_via_dispatch(
            &weights,
            backend.as_ref(),
            &bf16_policy(),
            &mut handle,
            &mut rs,
            &index,
            3,
            &mut scratch,
        )
        .expect("decode via dispatch");
        let cold_kv = rs.cold_kv.as_ref().expect("decode overflow keeps cold_kv");
        let n_cold = rs.cold_encoded.as_ref().unwrap()[0].n_positions;
        for (k, v) in cold_kv {
            assert_eq!(
                k.shape()[0],
                n_cold,
                "cold_kv rows must track cold_encoded positions"
            );
            assert_eq!(v.shape()[0], n_cold);
        }
    }

    #[test]
    fn decode_step_via_dispatch_overflow_extends_cold_tier() {
        clear_w10_override();
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = cpu_engine_backend();
        // window=2: after prefilling 2 tokens, one decode crosses the
        // window and the dispatch eviction path runs.
        let mut scratch = larql_inference::DequantScratch::new();
        let (_h, mut rs, mut handle) = try_prefill_via_dispatch(
            &weights,
            backend.as_ref(),
            &bf16_policy(),
            Some(2),
            &index,
            &[0u32, 1],
            &mut scratch,
        )
        .expect("prefill");
        assert!(rs.cold_encoded.is_none(), "no overflow at prefill");
        decode_step_via_dispatch(
            &weights,
            backend.as_ref(),
            &bf16_policy(),
            &mut handle,
            &mut rs,
            &index,
            2,
            &mut scratch,
        )
        .expect("decode");
        assert!(
            rs.cold_encoded.is_some(),
            "first decode past window should fire cold-tier append"
        );
        // Subsequent decode should extend an existing cold_encoded
        // (Some(layers) branch of the match).
        decode_step_via_dispatch(
            &weights,
            backend.as_ref(),
            &bf16_policy(),
            &mut handle,
            &mut rs,
            &index,
            3,
            &mut scratch,
        )
        .expect("decode 2");
        let n_cold = rs.cold_encoded.as_ref().unwrap()[0].n_positions;
        assert!(n_cold >= 2);
        // cold_kv must track cold_encoded row-for-row on every layer
        // (atomic extend contract).
        let cold_kv = rs.cold_kv.as_ref().expect("overflow keeps cold_kv");
        for (k, v) in cold_kv {
            assert_eq!(k.shape()[0], n_cold, "cold K rows must match cold_encoded");
            assert_eq!(v.shape()[0], n_cold, "cold V rows must match cold_encoded");
        }
    }
}
