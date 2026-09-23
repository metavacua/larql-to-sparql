//! W1-GPU dispatch path for `MarkovResidualCodecEngine`.
//!
//! Mirrors `MarkovResidualEngine`'s dispatch path. The two methods
//! ([`MarkovResidualCodecEngine::try_prefill_via_dispatch`] and
//! [`MarkovResidualCodecEngine::decode_step_via_dispatch`]) route
//! through the backend's per-layer state-dump kernel; on overflow the
//! payload is encoded onto the bf16 cold tier and `cold_kv` is
//! invalidated (codec round-trip is lossy → next decode recomputes
//! against the decoded bytes).
//!
//! W10 mask cascade is implemented here: `LARQL_W10_HONLY=1` drops
//! the `hot_kv` shadow and (when `window_size = None`) also the
//! `stored` shadow, with the corresponding `StateDumpMask` flowed
//! into the backend call. See `crates/larql-kv/docs/state-policy.md`.

use larql_inference::model::ModelWeights;
use larql_inference::PerLayerDecodeState;
use ndarray::Array2;

use crate::engines::markov_residual_codec::engine::MarkovResidualCodecEngine;
use crate::engines::markov_residual_codec::helpers::{
    append_row, grow_capacity_2d, window_capacity,
};
use crate::engines::markov_residual_codec::store::{EncodedColdLayer, RsStoreCodec};

impl MarkovResidualCodecEngine {
    /// W1-GPU: mirrors `MarkovResidualEngine::try_prefill_via_dispatch`.
    /// On the codec engine the prefill payload is identical (stored +
    /// hot_kv from state.h_in / k_new / v_new). The cold tier
    /// (`cold_encoded`) is codec-encoded; on overflow we still
    /// invalidate `cold_kv` because codec round-trip is lossy.
    pub(super) fn try_prefill_via_dispatch(
        &mut self,
        weights: &ModelWeights,
        index: &larql_inference::larql_vindex::VectorIndex,
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
        let hidden_size = weights.hidden_size;
        // W8.2: pre-allocate doubling-capacity stored / hot_kv buffers.
        let prompt_len = token_ids.len();
        let initial_cap = window_capacity(prompt_len, self.window_size);
        // W10 Phase A: consume each layer's handle into an owned Array2;
        // CpuStateHandle moves without a copy.
        let stored: Vec<Array2<f32>> = state
            .h_in_per_layer
            .into_iter()
            .map(|h| grow_capacity_2d(&h.into_array(), prompt_len, initial_cap))
            .collect();
        let hot_kv: Vec<larql_inference::attention::SharedKV> = state
            .k_new_per_layer
            .into_iter()
            .zip(state.v_new_per_layer)
            .map(|(k, v)| {
                (
                    grow_capacity_2d(&k.into_array(), prompt_len, initial_cap),
                    grow_capacity_2d(&v.into_array(), prompt_len, initial_cap),
                )
            })
            .collect();
        // W10 Phase B/C: drop shadows. On by default since 2026-05-21
        // (set LARQL_W10_DISABLE=1 to opt out — debug instrument).
        let drop_hot_kv_shadow = crate::engines::w10_enabled();
        let drop_stored_shadow = drop_hot_kv_shadow && self.window_size.is_none();
        let stored = if drop_stored_shadow {
            (0..num_layers)
                .map(|_| ndarray::Array2::<f32>::zeros((0, hidden_size)))
                .collect()
        } else {
            stored
        };
        let mut rs = RsStoreCodec {
            stored,
            cold_encoded: None,
            cold_kv: None,
            hot_kv: if drop_hot_kv_shadow {
                None
            } else {
                Some(hot_kv)
            },
            cold_abs_start: 0,
            next_position: prompt_len,
            max_window: self.window_size,
            codec: self.codec,
            hot_len: if drop_stored_shadow { 0 } else { prompt_len },
        };
        // Clip on prefill — overflow encoded into the bf16 cold tier.
        let mut overflow_per_layer: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            overflow_per_layer.push(rs.clip_layer_overflow(layer));
        }
        rs.finalise_hot_len_after_clip();
        if overflow_per_layer.first().map_or(0, |c| c.shape()[0]) > 0 {
            let mut encoded_layers: Vec<EncodedColdLayer> = Vec::with_capacity(num_layers);
            for overflow in overflow_per_layer.iter() {
                let mut enc = EncodedColdLayer::empty(hidden_size);
                enc.append(self.codec, overflow);
                encoded_layers.push(enc);
            }
            rs.cold_encoded = Some(encoded_layers);
            // Codec is lossy → cold_kv must be recomputed against the
            // decoded bytes on next decode. Leave as None.
            rs.cold_abs_start = 0;
        }
        self.store = Some(rs);
        self.kv_handle = Some(handle);
        self.abs_position = token_ids.len();
        Some(hidden)
    }

    /// W1-GPU: codec decode through the dispatch surface. Same shape
    /// as `MarkovResidualEngine::decode_step_via_dispatch` but the
    /// overflow path encodes into `cold_encoded` (bf16) and clears
    /// `cold_kv` so the next step recomputes against the decoded bytes.
    pub(super) fn decode_step_via_dispatch(
        &mut self,
        weights: &ModelWeights,
        index: &larql_inference::larql_vindex::VectorIndex,
        token_id: u32,
    ) -> Option<Array2<f32>> {
        let t_total = std::time::Instant::now();
        let num_layers = weights.num_layers;
        let hidden_size = weights.hidden_size;
        let mut state = PerLayerDecodeState::with_capacity(num_layers);
        let handle = self.kv_handle.as_mut()?;
        // W10 Phase B/C: same mask selection as MarkovResidualEngine.
        // On by default; opt out via LARQL_W10_DISABLE=1.
        let env_on = crate::engines::w10_enabled();
        let drop_hot_kv = self
            .store
            .as_ref()
            .map(|s| s.hot_kv.is_none())
            .unwrap_or(false)
            && env_on;
        let drop_stored = self
            .store
            .as_ref()
            .map(|s| s.stored.first().map(|a| a.shape()[0] == 0).unwrap_or(false))
            .unwrap_or(false)
            && env_on;
        let mask = if drop_stored && drop_hot_kv {
            larql_compute::StateDumpMask::None
        } else if drop_hot_kv {
            larql_compute::StateDumpMask::HOnly
        } else {
            larql_compute::StateDumpMask::Full
        };
        let t_capture = std::time::Instant::now();
        let hidden = self.backend.as_ref().coarse_decode_step_with_state_masked(
            weights,
            token_id,
            Some(index),
            handle,
            self.abs_position,
            Some(&mut state),
            mask,
        )?;
        if self.profiling {
            self.profile.state_capture.record(t_capture);
        }
        if !state.is_complete_under(num_layers, mask) {
            self.kv_handle = None;
            return None;
        }
        let mut rs = self.store.take()?;
        let len = rs.hot_len;
        let h_handles = std::mem::take(&mut state.h_in_per_layer);
        let k_handles = std::mem::take(&mut state.k_new_per_layer);
        let v_handles = std::mem::take(&mut state.v_new_per_layer);
        let did_append = !matches!(mask, larql_compute::StateDumpMask::None);
        if matches!(mask, larql_compute::StateDumpMask::None) {
            drop((h_handles, k_handles, v_handles));
        } else if matches!(mask, larql_compute::StateDumpMask::HOnly) {
            drop((k_handles, v_handles));
            for (layer, h) in h_handles.into_iter().enumerate() {
                let t_mat = std::time::Instant::now();
                let h_arr = h.into_array();
                if self.profiling {
                    self.profile.state_materialise.record(t_mat);
                }
                let t_app = std::time::Instant::now();
                append_row(&mut rs.stored[layer], &h_arr, len);
                if self.profiling {
                    self.profile.state_append.record(t_app);
                }
            }
        } else {
            for (layer, ((h, k), v)) in h_handles
                .into_iter()
                .zip(k_handles)
                .zip(v_handles)
                .enumerate()
            {
                let t_mat = std::time::Instant::now();
                let h_arr = h.into_array();
                let kv_arrs = if rs.hot_kv.is_some() {
                    Some((k.into_array(), v.into_array()))
                } else {
                    None
                };
                if self.profiling {
                    self.profile.state_materialise.record(t_mat);
                }
                let t_app = std::time::Instant::now();
                append_row(&mut rs.stored[layer], &h_arr, len);
                if let Some(hot_kv) = rs.hot_kv.as_mut() {
                    if let Some((k_arr, v_arr)) = kv_arrs {
                        append_row(&mut hot_kv[layer].0, &k_arr, len);
                        append_row(&mut hot_kv[layer].1, &v_arr, len);
                    }
                }
                if self.profiling {
                    self.profile.state_append.record(t_app);
                }
            }
        }
        if did_append {
            rs.hot_len = len + 1;
        }
        // Clip + bf16-encode the overflow.
        let mut overflow_per_layer: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            overflow_per_layer.push(rs.clip_layer_overflow(layer));
        }
        rs.finalise_hot_len_after_clip();
        if overflow_per_layer.first().map_or(0, |c| c.shape()[0]) > 0 {
            match rs.cold_encoded.as_mut() {
                Some(layers) => {
                    for (layer, overflow) in overflow_per_layer.iter().enumerate() {
                        layers[layer].append(rs.codec, overflow);
                    }
                }
                None => {
                    let mut layers: Vec<EncodedColdLayer> = Vec::with_capacity(num_layers);
                    for overflow in overflow_per_layer.iter() {
                        let mut enc = EncodedColdLayer::empty(hidden_size);
                        enc.append(rs.codec, overflow);
                        layers.push(enc);
                    }
                    rs.cold_encoded = Some(layers);
                }
            }
            // Lossy codec → invalidate cold_kv.
            rs.cold_kv = None;
        }
        self.abs_position += 1;
        // Keep the store's own position cursor in sync: a dispatch
        // failure on a later step falls back to the walk path, which
        // reads `rs.next_position` for RoPE.
        rs.next_position = self.abs_position;
        self.store = Some(rs);
        if self.profiling {
            self.profile.decode_total.record(t_total);
        }
        Some(hidden)
    }
}

#[cfg(test)]
mod tests {
    //! Coverage for the W1-GPU dispatch path on the codec engine.
    //! Mirrors `markov_residual::dispatch::tests` — same `CpuBackend`
    //! synthetic Q4K driver, same W10 cascade asserts (HOnly under
    //! windowed default, None under windowless, Full under
    //! `LARQL_W10_DISABLE=1`). Differences: overflow path encodes into
    //! `cold_encoded` (bf16) rather than raw `cold_residuals`, and
    //! `cold_kv` is invalidated on overflow (codec round-trip is lossy).

    use larql_inference::cpu_engine_backend;
    use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};

    use super::*;
    use crate::engines::markov_residual_codec::codec::ColdResidualCodec;
    use crate::engines::markov_residual_codec::engine::MarkovResidualCodecEngine;

    fn fixture(
        window_size: Option<usize>,
    ) -> (
        MarkovResidualCodecEngine,
        ModelWeights,
        larql_vindex::VectorIndex,
    ) {
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let engine = MarkovResidualCodecEngine::with_backend(
            window_size,
            ColdResidualCodec::Bf16,
            cpu_engine_backend(),
        );
        (engine, weights, index)
    }

    fn set_w10_disable(disabled: bool) {
        crate::engines::set_w10_disabled_override(Some(disabled));
    }

    #[test]
    fn try_prefill_via_dispatch_returns_none_when_index_lacks_direct_matvec() {
        set_w10_disable(false);
        let weights = make_test_q4k_weights();
        let empty_index = larql_vindex::VectorIndex::new(
            vec![None; weights.num_layers],
            vec![None; weights.num_layers],
            weights.num_layers,
            weights.hidden_size,
        );
        let mut engine = MarkovResidualCodecEngine::with_backend(
            Some(4),
            ColdResidualCodec::Bf16,
            cpu_engine_backend(),
        );
        let w = weights;
        assert!(engine
            .try_prefill_via_dispatch(&w, &empty_index, &[0u32, 1])
            .is_none());
        assert!(engine.store.is_none());
        assert!(engine.kv_handle.is_none());
    }

    #[test]
    fn try_prefill_via_dispatch_windowed_keeps_stored_under_w10_default() {
        set_w10_disable(false);
        let (mut engine, weights, index) = fixture(Some(8));
        let h = engine
            .try_prefill_via_dispatch(&weights, &index, &[0u32, 1, 2])
            .expect("prefill");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        let rs = engine.store.as_ref().unwrap();
        assert!(rs.hot_kv.is_none(), "W10 default drops hot_kv shadow");
        assert!(rs.stored[0].shape()[0] >= 3);
        assert_eq!(rs.hot_len, 3);
        assert_eq!(rs.next_position, 3);
        assert!(rs.cold_encoded.is_none(), "no overflow yet");
        assert_eq!(engine.abs_position, 3);
    }

    #[test]
    fn try_prefill_via_dispatch_windowless_drops_stored_under_w10() {
        set_w10_disable(false);
        let (mut engine, weights, index) = fixture(None);
        let hidden = weights.hidden_size;
        engine
            .try_prefill_via_dispatch(&weights, &index, &[0u32, 1, 2])
            .expect("prefill (windowless)");
        let rs = engine.store.as_ref().unwrap();
        assert!(rs.hot_kv.is_none());
        for slab in &rs.stored {
            assert_eq!(slab.shape(), &[0, hidden]);
        }
        assert_eq!(rs.hot_len, 0);
    }

    #[test]
    fn try_prefill_via_dispatch_windowed_overflow_codecs_cold_tier() {
        // window=2 against 4 tokens → prefill clip evicts 2 positions
        // into `cold_encoded` via the bf16 codec.
        set_w10_disable(false);
        let (mut engine, weights, index) = fixture(Some(2));
        engine
            .try_prefill_via_dispatch(&weights, &index, &[0u32, 1, 2, 3])
            .expect("prefill (overflow)");
        let rs = engine.store.as_ref().unwrap();
        let cold = rs.cold_encoded.as_ref().expect("cold_encoded populated");
        assert_eq!(cold.len(), weights.num_layers);
        assert!(cold[0].n_positions > 0);
        // Codec is lossy → cold_kv stays None.
        assert!(rs.cold_kv.is_none());
    }

    #[test]
    fn try_prefill_via_dispatch_full_mask_with_w10_disabled() {
        set_w10_disable(true);
        let (mut engine, weights, index) = fixture(Some(8));
        let res = engine.try_prefill_via_dispatch(&weights, &index, &[0u32, 1, 2]);
        set_w10_disable(false);
        res.expect("prefill");
        let rs = engine.store.as_ref().unwrap();
        assert!(rs.hot_kv.is_some(), "W10 disabled keeps hot_kv");
        assert_eq!(rs.hot_len, 3);
    }

    #[test]
    fn decode_step_via_dispatch_without_prefill_returns_none() {
        set_w10_disable(false);
        let (mut engine, weights, index) = fixture(Some(4));
        assert!(engine
            .decode_step_via_dispatch(&weights, &index, 0)
            .is_none());
    }

    #[test]
    fn decode_step_via_dispatch_windowed_appends_h_in_under_honly() {
        set_w10_disable(false);
        let (mut engine, weights, index) = fixture(Some(8));
        engine
            .try_prefill_via_dispatch(&weights, &index, &[0u32, 1])
            .expect("prefill");
        let hot_len_before = engine.store.as_ref().unwrap().hot_len;
        let h = engine
            .decode_step_via_dispatch(&weights, &index, 2)
            .expect("decode");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        let rs = engine.store.as_ref().unwrap();
        assert_eq!(rs.hot_len, hot_len_before + 1);
        assert!(rs.hot_kv.is_none());
        assert_eq!(engine.abs_position, 3);
    }

    #[test]
    fn decode_step_via_dispatch_windowless_uses_none_mask() {
        set_w10_disable(false);
        let (mut engine, weights, index) = fixture(None);
        let hidden = weights.hidden_size;
        engine
            .try_prefill_via_dispatch(&weights, &index, &[0u32, 1])
            .expect("prefill (windowless)");
        engine
            .decode_step_via_dispatch(&weights, &index, 2)
            .expect("decode (None mask)");
        let rs = engine.store.as_ref().unwrap();
        for slab in &rs.stored {
            assert_eq!(slab.shape(), &[0, hidden]);
        }
        assert_eq!(rs.hot_len, 0);
        assert_eq!(engine.abs_position, 3);
    }

    #[test]
    fn decode_step_via_dispatch_full_mask_with_w10_disabled() {
        set_w10_disable(true);
        let (mut engine, weights, index) = fixture(Some(8));
        engine
            .try_prefill_via_dispatch(&weights, &index, &[0u32, 1])
            .expect("prefill");
        let hot_len_before = engine.store.as_ref().unwrap().hot_len;
        let res = engine.decode_step_via_dispatch(&weights, &index, 2);
        set_w10_disable(false);
        res.expect("decode");
        let rs = engine.store.as_ref().unwrap();
        assert_eq!(rs.hot_len, hot_len_before + 1);
        assert!(rs.hot_kv.is_some());
    }

    #[test]
    fn decode_step_via_dispatch_overflow_extends_cold_encoded() {
        // window=2: after prefilling 2 → one decode evicts the oldest
        // position into cold_encoded (the None match arm constructs it).
        // Second decode extends an existing cold_encoded (Some arm).
        set_w10_disable(false);
        let (mut engine, weights, index) = fixture(Some(2));
        engine
            .try_prefill_via_dispatch(&weights, &index, &[0u32, 1])
            .expect("prefill");
        assert!(engine.store.as_ref().unwrap().cold_encoded.is_none());
        engine
            .decode_step_via_dispatch(&weights, &index, 2)
            .expect("decode 1");
        assert!(engine.store.as_ref().unwrap().cold_encoded.is_some());
        let n_before = engine
            .store
            .as_ref()
            .unwrap()
            .cold_encoded
            .as_ref()
            .unwrap()[0]
            .n_positions;
        engine
            .decode_step_via_dispatch(&weights, &index, 3)
            .expect("decode 2");
        let n_after = engine
            .store
            .as_ref()
            .unwrap()
            .cold_encoded
            .as_ref()
            .unwrap()[0]
            .n_positions;
        assert!(
            n_after > n_before,
            "second decode should extend cold_encoded"
        );
    }

    #[test]
    fn decode_step_via_dispatch_keeps_store_position_cursor_in_sync() {
        // Twin of the markov_residual test: `rs.next_position` feeds the
        // walk fallback's RoPE after a dispatch failure and must track
        // `self.abs_position` step for step.
        set_w10_disable(false);
        let (mut engine, weights, index) = fixture(Some(8));
        engine
            .try_prefill_via_dispatch(&weights, &index, &[0u32, 1])
            .expect("prefill");
        for tok in 2..5u32 {
            engine
                .decode_step_via_dispatch(&weights, &index, tok)
                .expect("decode");
            let rs = engine.store.as_ref().unwrap();
            assert_eq!(
                rs.next_position, engine.abs_position,
                "store position cursor fell behind the dispatch cursor"
            );
        }
        assert_eq!(engine.abs_position, 5);
    }

    #[test]
    fn decode_step_via_dispatch_with_profiling_records_stages() {
        set_w10_disable(false);
        let (engine, weights, index) = fixture(Some(8));
        let mut engine = engine.with_profiling(true);
        engine
            .try_prefill_via_dispatch(&weights, &index, &[0u32, 1])
            .expect("prefill");
        engine
            .decode_step_via_dispatch(&weights, &index, 2)
            .expect("decode");
        assert!(engine.profile.decode_total.count >= 1);
        assert!(engine.profile.state_capture.count >= 1);
        assert!(engine.profile.state_materialise.count >= 1);
        assert!(engine.profile.state_append.count >= 1);
    }
}
