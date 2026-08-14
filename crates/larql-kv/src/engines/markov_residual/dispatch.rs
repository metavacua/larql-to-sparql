//! W1-GPU dispatch path for `MarkovResidualEngine`.
//!
//! Routes prefill + decode through the backend's
//! `coarse_prefill_with_state` / `coarse_decode_step_with_state_masked`
//! surface, populating the engine's residual store (and optional
//! hot_kv shadow) from per-layer state-dump payloads.
//!
//! W10 mask cascade: `LARQL_W10_HONLY=1` drops the hot_kv shadow on
//! Metal (treating K/V as derivative state); when `window_size =
//! None` it also drops the residual shadow (`stored`). The matching
//! `StateDumpMask` is flowed into the backend call so the kernel
//! skips the corresponding GPU→CPU readback.
//!
//! See `crates/larql-kv/docs/state-policy.md` and PERFORMANCE.md's
//! "W10" section.

use larql_inference::model::ModelWeights;
use larql_inference::PerLayerDecodeState;
use larql_vindex::VectorIndex;
use ndarray::Array2;

use crate::engines::markov_residual::engine::MarkovResidualEngine;
use crate::engines::markov_residual::helpers::{append_row, grow_capacity_2d, window_capacity};
use crate::engines::markov_residual::store::RsStore;

impl MarkovResidualEngine {
    /// W1-GPU: attempt prefill through
    /// `KvDispatch::coarse_prefill_with_state`. Returns `Some(hidden)`
    /// when the backend implements the GPU/fused path; `None` when it
    /// doesn't (engine falls back to per-layer walk).
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
        // W8.2: pre-allocate `stored` and `hot_kv` to a doubling capacity
        // so subsequent decode steps append in-place.
        let prompt_len = token_ids.len();
        let initial_cap = window_capacity(prompt_len, self.window_size);
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
            let hidden_size = weights.hidden_size;
            (0..num_layers)
                .map(|_| Array2::<f32>::zeros((0, hidden_size)))
                .collect()
        } else {
            stored
        };
        let mut rs = RsStore {
            stored,
            cold_residuals: None,
            cold_kv: None,
            cold_len: 0,
            hot_kv: if drop_hot_kv_shadow {
                None
            } else {
                Some(hot_kv)
            },
            cold_abs_start: 0,
            next_position: prompt_len,
            max_window: self.window_size,
            hot_len: if drop_stored_shadow { 0 } else { prompt_len },
        };
        // Clip window on prefill — overflow goes into cold tier via
        // the snapshot helper (already-computed K/V from dispatch).
        let pre_clip: Vec<usize> = if rs.hot_kv.is_some() {
            let window = self.window_size.unwrap_or(usize::MAX);
            let evict_count = rs.hot_len.saturating_sub(window);
            vec![evict_count; rs.stored.len()]
        } else {
            Vec::new()
        };
        let evicted_hot_kv = rs
            .hot_kv
            .as_ref()
            .filter(|_| pre_clip.iter().any(|&n| n > 0))
            .and_then(|h| RsStore::snapshot_evicted_hot_kv(h, &pre_clip));
        let mut cold: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            rs.clip_layer(layer, &mut cold);
        }
        rs.finalise_hot_len_after_clip();
        // 2026-05-19 audit fix: geometric-capacity cold append.
        rs.append_cold_overflow(cold, evicted_hot_kv);
        if rs.cold_len > 0 {
            rs.cold_abs_start = 0;
        }
        self.store = Some(rs);
        self.kv_handle = Some(handle);
        self.abs_position = token_ids.len();
        Some(hidden)
    }

    /// W1-GPU: decode step through
    /// `KvDispatch::coarse_decode_step_with_state_masked`. Per-layer
    /// state is appended to the engine's store/hot_kv on each step
    /// (W8.2 doubling-capacity in-place append).
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
        // W10 mask cascade. On by default; opt out via
        // LARQL_W10_DISABLE=1.
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
        // W8.2: append per-layer h_in / K_new / V_new in-place.
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
                let k_arr_opt = if rs.hot_kv.is_some() {
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
                    if let Some((k_arr, v_arr)) = k_arr_opt {
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
        // Window clip — snapshot-evicted-into-cold flow (W2).
        let pre_clip: Vec<usize> = if rs.hot_kv.is_some() {
            let window = rs.max_window.unwrap_or(usize::MAX);
            let evict_count = rs.hot_len.saturating_sub(window);
            vec![evict_count; rs.stored.len()]
        } else {
            Vec::new()
        };
        let evicted_hot_kv = rs
            .hot_kv
            .as_ref()
            .filter(|_| pre_clip.iter().any(|&n| n > 0))
            .and_then(|h| RsStore::snapshot_evicted_hot_kv(h, &pre_clip));
        let mut overflow: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            rs.clip_layer(layer, &mut overflow);
        }
        rs.finalise_hot_len_after_clip();
        // 2026-05-19 audit fix: geometric-capacity cold append.
        rs.append_cold_overflow(overflow, evicted_hot_kv);
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
    //! Coverage for the W1-GPU dispatch path. Drives `CpuBackend` via
    //! the synthetic Q4K fixture. W10 mask cascade is exercised via
    //! [`crate::engines::set_w10_disabled_override`] — a per-thread
    //! override so tests don't race other parallel tests that also
    //! consult `w10_enabled()`.

    use larql_inference::cpu_engine_backend;
    use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};

    use super::*;
    use crate::engines::markov_residual::engine::MarkovResidualEngine;
    use crate::engines::set_w10_disabled_override;

    fn fixture(window_size: Option<usize>) -> (MarkovResidualEngine, ModelWeights, VectorIndex) {
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let engine = MarkovResidualEngine::with_backend(window_size, cpu_engine_backend());
        (engine, weights, index)
    }

    /// Pin W10 cascade state for this thread. `true` simulates
    /// `LARQL_W10_DISABLE=1` (Full mask); `false` simulates the
    /// default-on cascade (HOnly / None masks).
    fn set_w10_disable(disabled: bool) {
        set_w10_disabled_override(Some(disabled));
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
        let mut engine = MarkovResidualEngine::with_backend(Some(4), cpu_engine_backend());
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
        // window=Some + default env → drop_hot_kv_shadow=true,
        // drop_stored_shadow=false. stored populated, hot_kv dropped.
        // `stored[layer]` is a doubling-capacity buffer
        // (shape `[max(window,prompt_len), hidden]`); the logical row
        // count lives in `hot_len`.
        let (mut engine, weights, index) = fixture(Some(8));
        let h = engine
            .try_prefill_via_dispatch(&weights, &index, &[0u32, 1, 2])
            .expect("prefill");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        let rs = engine.store.as_ref().expect("store populated");
        assert_eq!(rs.stored.len(), weights.num_layers);
        assert!(
            rs.stored[0].shape()[0] >= 3,
            "capacity must hold prefill rows"
        );
        assert!(rs.hot_kv.is_none(), "W10 default drops hot_kv shadow");
        assert_eq!(rs.next_position, 3);
        assert_eq!(rs.hot_len, 3);
        assert!(engine.kv_handle.is_some());
        assert_eq!(engine.abs_position, 3);
    }

    #[test]
    fn try_prefill_via_dispatch_windowless_drops_stored_shadow_under_w10() {
        set_w10_disable(false);
        // window=None + default env → both drop_hot_kv_shadow and
        // drop_stored_shadow = true. The drop_stored branch replaces
        // each `stored[l]` with `Array2::<f32>::zeros((0, hidden))`.
        let (mut engine, weights, index) = fixture(None);
        let hidden = weights.hidden_size;
        engine
            .try_prefill_via_dispatch(&weights, &index, &[0u32, 1, 2])
            .expect("prefill (windowless)");
        let rs = engine.store.as_ref().unwrap();
        assert!(rs.hot_kv.is_none());
        for slab in &rs.stored {
            assert_eq!(slab.shape(), &[0, hidden], "stored slab should be empty");
        }
        assert_eq!(rs.hot_len, 0);
    }

    #[test]
    fn try_prefill_via_dispatch_full_mask_path_with_w10_disabled() {
        // LARQL_W10_DISABLE=1 → drop_hot_kv_shadow=false; both shadows
        // populated (Full mask in decode).
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
        // kv_handle is None → early return at `self.kv_handle.as_mut()?`.
        assert!(engine
            .decode_step_via_dispatch(&weights, &index, 0)
            .is_none());
    }

    #[test]
    fn decode_step_via_dispatch_windowed_appends_h_in_under_honly() {
        set_w10_disable(false);
        // window=Some + default env → mask=HOnly. hot_len grows by 1;
        // hot_kv stays None.
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
        // window=None + default env → mask=None; stored stays empty,
        // hot_len doesn't bump.
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
        // abs_position bumps on every decode regardless of mask.
        assert_eq!(engine.abs_position, 3);
    }

    #[test]
    fn decode_step_via_dispatch_full_mask_appends_hot_kv_with_w10_disabled() {
        // Full mask path: appends to BOTH stored and hot_kv on every layer.
        set_w10_disable(true);
        let (mut engine, weights, index) = fixture(Some(8));
        engine
            .try_prefill_via_dispatch(&weights, &index, &[0u32, 1])
            .expect("prefill");
        let hot_len_before = engine.store.as_ref().unwrap().hot_len;
        let result = engine.decode_step_via_dispatch(&weights, &index, 2);
        set_w10_disable(false);
        result.expect("decode");
        let rs = engine.store.as_ref().unwrap();
        assert_eq!(rs.hot_len, hot_len_before + 1);
        assert!(rs.hot_kv.is_some(), "hot_kv populated under Full mask");
    }

    #[test]
    fn decode_step_via_dispatch_keeps_store_position_cursor_in_sync() {
        // The store's `next_position` is what the walk path reads for
        // RoPE when a dispatch failure forces a fallback mid-generation.
        // If only `self.abs_position` advances, the fallback decodes at
        // a position stale by however many dispatch steps ran.
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
        // HOnly mask path populates state_capture / state_materialise /
        // state_append (the loop that runs under HOnly).
        assert!(engine.profile.decode_total.count >= 1);
        assert!(engine.profile.state_capture.count >= 1);
        assert!(engine.profile.state_materialise.count >= 1);
        assert!(engine.profile.state_append.count >= 1);
    }
}
