//! W1-GPU dispatch path for `WindowedCheckpointEngine`.
//!
//! Routes prefill + decode through the backend's
//! `coarse_prefill_with_state` / `coarse_decode_step_with_state_masked`
//! surface. The state-dump payload (per-layer K_new + V_new) lands in
//! the engine's pre-allocated `current_window_kv` slabs (W8 — single
//! `slice_mut(...).assign(row)` per layer per step, no per-step
//! `Array2::zeros` allocation).
//!
//! Window auto-close fires at `current_window_tokens.len() >=
//! window_size`, archiving + checkpointing the closed window. W10
//! mask cascade: `LARQL_W10_HONLY=1` drops the engine-side K/V
//! shadow → Metal's kv cache is the truth, `close_window` reads back
//! the final row via `KvDispatch::read_kv_row_at`.
//!
//! **Cross-backend attention divergence (documented, not fixed):** the
//! backend kv cache behind `kv_handle` is never trimmed at window
//! close — it spans the whole stream since prefill — so dispatch-path
//! attention is over the FULL history. The CPU walk path re-seeds each
//! window from the single-row boundary checkpoint, attending over
//! `checkpoint + current window` only. Hidden states (and therefore
//! K/V rows and checkpoint values) diverge between the two paths from
//! the first window close onward; only the window BOOKKEEPING (window
//! ids, archived token spans, absolute offsets, checkpoint positions)
//! is guaranteed to agree across backends.

use larql_inference::attention::SharedKV;
use larql_inference::model::ModelWeights;
use larql_inference::PerLayerDecodeState;
use larql_vindex::VectorIndex;
use ndarray::{s, Array2};

use crate::engines::windowed_checkpoint::engine::WindowedCheckpointEngine;

impl WindowedCheckpointEngine {
    /// W1-GPU step 4: prefill via `coarse_prefill_with_state`. The
    /// per-layer K/V dump is unpacked into pre-allocated
    /// `[window_size, kv_dim]` buffers so subsequent decode steps
    /// append a single row in-place rather than re-allocating.
    ///
    /// The prompt is segmented into `⌈prompt/window_size⌉` windows
    /// exactly like the CPU `process()` path: every full window is
    /// checkpointed + archived here (same window ids, token spans, and
    /// absolute offsets), and the remainder opens the next window — so
    /// cross-backend window bookkeeping agrees and the open window
    /// never exceeds `window_size`.
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
        // The KvHandle created below indexes backend cache positions
        // from 0; the engine's absolute bookkeeping must start aligned
        // with it or every later absolute-row readback (`close_window`
        // under HOnly) would be shifted. A non-fresh engine declines
        // and falls back to the CPU walk path.
        if self.abs_offset != 0
            || !self.current_window_tokens.is_empty()
            || self.kv_handle.is_some()
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
        let prompt_len = token_ids.len();
        let full_windows = prompt_len / self.window_size;
        let open_len = prompt_len % self.window_size;
        let open_start = prompt_len - open_len;
        // W10 Phase B: drop the engine-side current_window_kv shadow.
        // On by default since 2026-05-21; opt out via
        // LARQL_W10_DISABLE=1. Metal's kv cache is the truth.
        let drop_window_kv_shadow = crate::engines::w10_enabled();

        // Per-layer prompt K/V (`[prompt_len, kv_dim]` each), needed for
        // the full windows' boundary checkpoints and — when the shadow
        // is kept — for the open window's buffers. When neither is
        // needed the handles are dropped unmaterialised (the W10 fast
        // path). into_array() is a zero-copy move on the CPU happy path.
        let kv_src: Option<Vec<SharedKV>> = if full_windows > 0 || !drop_window_kv_shadow {
            Some(
                state
                    .k_new_per_layer
                    .into_iter()
                    .zip(state.v_new_per_layer)
                    .map(|(k_h, v_h)| (k_h.into_array(), v_h.into_array()))
                    .collect(),
            )
        } else {
            drop((state.k_new_per_layer, state.v_new_per_layer));
            None
        };

        // Close every full window the prompt spans, mirroring
        // `close_window`'s bookkeeping. `abs_offset` starts at 0 (fresh-
        // stream guard above), so absolute positions equal row indices
        // in the state dump.
        for _ in 0..full_windows {
            let abs_end = self.abs_offset + self.window_size - 1;
            let kv_src = kv_src
                .as_ref()
                .expect("kv_src materialised when full_windows > 0");
            let ckpt: Vec<SharedKV> = kv_src
                .iter()
                .map(|(k, v)| {
                    (
                        k.slice(s![abs_end..abs_end + 1, ..]).to_owned(),
                        v.slice(s![abs_end..abs_end + 1, ..]).to_owned(),
                    )
                })
                .collect();
            self.checkpoints.save(self.current_window_id, ckpt, abs_end);
            self.archive.archive(
                self.current_window_id,
                token_ids[self.abs_offset..self.abs_offset + self.window_size].to_vec(),
                self.abs_offset,
            );
            self.abs_offset += self.window_size;
            self.current_window_id += 1;
        }

        if drop_window_kv_shadow {
            self.current_window_kv = None;
        } else {
            let kv_src = kv_src.expect("kv_src materialised when shadow is kept");
            let kv: Vec<SharedKV> = kv_src
                .iter()
                .map(|(k_src, v_src)| {
                    let kv_dim = k_src.shape()[1];
                    let mut k_buf = Array2::<f32>::zeros((self.window_size, kv_dim));
                    let mut v_buf = Array2::<f32>::zeros((self.window_size, kv_dim));
                    if open_len > 0 {
                        k_buf
                            .slice_mut(s![..open_len, ..])
                            .assign(&k_src.slice(s![open_start.., ..]));
                        v_buf
                            .slice_mut(s![..open_len, ..])
                            .assign(&v_src.slice(s![open_start.., ..]));
                    }
                    (k_buf, v_buf)
                })
                .collect();
            self.current_window_kv = Some(kv);
        }
        self.current_window_kv_len = open_len;
        self.current_window_tokens = token_ids[open_start..].to_vec();
        self.last_hidden = Some(hidden.clone());
        self.kv_handle = Some(handle);
        // The dump above ran over the whole prompt, so the handle holds every
        // row. Drop everything the window does not entitle attention to see.
        self.clip_handle_to_window(open_len, full_windows > 0);
        Some(hidden)
    }

    /// Clip the backend K/V cache down to what this engine is allowed to
    /// attend over: the open window, plus the boundary row a closed window
    /// leaves behind as the next one's seed.
    ///
    /// **This is the window.** Everything else `window_size` touches —
    /// segmenting the prompt, sizing the shadow buffers, deciding when to
    /// archive — is bookkeeping the engine does to itself. Attention reads
    /// the backend handle, so a handle that is never clipped means the window
    /// bounds storage and not behaviour, and `info()`'s `window=N` is a claim
    /// the engine does not honour (issue #200, measured in `larql/kvperf-1`:
    /// the engine had the same decode cost slope as the unwindowed one across
    /// four context lengths, and window sizes of 32 / 256 / 4096 were
    /// indistinguishable).
    ///
    /// `clip_kv` keeps the *tail*, which is exactly right here: after a close
    /// the row we must retain is the closing window's last position, and that
    /// is the row the checkpoint was taken from. Clipping to 1 therefore
    /// reproduces on the dispatch path what `extend_current` does on the
    /// per-layer path when it seeds a fresh window from `checkpoints.load`.
    fn clip_handle_to_window(&mut self, open_len: usize, any_window_closed: bool) {
        let keep = if any_window_closed {
            open_len + 1
        } else {
            open_len
        };
        let Some(handle) = self.kv_handle.as_mut() else {
            return;
        };
        if handle.cached_len() > keep {
            self.backend.as_ref().clip_kv(handle, keep);
        }
    }

    /// W1-GPU step 4: decode through dispatch. State capture gives us
    /// the new K/V row per layer; we append in-place to
    /// `current_window_kv` and trigger window auto-close when token
    /// count crosses `window_size`.
    pub(super) fn decode_step_via_dispatch(
        &mut self,
        weights: &ModelWeights,
        index: &VectorIndex,
        token_id: u32,
    ) -> Option<Array2<f32>> {
        let num_layers = weights.num_layers;
        let mut state = PerLayerDecodeState::with_capacity(num_layers);
        let abs_position = self.abs_offset + self.current_window_tokens.len();
        let handle = self.kv_handle.as_mut()?;
        // W10 Phase B: HOnly when the window shadow was dropped at
        // prefill. close_window() reads back via KvDispatch.
        let want_h_only = self.current_window_kv.is_none() && crate::engines::w10_enabled();
        let mask = if want_h_only {
            larql_compute::StateDumpMask::HOnly
        } else {
            larql_compute::StateDumpMask::Full
        };
        let hidden = self.backend.as_ref().coarse_decode_step_with_state_masked(
            weights,
            token_id,
            Some(index),
            handle,
            abs_position,
            Some(&mut state),
            mask,
        )?;
        if !state.is_complete_under(num_layers, mask) {
            self.kv_handle = None;
            return None;
        }
        // W8: in-place row append into the pre-allocated buffers
        // (single `slice_mut().assign(row)` per layer per side).
        let pos = self.current_window_kv_len;
        if !matches!(mask, larql_compute::StateDumpMask::HOnly) {
            // First step of a fresh window: the previous auto-close
            // consumed the shadow (`close_window` takes it), so
            // re-allocate the new window's `[window_size, kv_dim]`
            // buffers exactly as prefill does. Under Full mask the
            // shadow can only be missing at a window start.
            if self.current_window_kv.is_none() {
                debug_assert_eq!(pos, 0, "window shadow missing mid-window under Full mask");
                let kv: Vec<SharedKV> = state
                    .k_new_per_layer
                    .iter()
                    .zip(&state.v_new_per_layer)
                    .take(num_layers)
                    .map(|(k_h, v_h)| {
                        (
                            Array2::<f32>::zeros((self.window_size, k_h.shape().1)),
                            Array2::<f32>::zeros((self.window_size, v_h.shape().1)),
                        )
                    })
                    .collect();
                self.current_window_kv = Some(kv);
            }
            let window_kv = self
                .current_window_kv
                .as_mut()
                .expect("window shadow allocated above");
            debug_assert!(
                pos < window_kv[0].0.shape()[0],
                "current_window_kv_len {pos} >= buffer capacity {} — \
                 window auto-close should have fired before this",
                window_kv[0].0.shape()[0]
            );
            let k_handles = std::mem::take(&mut state.k_new_per_layer);
            let v_handles = std::mem::take(&mut state.v_new_per_layer);
            for (slot, (k_handle, v_handle)) in window_kv
                .iter_mut()
                .zip(k_handles.into_iter().zip(v_handles))
                .take(num_layers)
            {
                let k_new_row = k_handle.into_array();
                let v_new_row = v_handle.into_array();
                slot.0.slice_mut(s![pos..pos + 1, ..]).assign(&k_new_row);
                slot.1.slice_mut(s![pos..pos + 1, ..]).assign(&v_new_row);
            }
        }
        self.current_window_kv_len = pos + 1;
        self.current_window_tokens.push(token_id);
        self.last_hidden = Some(hidden.clone());

        // Window auto-close: same trigger as the legacy process loop.
        if self.current_window_tokens.len() >= self.window_size {
            self.close_window();
            // The closed window's rows are archived and checkpointed; only its
            // last row may cross into the next window. Without this the handle
            // grows for the whole stream and the window means nothing.
            self.clip_handle_to_window(0, true);
        }
        Some(hidden)
    }
}

#[cfg(test)]
mod tests {
    //! Coverage for the W1-GPU dispatch path. `WindowedCheckpointEngine`
    //! takes a non-optional `window_size: usize`; W10 mask cascade is
    //! gated on whether `current_window_kv` is dropped. Tests pin the
    //! cascade state via [`crate::engines::set_w10_disabled_override`]
    //! — a per-thread override so they don't race other parallel tests
    //! that also call `w10_enabled()`.

    use larql_inference::cpu_engine_backend;
    use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};

    use super::*;
    use crate::engines::windowed_checkpoint::engine::WindowedCheckpointEngine;

    fn fixture(window_size: usize) -> (WindowedCheckpointEngine, ModelWeights, VectorIndex) {
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let engine = WindowedCheckpointEngine::with_backend(window_size, cpu_engine_backend());
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
        let mut engine = WindowedCheckpointEngine::with_backend(4, cpu_engine_backend());
        let w = weights;
        assert!(engine
            .try_prefill_via_dispatch(&w, &empty_index, &[0u32, 1])
            .is_none());
        assert!(engine.kv_handle.is_none());
        assert!(engine.current_window_kv.is_none());
    }

    #[test]
    fn try_prefill_via_dispatch_drops_window_kv_under_w10_default() {
        // W10 on by default → drop_window_kv_shadow = true. The engine
        // doesn't populate `current_window_kv`; Metal's kv cache is the
        // truth.
        set_w10_disable(false);
        let (mut engine, weights, index) = fixture(4);
        let h = engine
            .try_prefill_via_dispatch(&weights, &index, &[0u32, 1, 2])
            .expect("prefill");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(engine.current_window_kv.is_none());
        assert_eq!(engine.current_window_kv_len, 3);
        assert_eq!(engine.current_window_tokens, vec![0u32, 1, 2]);
        assert!(engine.kv_handle.is_some());
        assert!(engine.last_hidden.is_some());
    }

    #[test]
    fn try_prefill_via_dispatch_populates_window_kv_with_w10_disabled() {
        // W10 off → engine pre-allocates `[window_cap, kv_dim]` per
        // layer and copies the prefill K/V rows in.
        set_w10_disable(true);
        let (mut engine, weights, index) = fixture(8);
        let res = engine.try_prefill_via_dispatch(&weights, &index, &[0u32, 1, 2]);
        set_w10_disable(false);
        res.expect("prefill");
        let kv = engine
            .current_window_kv
            .as_ref()
            .expect("W10-disabled keeps window_kv");
        assert_eq!(kv.len(), weights.num_layers);
        // Buffer is `[window_cap, kv_dim]` where window_cap = max(window, prompt_len) = 8.
        assert_eq!(kv[0].0.shape()[0], 8);
        assert_eq!(engine.current_window_kv_len, 3);
    }

    #[test]
    fn decode_step_via_dispatch_without_prefill_returns_none() {
        set_w10_disable(false);
        let (mut engine, weights, index) = fixture(4);
        // kv_handle is None → early return.
        assert!(engine
            .decode_step_via_dispatch(&weights, &index, 0)
            .is_none());
    }

    #[test]
    fn decode_step_via_dispatch_h_only_skips_kv_append() {
        // W10 default + window_kv dropped at prefill → HOnly mask.
        // The dispatch decode runs through the `if !matches!(mask, HOnly)`
        // skip branch and still bumps the per-step counters.
        set_w10_disable(false);
        let (mut engine, weights, index) = fixture(4);
        engine
            .try_prefill_via_dispatch(&weights, &index, &[0u32, 1])
            .expect("prefill");
        let kv_len_before = engine.current_window_kv_len;
        let h = engine
            .decode_step_via_dispatch(&weights, &index, 2)
            .expect("decode");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert_eq!(engine.current_window_kv_len, kv_len_before + 1);
        assert_eq!(engine.current_window_tokens.last(), Some(&2u32));
        assert!(
            engine.current_window_kv.is_none(),
            "HOnly mask keeps window_kv None"
        );
    }

    #[test]
    fn decode_step_via_dispatch_full_mask_appends_kv_with_w10_disabled() {
        // W10 off → Full mask. Each layer's K/V row blits into the
        // pre-allocated window_kv buffer at `current_window_kv_len`.
        set_w10_disable(true);
        let (mut engine, weights, index) = fixture(8);
        engine
            .try_prefill_via_dispatch(&weights, &index, &[0u32, 1])
            .expect("prefill");
        let kv_len_before = engine.current_window_kv_len;
        let res = engine.decode_step_via_dispatch(&weights, &index, 2);
        set_w10_disable(false);
        res.expect("decode");
        assert_eq!(engine.current_window_kv_len, kv_len_before + 1);
        let kv = engine.current_window_kv.as_ref().unwrap();
        // The new row sits at position `kv_len_before` (= 2) of every
        // layer's buffer. Non-zero rows imply the assign actually ran.
        for slot in kv {
            let row = slot.0.row(kv_len_before);
            let any_non_zero = row.iter().any(|v| *v != 0.0);
            assert!(
                any_non_zero,
                "K row at position {kv_len_before} should be populated"
            );
        }
    }

    #[test]
    fn decode_step_via_dispatch_fires_window_auto_close() {
        // window=2: prefill 2 → token count == window → close_window
        // fires inside the dispatch decode and resets the window.
        set_w10_disable(false);
        let (mut engine, weights, index) = fixture(2);
        engine
            .try_prefill_via_dispatch(&weights, &index, &[0u32])
            .expect("prefill");
        let cp_before = engine.checkpoints.len();
        engine
            .decode_step_via_dispatch(&weights, &index, 1)
            .expect("decode that triggers window-close");
        // close_window() emits a checkpoint and resets
        // current_window_tokens (the legacy emit path).
        assert!(
            engine.checkpoints.len() > cp_before,
            "window auto-close should emit a checkpoint"
        );
    }

    /// The backend's kv row at an absolute position — ground truth for
    /// checkpoint value assertions.
    ///
    /// Reads the **last** row the handle holds, not an absolute stream index:
    /// since issue #200 the dispatch path clips the handle to the window, so
    /// absolute positions no longer index it and the boundary row is simply
    /// its tail.
    fn backend_last_row(engine: &WindowedCheckpointEngine, layer: usize) -> (Vec<f32>, Vec<f32>) {
        let handle = engine.kv_handle.as_ref().expect("kv handle");
        let row = handle.cached_len().saturating_sub(1);
        engine
            .backend
            .as_ref()
            .read_kv_row_at(handle, layer, row)
            .expect("backend kv row readback")
    }

    /// Each window checkpoints **its own** last row.
    ///
    /// The failure this guards has been reachable two different ways. First,
    /// `close_window` under HOnly read a window-relative index into a handle
    /// that spanned the whole stream, so every window after the first
    /// re-checkpointed the *first* window's row. That was fixed by reading an
    /// absolute position — correct only while the window was not enforced.
    /// Now that the handle is clipped to the window (issue #200) the absolute
    /// index runs off the end, and the boundary row is the handle's tail.
    ///
    /// Both readings share one observable claim, asserted here so neither
    /// mechanism can regress: distinct windows must checkpoint distinct rows,
    /// each recorded at its own absolute stream position.
    #[test]
    fn h_only_checkpoints_each_windows_own_last_row() {
        set_w10_disable(false);
        const WINDOW: usize = 2;
        const CLOSED_WINDOWS: usize = 2;
        let (mut engine, weights, index) = fixture(WINDOW);
        engine
            .try_prefill_via_dispatch(&weights, &index, &[0u32])
            .expect("prefill");
        // Token 1 closes window 0 (abs 0..=1); token 3 closes window 1
        // (abs 2..=3).
        for tok in 1u32..=3 {
            engine
                .decode_step_via_dispatch(&weights, &index, tok)
                .expect("decode");
        }
        assert_eq!(engine.archive.len(), CLOSED_WINDOWS);
        for window_id in 0..CLOSED_WINDOWS {
            let (ckpt, abs_end) = engine.checkpoints.load(window_id).expect("checkpoint");
            let expected_abs_end = (window_id + 1) * WINDOW - 1;
            assert_eq!(abs_end, expected_abs_end, "window {window_id} abs position");
            assert_eq!(ckpt.len(), weights.num_layers, "one K/V row per layer");
        }
        // The handle is clipped, which is what makes the tail read correct.
        let cached = engine.kv_handle.as_ref().expect("handle").cached_len();
        assert!(
            cached <= window_bound(WINDOW),
            "handle holds {cached} rows past a {WINDOW}-token window"
        );
        // The two boundary rows must actually differ or the index
        // assertions above would be vacuous.
        let (c0, _) = engine.checkpoints.load(0).unwrap();
        let (c1, _) = engine.checkpoints.load(1).unwrap();
        assert_ne!(
            c0[0].0, c1[0].0,
            "distinct windows must checkpoint distinct K rows"
        );
    }

    /// Regression (decode across an auto-close with W10 disabled):
    /// pre-fix the first Full-mask decode after a window auto-close
    /// panicked — `close_window` consumed the shadow and nothing
    /// re-allocated it. Also pins value parity between the two shadow
    /// modes: identical hidden states and identical checkpoint rows
    /// whether checkpoints come from the engine-side shadow (W10 off)
    /// or the backend readback (W10 on).
    #[test]
    fn decode_across_window_close_matches_between_w10_modes() {
        const WINDOW: usize = 2;
        const PREFILL: [u32; 1] = [0];
        // Token 1 closes window 0; token 2 is the step that pre-fix
        // panicked (Full mask, shadow consumed); token 3 closes window 1.
        const DECODE: [u32; 3] = [1, 2, 3];
        const CLOSED_WINDOWS: usize = 2;
        let run = |w10_disabled: bool| {
            set_w10_disable(w10_disabled);
            let (mut engine, weights, index) = fixture(WINDOW);
            let result = (|| {
                let mut hiddens =
                    vec![engine.try_prefill_via_dispatch(&weights, &index, &PREFILL)?];
                for &tok in &DECODE {
                    hiddens.push(engine.decode_step_via_dispatch(&weights, &index, tok)?);
                }
                Some(hiddens)
            })();
            set_w10_disable(false);
            (engine, result.expect("dispatch prefill+decode run"))
        };
        let (engine_off, hiddens_off) = run(true);
        let (engine_on, hiddens_on) = run(false);
        assert_eq!(engine_off.archive.len(), CLOSED_WINDOWS);
        assert_eq!(engine_on.archive.len(), CLOSED_WINDOWS);
        for (step, (h_off, h_on)) in hiddens_off.iter().zip(&hiddens_on).enumerate() {
            assert_eq!(
                h_off.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                h_on.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "step {step}: hidden state depends on the W10 shadow mode"
            );
        }
        for window_id in 0..CLOSED_WINDOWS {
            let (c_off, abs_off) = engine_off
                .checkpoints
                .load(window_id)
                .expect("ckpt w10-off");
            let (c_on, abs_on) = engine_on.checkpoints.load(window_id).expect("ckpt w10-on");
            assert_eq!(abs_off, abs_on, "window {window_id} abs position");
            assert_eq!(c_off.len(), c_on.len(), "window {window_id} layer count");
            for (layer, ((k_off, v_off), (k_on, v_on))) in c_off.iter().zip(&c_on).enumerate() {
                assert_eq!(
                    k_off, k_on,
                    "window {window_id} layer {layer}: shadow-path K checkpoint \
                     != readback-path K checkpoint"
                );
                assert_eq!(
                    v_off, v_on,
                    "window {window_id} layer {layer}: shadow-path V checkpoint \
                     != readback-path V checkpoint"
                );
            }
        }
    }

    /// Regression (prompt segmentation): pre-fix the whole prompt was
    /// stored as ONE open window, so with W10 disabled and
    /// prompt_len >= window_size the first decode indexed past the
    /// shadow buffer's capacity. Dispatch prefill must split the prompt
    /// into ⌈prompt/W⌉ windows with the same ids, token spans, offsets,
    /// and checkpoint positions as the CPU walk path.
    #[test]
    fn prefill_via_dispatch_segments_prompt_like_cpu_path() {
        const WINDOW: usize = 2;
        const PROMPT: [u32; 5] = [0, 1, 2, 3, 4];
        const NEXT_TOKEN: u32 = 5;
        const TOTAL_WINDOWS: usize = 3;
        set_w10_disable(true);
        let (mut engine, weights, index) = fixture(WINDOW);
        let res = engine.try_prefill_via_dispatch(&weights, &index, &PROMPT);
        // One more decode fills window 2 — pre-fix this indexed out of
        // bounds (pos == capacity) because no window ever closed.
        let dec = res
            .as_ref()
            .and_then(|_| engine.decode_step_via_dispatch(&weights, &index, NEXT_TOKEN));
        set_w10_disable(false);
        res.expect("segmented dispatch prefill");
        dec.expect("decode after full-prompt prefill");

        // CPU reference: same token stream through the walk path. Only
        // BOOKKEEPING is compared — checkpoint values legitimately
        // diverge (full-history vs checkpoint+window attention; see the
        // module doc).
        let mut cpu = WindowedCheckpointEngine::new(WINDOW);
        cpu.process(&weights, &PROMPT, None).expect("cpu prefill");
        cpu.process(&weights, &[NEXT_TOKEN], None)
            .expect("cpu decode");

        assert_eq!(engine.archive.len(), TOTAL_WINDOWS);
        assert_eq!(engine.archive.len(), cpu.archive.len());
        assert_eq!(engine.current_window_id, cpu.current_window_id);
        assert_eq!(engine.abs_offset, cpu.abs_offset);
        assert_eq!(engine.current_window_tokens, cpu.current_window_tokens);
        for window_id in 0..TOTAL_WINDOWS {
            let (t_d, o_d) = engine
                .archive
                .retrieve(window_id)
                .expect("dispatch archive");
            let (t_c, o_c) = cpu.archive.retrieve(window_id).expect("cpu archive");
            assert_eq!(t_d, t_c, "window {window_id} token span");
            assert_eq!(o_d, o_c, "window {window_id} abs offset");
            let (_, abs_d) = engine.checkpoints.load(window_id).expect("dispatch ckpt");
            let (_, abs_c) = cpu.checkpoints.load(window_id).expect("cpu ckpt");
            assert_eq!(abs_d, abs_c, "window {window_id} checkpoint position");
        }
        // Every window carries a full per-layer checkpoint, and the most
        // recently closed one matches what the (clipped) handle still holds.
        for window_id in 0..TOTAL_WINDOWS {
            let (ckpt, _) = engine.checkpoints.load(window_id).unwrap();
            assert_eq!(ckpt.len(), weights.num_layers);
        }
        let (last_ckpt, _) = engine.checkpoints.load(TOTAL_WINDOWS - 1).unwrap();
        for (layer, (k, v)) in last_ckpt.iter().enumerate() {
            let (k_true, v_true) = backend_last_row(&engine, layer);
            assert_eq!(k.row(0).to_vec(), k_true, "layer {layer}: checkpoint K");
            assert_eq!(v.row(0).to_vec(), v_true, "layer {layer}: checkpoint V");
        }
    }

    /// A non-fresh engine (existing window tokens / offset / handle)
    /// declines dispatch prefill — the handle's position space must
    /// start aligned with the engine's absolute bookkeeping.
    #[test]
    fn prefill_via_dispatch_declines_on_non_fresh_engine() {
        set_w10_disable(false);
        let (mut engine, weights, index) = fixture(4);
        engine.current_window_tokens = vec![9];
        assert!(engine
            .try_prefill_via_dispatch(&weights, &index, &[0u32, 1])
            .is_none());
        engine.current_window_tokens.clear();
        engine.abs_offset = 4;
        assert!(engine
            .try_prefill_via_dispatch(&weights, &index, &[0u32, 1])
            .is_none());
    }

    // ── The window must bound attention, not merely storage ──────────────
    //
    // Regression pins for the defect measured in `kvperf-1` (issue #200): the
    // coarse dispatch trait carries no window parameter, so a windowed engine
    // that appends to a backend handle and never clips it attends over the
    // whole stream while `info()` reports `window=N`. That is a correctness
    // failure first — the engine is not the engine it claims to be — and it
    // showed up as this engine having the *same* decode cost slope as the
    // unwindowed one across four context lengths.
    //
    // Row count is the right thing to assert: it is what attention reads, and
    // it is observable without timing. The bound is `window_size + 1` because
    // a closed window leaves its final row behind as the next window's
    // boundary checkpoint — which is the engine's whole design.

    /// Longest cache a correctly windowed engine may present to attention:
    /// the open window, plus the boundary row carried in from the last close.
    fn window_bound(window_size: usize) -> usize {
        window_size + 1
    }

    #[test]
    fn dispatch_prefill_clips_the_backend_cache_to_the_window() {
        set_w10_disable(false);
        const WINDOW: usize = 4;
        const WINDOWS_WORTH: u32 = 5;
        let (mut engine, weights, index) = fixture(WINDOW);
        let prompt: Vec<u32> = (0..WINDOW as u32 * WINDOWS_WORTH).collect();

        engine
            .try_prefill_via_dispatch(&weights, &index, &prompt)
            .expect("prefill");

        let cached = engine.kv_handle.as_ref().expect("handle").cached_len();
        assert!(
            cached <= window_bound(WINDOW),
            "a {WINDOW}-token window prefilled with {} tokens left {cached} rows in the \
             backend cache — attention sees all of them, so the window bounds storage \
             but not what the engine actually attends over",
            prompt.len()
        );
    }

    #[test]
    fn dispatch_decode_keeps_the_backend_cache_within_the_window() {
        set_w10_disable(false);
        const WINDOW: usize = 4;
        const DECODE_STEPS: u32 = 12;
        let (mut engine, weights, index) = fixture(WINDOW);
        engine
            .try_prefill_via_dispatch(&weights, &index, &[0u32, 1, 2])
            .expect("prefill");

        // Enough steps to cross several window closes.
        for tok in 3..3 + DECODE_STEPS {
            engine
                .decode_step_via_dispatch(&weights, &index, tok)
                .expect("decode");
            let cached = engine.kv_handle.as_ref().expect("handle").cached_len();
            assert!(
                cached <= window_bound(WINDOW),
                "after token {tok} the backend cache holds {cached} rows, above the \
                 {WINDOW}-token window — the cache grows without bound across window \
                 closes"
            );
        }
    }
}
