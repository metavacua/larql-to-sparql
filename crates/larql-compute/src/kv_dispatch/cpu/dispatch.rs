//! The `impl KvDispatch for CpuBackend` block. Split from
//! `kv_dispatch/cpu.rs` — see `cpu/mod.rs` for the module doc.

use crate::CpuBackend;
use ndarray::Array2;

use super::handles::{
    cpu_handle, cpu_handle_mut, cpu_residual, try_cpu_q4k_cache_mut, CpuKvHandle,
    CpuQ4kCacheHandle, CpuResidualHandle,
};
use crate::attention::{run_attention_block_decode_step_backend, run_attention_with_kv_backend};
use crate::kv_dispatch::{KvDispatch, KvHandle, ResidualHandle};
use larql_models::ModelWeights;

// Opt-in flag (`LARQL_Q4K_DIRECT_ATTN`) lives in `attention::decode` — single
// source shared with `run_attention_block_decode_step_auto` so the dispatch
// path and the engine walk loops make the same per-layer choice.
//
// ⚠️ SIBLING PATH: `cached_decode_step_q4k` / `CpuQ4kCacheHandle` (in
// `crate::kquant_forward`) is a SECOND independent CPU Q4K attention decode.
// Any RoPE / softcap / QK-V-norm / GQA change here must mirror there (and vice
// versa) or the two silently diverge. Consolidate before either is load-bearing
// — see `docs/diagnoses/q4k-direct-attention.md` §"CONSOLIDATION HAZARD".
use crate::attention::decode::q4k_direct_attn_enabled;

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

// ─── KvDispatch impl ────────────────────────────────────────────────────────

impl KvDispatch for CpuBackend {
    fn alloc_kv_buffer(&self, layer: usize, _max_tokens: usize, kv_dim: usize) -> KvHandle {
        // `max_tokens` is informational on CPU — we grow the buffer on
        // append rather than pre-allocate. GPU backends will pre-allocate.
        KvHandle::new(CpuKvHandle::new(layer, kv_dim))
    }

    fn append_kv(&self, handle: &mut KvHandle, k_row: &[f32], v_row: &[f32], _abs_position: usize) {
        // `abs_position` is informational on CPU — the K/V buffer is
        // ordered by insertion, and RoPE rotations are applied by the
        // caller (or by attention_step's underlying function).
        let h = cpu_handle_mut(handle);
        h.append_row(k_row, v_row);
    }

    /// Keep the tail `window_size` rows.
    ///
    /// Handles **both** cache shapes this backend allocates: the coarse
    /// `CpuQ4kCacheHandle` (one handle for the whole model, a per-layer
    /// `[rows, kv_dim]` pair inside) and the per-layer `CpuKvHandle`. Only the
    /// second was handled before, so a windowed engine running on the coarse
    /// path had its clip panic as a "foreign handle" — or, where the engine
    /// never called clip at all, silently attended over its whole stream while
    /// reporting a bounded window (issue #200).
    fn clip_kv(&self, handle: &mut KvHandle, window_size: usize) {
        if let Some(coarse) = try_cpu_q4k_cache_mut(handle) {
            for slot in coarse.cache.iter_mut() {
                let Some((k, v)) = slot.as_mut() else {
                    continue;
                };
                let rows = k.shape()[0];
                if rows > window_size {
                    let start = rows - window_size;
                    *k = k.slice(ndarray::s![start.., ..]).to_owned();
                    *v = v.slice(ndarray::s![start.., ..]).to_owned();
                }
            }
            return;
        }
        let h = cpu_handle_mut(handle);
        if h.rows > window_size {
            let start = h.rows - window_size;
            let kv_dim = h.kv_dim;
            h.k_buf.copy_within(start * kv_dim..h.rows * kv_dim, 0);
            h.v_buf.copy_within(start * kv_dim..h.rows * kv_dim, 0);
            h.rows = window_size;
            h.k_buf.truncate(h.rows * kv_dim);
            h.v_buf.truncate(h.rows * kv_dim);
        }
    }

    fn truncate_kv(&self, handle: &mut KvHandle, len: usize) -> bool {
        let h = cpu_handle_mut(handle);
        // Growing only: `len` above the current row count would be asking
        // this to invent rows, which is the one thing a rewind must not do.
        if len > h.rows {
            return false;
        }
        let kv_dim = h.kv_dim;
        h.rows = len;
        h.k_buf.truncate(len * kv_dim);
        h.v_buf.truncate(len * kv_dim);
        true
    }

    fn read_kv_to_host(&self, handle: &KvHandle) -> Option<(Array2<f32>, Array2<f32>)> {
        cpu_handle(handle).to_shared()
    }

    fn attention_step(
        &self,
        weights: larql_models::WeightsView,
        query: &Array2<f32>,
        kv: &mut KvHandle,
        layer: usize,
        abs_position: usize,
        index: Option<&dyn crate::KvIndex>,
    ) -> Option<Array2<f32>> {
        // Opt-in Q4K-direct path (`LARQL_Q4K_DIRECT_ATTN=1`, task #16): project
        // the new token (no cache access), APPEND the new K/V row to the
        // handle's growable buffers in place (amortised O(kv_dim) — no O(ctx)
        // copy), then attend over zero-copy views of the full cache. Falls
        // back per layer to the f32 path when the index lacks Q4K attn bytes.
        if let Some(idx) = index.filter(|_| q4k_direct_attn_enabled()) {
            if let Some(proj) = crate::attention::decode::decode_step_project_q4k_direct(
                weights.canonical(),
                query,
                layer,
                abs_position,
                self,
                idx,
            ) {
                let h = cpu_handle_mut(kv);
                let k_row = proj
                    .k_new_rope
                    .as_slice()
                    .expect("[1, kv_dim] projection row is contiguous");
                let v_row = proj
                    .v_new
                    .as_slice()
                    .expect("[1, kv_dim] projection row is contiguous");
                h.append_row(k_row, v_row);
                let (k_all, v_all) = h.views().expect("non-empty after append");
                match crate::attention::decode::decode_step_attend_q4k_direct(
                    weights.canonical(),
                    query,
                    layer,
                    &proj.q_rope,
                    k_all,
                    v_all,
                    self,
                    idx,
                ) {
                    Some(h_post_attn) => return Some(h_post_attn),
                    None => {
                        // Attend failed after the append — undo it so the f32
                        // fallback (and any engine-level fallback that reuses
                        // this handle) sees the pre-step state, matching the
                        // old monolithic form's failure semantics.
                        cpu_handle_mut(kv).pop_row();
                    }
                }
            }
        }

        // Default (f32) path: CpuBackend reads f32 attention tensors out of
        // `weights.tensors`, which the caller pre-populates via
        // `ensure_attn_tensors_dequantised` (the up-front dequant-to-f32 tax).
        // The prior state is COPIED out (not moved) so a backend failure
        // leaves the handle intact — same semantics as the pre-refactor
        // clone, and this path is cold whenever the q4k flags are on.
        let prior_kv = cpu_handle(kv).to_shared();
        let (h_post_attn, new_kv) = run_attention_block_decode_step_backend(
            weights,
            query,
            layer,
            prior_kv.as_ref(),
            abs_position,
            Some(self),
        )?;
        cpu_handle_mut(kv).replace_state(new_kv);
        Some(h_post_attn)
    }

    fn attention_prefill(
        &self,
        weights: larql_models::WeightsView,
        tokens_embedded: &Array2<f32>,
        layer: usize,
        _window: Option<usize>,
        _index: Option<&dyn crate::KvIndex>,
    ) -> Option<(Array2<f32>, KvHandle)> {
        // See `attention_step` doc for the `_index` convention.
        let (h_post_attn, k_rope, v) =
            run_attention_with_kv_backend(weights, tokens_embedded, layer, Some(self), None)?;
        let phase = crate::phase_timing::start();
        let kv_dim = k_rope.shape()[1];
        let mut handle = CpuKvHandle::new(layer, kv_dim);
        handle.replace_state((k_rope, v));
        let handle = KvHandle::new(handle);
        crate::phase_timing::finish(phase, "attn.kv_handle_write");
        Some((h_post_attn, handle))
    }

    fn upload_boundary_residual(&self, residual: &Array2<f32>) -> Option<ResidualHandle> {
        let s = residual.shape();
        let (rows, cols) = (s[0], s[1]);
        let flat = residual
            .as_slice()
            .map(|s| s.to_vec())
            .unwrap_or_else(|| residual.iter().copied().collect());
        Some(ResidualHandle::new(CpuResidualHandle {
            flat,
            shape: (rows, cols),
        }))
    }

    fn forward_from_layer(
        &self,
        weights: larql_models::WeightsView,
        start_layer: usize,
        residuals: &ResidualHandle,
        token_ids: &[u32],
    ) -> Option<Array2<f32>> {
        let r = cpu_residual(residuals);
        let raw =
            crate::forward::forward_from_layer(weights, token_ids, &r.flat, start_layer, None);
        // The returned `RawForward` has `h_pre_norm` shape [seq_len, hidden];
        // engines want the last position's hidden as [1, hidden].
        let h = raw.h_pre_norm;
        let last = h.shape()[0] - 1;
        Some(h.slice(ndarray::s![last..=last, ..]).to_owned())
    }

    // `recompute_kv_from_residuals`, `compressed_kv_append`,
    // `attention_step_windowed`, and `residual_norm_store` use the
    // trait defaults (decomposition / unimplemented). Step 3 engine
    // migration adds overrides when the engines that consume them
    // actually need a CPU body.

    // ── Coarse fused intents ────────────────────────────────────────
    //
    // Route through the production cached-decode pipeline. Backend
    // inspects `index` (when present) and `weights` to pick the right
    // kernel — Q4K matvec today, future quant formats slot in without
    // changing the trait surface or the engine call sites.

    fn coarse_prefill(
        &self,
        weights: &ModelWeights,
        token_ids: &[u32],
        index: Option<&dyn crate::KvIndex>,
    ) -> Option<(Array2<f32>, KvHandle)> {
        self.coarse_prefill_with_state(weights, token_ids, index, None)
    }

    fn coarse_prefill_with_state(
        &self,
        weights: &ModelWeights,
        token_ids: &[u32],
        index: Option<&dyn crate::KvIndex>,
        state: Option<&mut crate::PerLayerDecodeState>,
    ) -> Option<(Array2<f32>, KvHandle)> {
        if token_ids.is_empty() {
            return None;
        }
        let index = index?;
        if !crate::kquant_forward::supports_cached_decode(weights) {
            return None;
        }
        let (h_full, cache, _timings) = crate::kquant_forward::predict_kquant_prefill_with_state(
            weights, token_ids, index, state,
        );
        let last = h_full.shape()[0] - 1;
        let h = h_full.slice(ndarray::s![last..=last, ..]).to_owned();
        let handle = KvHandle::new(CpuQ4kCacheHandle { cache });
        Some((h, handle))
    }

    /// Coarse prefill under an engine-requested window.
    ///
    /// Accepts the window only when it **cannot bind during the prompt**
    /// (`token_ids.len() <= window`). That is the production shape — a
    /// prompt inside the window, then decode sliding — and it lets the
    /// engine keep the fused path for it.
    ///
    /// A prompt longer than the window would need per-query-position
    /// masking inside `predict_kquant_prefill_with_state`, which this
    /// entry point does not thread; rather than silently attend the full
    /// prompt while the engine advertises a window, it declines and the
    /// engine falls back to the per-layer path. Fail closed: a wrong
    /// answer at full speed is worse than a right one at 2.4x.
    fn coarse_prefill_windowed(
        &self,
        weights: &ModelWeights,
        token_ids: &[u32],
        index: Option<&dyn crate::KvIndex>,
        window: Option<usize>,
    ) -> Option<(Array2<f32>, KvHandle)> {
        if let Some(w) = window {
            if token_ids.len() > w {
                return None;
            }
        }
        self.coarse_prefill_with_state(weights, token_ids, index, None)
    }

    /// Coarse decode under an engine-requested window.
    ///
    /// The cache rows are absolute stream positions carrying their own
    /// RoPE, and softmax over keys is order-independent, so a sliding
    /// window is exactly "drop the oldest rows". Trimming to `w - 1`
    /// *before* the step leaves room for the row this step appends, so
    /// the cache holds at most `w` rows afterwards — bounding attention
    /// and K/V together, which is the whole contract a windowed engine
    /// advertises.
    fn coarse_decode_step_windowed(
        &self,
        weights: &ModelWeights,
        token_id: u32,
        index: Option<&dyn crate::KvIndex>,
        handle: &mut KvHandle,
        abs_position: usize,
        window: Option<usize>,
    ) -> Option<Array2<f32>> {
        if let Some(w) = window {
            if w == 0 {
                return None;
            }
            // Decline a handle this backend did not mint before mutating
            // anything — `clip_kv` would panic on a foreign one.
            try_cpu_q4k_cache_mut(handle)?;
            // Reuse the existing clip rather than a second trim: `clip_kv`
            // already keeps the tail of both cache shapes, and two copies
            // of "drop the oldest rows" is how they drift apart.
            self.clip_kv(handle, w - 1);
        }
        self.coarse_decode_step(weights, token_id, index, handle, abs_position)
    }

    fn coarse_decode_step(
        &self,
        weights: &ModelWeights,
        token_id: u32,
        index: Option<&dyn crate::KvIndex>,
        handle: &mut KvHandle,
        abs_position: usize,
    ) -> Option<Array2<f32>> {
        let index = index?;
        // Graceful decline (not a panic) when the handle isn't the
        // coarse `CpuQ4kCacheHandle` — e.g. a per-layer handle from a
        // per-layer prefill. `None` tells the engine to use its
        // per-layer decode path instead.
        let inner = try_cpu_q4k_cache_mut(handle)?;
        // Prefer direct-matvec (no per-layer dequant) when supported.
        if crate::kquant_forward::supports_direct_matvec_decode(weights, index) {
            crate::kquant_forward::predict_kquant_decode_step_direct(
                weights,
                token_id,
                index,
                self,
                &mut inner.cache,
                abs_position,
            )
        } else {
            crate::kquant_forward::predict_kquant_decode_step(
                weights,
                token_id,
                index,
                &mut inner.cache,
                abs_position,
            )
            .map(|(h, _)| h)
        }
    }

    /// Read K/V at absolute position `pos` for `layer` out of the
    /// coarse Q4K cache handle. The cache rows are indexed by absolute
    /// stream position (prefill row 0 onward), matching the trait
    /// contract engines rely on for boundary-checkpoint readback
    /// (`WindowedCheckpointEngine::close_window` under HOnly). Returns
    /// `None` for foreign handle shapes (per-layer `CpuKvHandle` has no
    /// cross-layer cache) and for out-of-range `layer`/`pos`.
    fn read_kv_row_at(
        &self,
        handle: &KvHandle,
        layer: usize,
        pos: usize,
    ) -> Option<(Vec<f32>, Vec<f32>)> {
        let inner = handle
            .as_inner()
            .as_any()
            .downcast_ref::<CpuQ4kCacheHandle>()?;
        let (k, v) = inner.cache.get(layer)?.as_ref()?;
        if pos >= k.shape()[0] {
            return None;
        }
        Some((k.row(pos).to_vec(), v.row(pos).to_vec()))
    }

    /// CPU per-layer decode with optional state capture (W1-GPU step 3).
    /// Threads `Option<&mut PerLayerDecodeState>` into the same direct-
    /// matvec walk; when `Some`, each layer's `h_in` / `k_new` / `v_new`
    /// is captured at zero re-compute cost (the values already flow
    /// through the per-layer loop). Falls back to the plain
    /// `coarse_decode_step` for the non-direct-matvec path — that
    /// path doesn't expose per-layer state today (would need a
    /// `predict_kquant_decode_step_with_state` sibling; deferred until
    /// an engine asks for it on the indirect path).
    fn coarse_decode_step_with_state(
        &self,
        weights: &ModelWeights,
        token_id: u32,
        index: Option<&dyn crate::KvIndex>,
        handle: &mut KvHandle,
        abs_position: usize,
        state: Option<&mut crate::PerLayerDecodeState>,
    ) -> Option<Array2<f32>> {
        let index = index?;
        // Same graceful decline as `coarse_decode_step` — see there.
        let inner = try_cpu_q4k_cache_mut(handle)?;
        if crate::kquant_forward::supports_direct_matvec_decode(weights, index) {
            crate::kquant_forward::predict_kquant_decode_step_direct_with_state(
                weights,
                token_id,
                index,
                self,
                &mut inner.cache,
                abs_position,
                state,
            )
        } else {
            // Indirect-matvec path; no state capture wired yet.
            // Drop the state arg and run the standard decode.
            let _ = state;
            crate::kquant_forward::predict_kquant_decode_step(
                weights,
                token_id,
                index,
                &mut inner.cache,
                abs_position,
            )
            .map(|(h, _)| h)
        }
    }
}
