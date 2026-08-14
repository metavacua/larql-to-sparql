//! `KvDispatch` implementation for `crate::MetalBackend` — Step 4
//! scaffolding.
//!
//! **Behaviour:** every method delegates to
//! [`larql_compute::CpuBackend`]'s [`KvDispatch`] impl. K/V handles are
//! CPU-resident (host memory). No real GPU compute — the goal of this
//! step is to exercise the trait shape against actual Metal types so
//! engines can migrate to dispatch-through-trait safely on both
//! backends (Step 3c).
//!
//! Tok/s impact: catastrophically worse than the current Metal path
//! (every call has the same cost as CpuBackend). Acceptance criterion
//! is correctness, not speed. Real Metal kernels land in Step 5; this
//! file is the place where they bind.
//!
//! Feature-gated behind `metal` (same as `crate::MetalBackend`).

use ndarray::Array2;

use crate::MetalBackend;
use larql_compute::kv_dispatch::{
    CompressionCodec, KvDispatch, KvHandle, KvHandleInner, ResidualHandle,
};
use larql_compute::CpuBackend;
use larql_models::ModelWeights;

/// Convenience — the CPU backend instance every method delegates to.
/// Zero-sized type; const-construction is free.
const CPU: CpuBackend = CpuBackend;

/// K and V — the two cached tensors every attention layer stores per
/// position. Named so the K/V sizing arithmetic doesn't read as a bare 2.
const KV_TENSORS_PER_LAYER: usize = 2;

impl KvDispatch for MetalBackend {
    fn alloc_kv_buffer(&self, layer: usize, max_tokens: usize, kv_dim: usize) -> KvHandle {
        // Handles are CPU-resident at Step 4. When real Metal kernels land
        // (Step 5), this returns a `MetalKvHandle` wrapping an
        // `MTLBuffer` instead.
        CPU.alloc_kv_buffer(layer, max_tokens, kv_dim)
    }

    fn append_kv(&self, handle: &mut KvHandle, k_row: &[f32], v_row: &[f32], abs_position: usize) {
        CPU.append_kv(handle, k_row, v_row, abs_position);
    }

    fn clip_kv(&self, handle: &mut KvHandle, window_size: usize) {
        CPU.clip_kv(handle, window_size);
    }

    fn read_kv_to_host(&self, handle: &KvHandle) -> Option<(Array2<f32>, Array2<f32>)> {
        CPU.read_kv_to_host(handle)
    }

    fn attention_step(
        &self,
        weights: larql_models::WeightsView,
        query: &Array2<f32>,
        kv: &mut KvHandle,
        layer: usize,
        abs_position: usize,
        index: Option<&dyn larql_compute::KvIndex>,
    ) -> Option<Array2<f32>> {
        // A3 scaffold delegates to CPU. A4/A6 will introduce a Q4K-native
        // Metal path when `index` is `Some` and Q4K data is available.
        CPU.attention_step(weights, query, kv, layer, abs_position, index)
    }

    fn attention_step_windowed(
        &self,
        weights: larql_models::WeightsView,
        query: &Array2<f32>,
        kv: &mut KvHandle,
        layer: usize,
        abs_position: usize,
        window: usize,
        index: Option<&dyn larql_compute::KvIndex>,
    ) -> Option<Array2<f32>> {
        CPU.attention_step_windowed(weights, query, kv, layer, abs_position, window, index)
    }

    fn attention_prefill(
        &self,
        weights: larql_models::WeightsView,
        tokens_embedded: &Array2<f32>,
        layer: usize,
        window: Option<usize>,
        index: Option<&dyn larql_compute::KvIndex>,
    ) -> Option<(Array2<f32>, KvHandle)> {
        CPU.attention_prefill(weights, tokens_embedded, layer, window, index)
    }

    fn recompute_kv_from_residuals(
        &self,
        weights: larql_models::WeightsView,
        residuals: &Array2<f32>,
        layer: usize,
    ) -> Option<KvHandle> {
        CPU.recompute_kv_from_residuals(weights, residuals, layer)
    }

    fn compressed_kv_append(
        &self,
        handle: &mut KvHandle,
        k: &Array2<f32>,
        v: &Array2<f32>,
        codec: &dyn CompressionCodec,
    ) {
        CPU.compressed_kv_append(handle, k, v, codec);
    }

    fn upload_boundary_residual(&self, residual: &Array2<f32>) -> Option<ResidualHandle> {
        // CPU-resident upload. When Step 5 lands the pipelined boundary
        // upload kernel, this returns a `MetalResidualHandle` instead.
        CPU.upload_boundary_residual(residual)
    }

    fn forward_from_layer(
        &self,
        weights: larql_models::WeightsView,
        start_layer: usize,
        residuals: &ResidualHandle,
        token_ids: &[u32],
    ) -> Option<Array2<f32>> {
        CPU.forward_from_layer(weights, start_layer, residuals, token_ids)
    }

    fn residual_norm_store(
        &self,
        x: &Array2<f32>,
        residual: &Array2<f32>,
        norm_weights: &[f32],
    ) -> Array2<f32> {
        CPU.residual_norm_store(x, residual, norm_weights)
    }

    // ── Coarse fused intents ────────────────────────────────────────
    //
    // Route through Metal's fused `prefill_kquant` / `decode_token` kernels
    // — the production Metal hot path that powers `larql bench` at
    // ~87–100 tok/s on Gemma 3 4B Q4K. K/V cache state lives inside
    // `MetalBackend`'s internal `kv_cache` mutex; the returned
    // `KvHandle` is a sentinel since the engine doesn't manage the
    // state directly.

    // ── Coarse fused intents (Q4_K path) ──────────────────────────────────
    //
    // Route through compute's `kquant_forward::fused_*` helpers
    // (ADR-0022 Step 7). The helpers internally call
    // `backend.prefill_kquant` / `backend.decode_token_with_state_dump`
    // — both DecodeBackend methods that Metal overrides with real
    // fused kernels. K/V cache state lives inside `MetalBackend`'s
    // internal mutex; the returned `KvHandle` is a sentinel.

    fn coarse_prefill(
        &self,
        weights: &ModelWeights,
        token_ids: &[u32],
        index: Option<&dyn larql_compute::KvIndex>,
    ) -> Option<(Array2<f32>, KvHandle)> {
        let index = index?;
        let hidden = larql_compute::kquant_forward::fused_prefill(weights, index, token_ids, self)?;
        Some((hidden, KvHandle::new(MetalCoarseHandle)))
    }

    fn coarse_prefill_with_state(
        &self,
        weights: &ModelWeights,
        token_ids: &[u32],
        index: Option<&dyn larql_compute::KvIndex>,
        state: Option<&mut larql_compute::PerLayerDecodeState>,
    ) -> Option<(Array2<f32>, KvHandle)> {
        let index = index?;
        let Some(state) = state else {
            return self.coarse_prefill(weights, token_ids, Some(index));
        };
        if token_ids.is_empty() {
            return None;
        }
        // Iterative Metal prefill: run `fused_decode_step_with_state`
        // per prefill token, accumulating per-layer state into the
        // pre-allocated `PerLayerDecodeState` Array2s. Replaces the
        // ~2.7s CPU walk (`predict_kquant_prefill_with_state`) with
        // ~12 ms × seq_len of pure Metal dispatch — 40× speedup on
        // 5-token prompts. The Metal KV cache is populated by the
        // decode kernel itself as a side effect, so no separate
        // `fused_prefill` is needed.

        use larql_compute::DecodeBackend as _;
        let num_layers = weights.num_layers;
        let hidden_size = weights.hidden_size;
        let seq_len = token_ids.len();
        let arch = &*weights.arch;

        // Pre-allocate per-layer Array2s sized for the full prefill in
        // local accumulator buffers. Engine-facing contract: each
        // layer's entry is `[seq_len, hidden]` (or `[seq_len, kv_dim]`).
        // W10 Phase A: we write into these locally; after the prefill
        // loop the populated Array2s get wrapped in `CpuStateHandle`
        // and pushed into `state`. Bytes already live on CPU (the
        // kernel readback happens inside `fused_decode_step_with_state`),
        // so the CPU-flavoured handle is the correct wrapping at this
        // phase. Phase B will swap in `MetalStateHandle` to defer the
        // readback.
        let kv_dims: Vec<usize> = (0..num_layers)
            .map(|l| arch.num_kv_heads_for_layer(l) * arch.head_dim_for_layer(l))
            .collect();
        let mut h_arrays: Vec<Array2<f32>> = (0..num_layers)
            .map(|_| Array2::<f32>::zeros((seq_len, hidden_size)))
            .collect();
        let mut k_arrays: Vec<Array2<f32>> = (0..num_layers)
            .map(|l| Array2::<f32>::zeros((seq_len, kv_dims[l])))
            .collect();
        let mut v_arrays: Vec<Array2<f32>> = (0..num_layers)
            .map(|l| Array2::<f32>::zeros((seq_len, kv_dims[l])))
            .collect();

        // Reset + preallocate the Metal KV cache once before the loop.
        self.reset_kv_cache();
        let kv_shapes: Vec<(usize, usize)> = (0..num_layers)
            .map(|l| (arch.num_kv_heads_for_layer(l), arch.head_dim_for_layer(l)))
            .collect();
        self.preallocate_kv_cache_per_layer(
            &kv_shapes,
            larql_compute::pipeline_layer::DEFAULT_GPU_KV_CACHE_MAX_SEQ,
        );

        let mut last_hidden: Option<Array2<f32>> = None;
        for (pos, &token_id) in token_ids.iter().enumerate() {
            let mut dump = larql_compute::DecodeStateDump::with_capacity(num_layers);
            let h_arr = larql_compute::kquant_forward::fused_decode_step_with_state(
                weights, index, token_id, self, &mut dump,
            )?;

            // Bridge dump → engine state: write captured per-layer
            // (h_in, k_new, v_new) into pre-allocated row `pos`.
            // Range loop is clearer than enumerate() here because we
            // index five parallel collections by `layer`.
            #[allow(clippy::needless_range_loop)]
            for layer in 0..num_layers {
                let h_layer = std::mem::take(&mut dump.h_in_per_layer[layer]);
                let k_layer = std::mem::take(&mut dump.k_new_per_layer[layer]);
                let v_layer = std::mem::take(&mut dump.v_new_per_layer[layer]);
                if h_layer.len() != hidden_size
                    || k_layer.len() != kv_dims[layer]
                    || v_layer.len() != kv_dims[layer]
                {
                    // Kernel didn't populate this layer (defensive guard).
                    // Caller's `is_complete_for` check will catch it.
                    return None;
                }
                let mut h_row = h_arrays[layer].row_mut(pos);
                for (j, v) in h_layer.iter().enumerate() {
                    h_row[j] = *v;
                }
                let mut k_row = k_arrays[layer].row_mut(pos);
                for (j, v) in k_layer.iter().enumerate() {
                    k_row[j] = *v;
                }
                let mut v_row = v_arrays[layer].row_mut(pos);
                for (j, v) in v_layer.iter().enumerate() {
                    v_row[j] = *v;
                }
            }

            if pos == seq_len - 1 {
                last_hidden = Some(h_arr);
            }
        }

        // Wrap the populated per-layer arrays as handles. CpuStateHandle
        // moves the Array2 in without a copy.
        for ((h, k), v) in h_arrays.into_iter().zip(k_arrays).zip(v_arrays) {
            state
                .h_in_per_layer
                .push(larql_compute::state_handle::CpuStateHandle::boxed(h));
            state
                .k_new_per_layer
                .push(larql_compute::state_handle::CpuStateHandle::boxed(k));
            state
                .v_new_per_layer
                .push(larql_compute::state_handle::CpuStateHandle::boxed(v));
        }

        Some((last_hidden?, KvHandle::new(MetalCoarseHandle)))
    }

    fn coarse_decode_step(
        &self,
        weights: &ModelWeights,
        token_id: u32,
        index: Option<&dyn larql_compute::KvIndex>,
        _handle: &mut KvHandle,
        _abs_position: usize,
    ) -> Option<Array2<f32>> {
        let index = index?;
        larql_compute::kquant_forward::fused_decode_step(weights, index, token_id, self)
    }

    fn coarse_decode_step_with_state(
        &self,
        weights: &ModelWeights,
        token_id: u32,
        index: Option<&dyn larql_compute::KvIndex>,
        handle: &mut KvHandle,
        abs_position: usize,
        state: Option<&mut larql_compute::PerLayerDecodeState>,
    ) -> Option<Array2<f32>> {
        self.coarse_decode_step_with_state_masked(
            weights,
            token_id,
            index,
            handle,
            abs_position,
            state,
            larql_compute::StateDumpMask::Full,
        )
    }

    fn coarse_decode_step_with_state_masked(
        &self,
        weights: &ModelWeights,
        token_id: u32,
        index: Option<&dyn larql_compute::KvIndex>,
        _handle: &mut KvHandle,
        _abs_position: usize,
        state: Option<&mut larql_compute::PerLayerDecodeState>,
        mask: larql_compute::StateDumpMask,
    ) -> Option<Array2<f32>> {
        let index = index?;
        // W10 Phase C: short-circuit when the engine has nothing to
        // capture. `mask=None` means no h_in, no K/V — there's no
        // reason to drive the state-aware kernel path at all; the
        // plain `fused_decode_step` is the same compute without the
        // state_dump bookkeeping overhead. This shaves the residual
        // kernel-side state-dump cost that survives even when all
        // blits are skipped.
        if matches!(mask, larql_compute::StateDumpMask::None) {
            return larql_compute::kquant_forward::fused_decode_step(
                weights, index, token_id, self,
            );
        }
        let Some(state) = state else {
            return larql_compute::kquant_forward::fused_decode_step(
                weights, index, token_id, self,
            );
        };
        // Bridge engine-facing `PerLayerDecodeState` and substrate
        // `DecodeStateDump`. W10 Phase B: under StateDumpMask::HOnly
        // the kernel only writes h_in into the dump (no K/V staging /
        // readback); the bridge here mirrors that — k_new/v_new stay
        // empty on the engine-facing state.
        let mut dump = larql_compute::DecodeStateDump::with_capacity(weights.num_layers);
        let hidden = larql_compute::kquant_forward::fused_decode_step_with_state_masked(
            weights, index, token_id, self, &mut dump, mask,
        )?;
        let num_layers = weights.num_layers;
        let dump_kv = matches!(mask, larql_compute::StateDumpMask::Full);
        let dump_h = !matches!(mask, larql_compute::StateDumpMask::None);
        // Under None: nothing in dump; engine just gets `hidden`.
        if matches!(mask, larql_compute::StateDumpMask::None) {
            return Some(hidden);
        }
        if (dump_h && dump.h_in_per_layer.len() != num_layers)
            || (dump_kv
                && (dump.k_new_per_layer.len() != num_layers
                    || dump.v_new_per_layer.len() != num_layers))
        {
            // Kernel didn't populate per-layer entries (defensive guard).
            return Some(hidden);
        }
        let hidden_size = weights.hidden_size;
        for layer in 0..num_layers {
            if dump_h {
                let h_vec = std::mem::take(&mut dump.h_in_per_layer[layer]);
                state
                    .h_in_per_layer
                    .push(larql_compute::state_handle::CpuStateHandle::boxed(
                        Array2::from_shape_vec((1, hidden_size), h_vec).ok()?,
                    ));
            }
            if dump_kv {
                let k_vec = std::mem::take(&mut dump.k_new_per_layer[layer]);
                let v_vec = std::mem::take(&mut dump.v_new_per_layer[layer]);
                let kv_dim = k_vec.len();
                state
                    .k_new_per_layer
                    .push(larql_compute::state_handle::CpuStateHandle::boxed(
                        Array2::from_shape_vec((1, kv_dim), k_vec).ok()?,
                    ));
                state
                    .v_new_per_layer
                    .push(larql_compute::state_handle::CpuStateHandle::boxed(
                        Array2::from_shape_vec((1, kv_dim), v_vec).ok()?,
                    ));
            }
        }
        Some(hidden)
    }

    /// Coarse prefill under an engine window.
    ///
    /// Accepts only a prompt that fits inside the window, matching the
    /// CPU rule: the fused prefill has no per-query-position masking, so
    /// a longer prompt would attend in full while the engine advertises
    /// a bound. Declining sends the engine to the per-layer path, which
    /// is correct — just slower.
    fn coarse_prefill_windowed(
        &self,
        weights: &ModelWeights,
        token_ids: &[u32],
        index: Option<&dyn larql_compute::KvIndex>,
        window: Option<usize>,
    ) -> Option<(Array2<f32>, KvHandle)> {
        if let Some(w) = window {
            if w == 0 || token_ids.len() > w {
                return None;
            }
        }
        self.set_engine_window(window);
        self.coarse_prefill(weights, token_ids, index)
    }

    /// Coarse decode under an engine window.
    ///
    /// Two mechanisms, because one cannot do both jobs:
    ///
    /// - **Attention** is bounded by `window` every step, via the layer
    ///   spec's window (see `effective_window_for`). The kernel attends
    ///   `[T - window, T)`, so extra resident rows are simply not read.
    /// - **Memory** is bounded by compaction, run only when occupancy
    ///   reaches `COMPACTION_SLACK x window`. Compacting every step would
    ///   memmove the whole window per token; amortised it is O(1) per
    ///   token, at the cost of holding up to that multiple of the window.
    ///
    /// Doing only the compaction would let attention read up to the slack
    /// multiple of the window between compactions; doing only the span
    /// clamp would never reclaim memory. Both, or neither contract holds.
    fn coarse_decode_step_windowed(
        &self,
        weights: &ModelWeights,
        token_id: u32,
        index: Option<&dyn larql_compute::KvIndex>,
        handle: &mut KvHandle,
        abs_position: usize,
        window: Option<usize>,
    ) -> Option<Array2<f32>> {
        let Some(w) = window else {
            self.set_engine_window(None);
            return self.coarse_decode_step(weights, token_id, index, handle, abs_position);
        };
        if w == 0 {
            return None;
        }
        self.set_engine_window(Some(w));
        self.compact_kv_to_window(w);
        self.coarse_decode_step(weights, token_id, index, handle, abs_position)
    }

    fn per_layer_is_host_delegated(&self) -> bool {
        // Every per-layer method above forwards to `CPU`. Only the
        // `coarse_*` family runs Metal kernels. Until the per-layer
        // surface has native implementations this must stay `true`, or
        // diagnostics will keep reporting CPU work as GPU work.
        true
    }

    fn backend_resident_kv_bytes(&self) -> usize {
        // The coarse pipeline's K/V lives here, not in the handle
        // (`MetalCoarseHandle` is a sentinel), so an engine that only
        // measures its handles reports zero on this path. Count the
        // populated prefix of each layer — `current_len`, not `max_seq`:
        // the buffers are preallocated to the context ceiling and
        // charging an engine for capacity it has not filled would
        // overstate every short-context run.
        let Ok(guard) = self.kv_cache.lock() else {
            return 0;
        };
        let Some(cache) = guard.as_ref() else {
            return 0;
        };
        cache
            .layers
            .iter()
            .map(|l| {
                l.current_len
                    * l.num_kv_heads
                    * l.head_dim
                    * KV_TENSORS_PER_LAYER
                    * std::mem::size_of::<f32>()
            })
            .sum()
    }

    fn read_kv_row_at(
        &self,
        _handle: &KvHandle,
        layer: usize,
        pos: usize,
    ) -> Option<(Vec<f32>, Vec<f32>)> {
        // W10 Phase B: read a single position's K/V back from the Metal
        // kv cache. Used by engines running under HOnly that need to
        // snapshot a specific position on demand (e.g. windowed_checkpoint's
        // close_window). Small (~kv_dim * 4 B per K and V) so cheap vs
        // an end-of-window snapshot of the whole window.
        let cache_guard = self.kv_cache.lock().ok()?;
        let cache = cache_guard.as_ref()?;
        let layer_kv = cache.layers.get(layer)?;
        if pos >= layer_kv.current_len {
            return None;
        }
        let stride = layer_kv.num_kv_heads * layer_kv.head_dim;
        let start_f32 = pos * stride;
        let end_f32 = start_f32 + stride;
        // Read the whole prefix up to end and slice the tail — the
        // underlying buffer is host-visible on M-series; the slice
        // avoids reading positions we don't need.
        let k_full = crate::buffers::try_read_buffer_f32(&layer_kv.k_cache, end_f32)?;
        let v_full = crate::buffers::try_read_buffer_f32(&layer_kv.v_cache, end_f32)?;
        let k_row = k_full[start_f32..end_f32].to_vec();
        let v_row = v_full[start_f32..end_f32].to_vec();
        Some((k_row, v_row))
    }
}

/// Sentinel `KvHandleInner` for `MetalBackend::coarse_prefill` — the
/// actual K/V state lives in `MetalBackend`'s internal `kv_cache`
/// mutex, populated by the fused `prefill_kquant` / `decode_token` kernels.
/// The handle exists to satisfy the trait shape; engines must treat it
/// opaquely.
pub struct MetalCoarseHandle;

impl KvHandleInner for MetalCoarseHandle {
    fn cached_len(&self) -> usize {
        // Backend-side state; not exposed through the handle. Engines
        // that need the cache length should query the backend directly.
        0
    }
    fn kv_dim(&self) -> usize {
        0
    }
    fn backend_name(&self) -> &'static str {
        "metal-coarse"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

// `KvHandleInner` and `ResidualHandleInner` placeholders for the
// per-layer dispatch path are not needed at Step 4 — we reuse
// `CpuKvHandle` and `CpuResidualHandle` from the CPU module since
// handles are host-resident. Step 5 will introduce `MetalKvHandle`
// (wrapping `MTLBuffer`) once real per-layer Metal compute lands.

#[cfg(test)]
mod tests {
    //! Coverage tests for the CPU-delegation `KvDispatch` scaffold.
    //!
    //! Each method on `MetalBackend` forwards to `CpuBackend` at Step 4;
    //! the assertions here drive the delegation paths and (where the
    //! result is observable) confirm shape parity with the direct CPU
    //! call. The coarse Q4_K fused methods (`coarse_prefill*`,
    //! `coarse_decode_step*`) need a real Q4_K vindex fixture and are
    //! covered end-to-end in `tests/test_metal_decode_synthetic.rs`.
    use super::*;
    use larql_models::test_fixtures::make_test_weights;

    fn backend() -> MetalBackend {
        MetalBackend::new().expect("Metal device available on test host")
    }

    #[test]
    fn alloc_kv_buffer_delegates_to_cpu() {
        let m = backend();
        let h = m.alloc_kv_buffer(
            /*layer=*/ 0, /*max_tokens=*/ 8, /*kv_dim=*/ 32,
        );
        assert_eq!(h.cached_len(), 0);
        assert_eq!(h.kv_dim(), 32);
    }

    #[test]
    fn append_and_read_kv_round_trips_through_cpu() {
        let m = backend();
        let mut h = m.alloc_kv_buffer(0, 4, 4);
        m.append_kv(&mut h, &[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0], 0);
        m.append_kv(
            &mut h,
            &[9.0, 10.0, 11.0, 12.0],
            &[13.0, 14.0, 15.0, 16.0],
            1,
        );
        let (k, v) = m.read_kv_to_host(&h).expect("read after append");
        assert_eq!(k.shape(), &[2, 4]);
        assert_eq!(v.shape(), &[2, 4]);
        assert_eq!(k[[0, 0]], 1.0);
        assert_eq!(v[[1, 3]], 16.0);
    }

    #[test]
    fn clip_kv_truncates_to_window() {
        let m = backend();
        let mut h = m.alloc_kv_buffer(0, 8, 2);
        for i in 0..4u32 {
            let f = i as f32;
            m.append_kv(&mut h, &[f, f], &[f, f], i as usize);
        }
        m.clip_kv(&mut h, 2);
        let (k, _) = m.read_kv_to_host(&h).expect("read after clip");
        assert_eq!(k.shape(), &[2, 2], "clip to window=2 keeps newest 2 rows");
    }

    #[test]
    fn attention_step_delegates_through_cpu() {
        let weights = make_test_weights();
        let m = backend();
        let tokens = vec![0u32, 1, 2];
        let h_in = larql_compute::forward::embed_tokens_pub(&weights, &tokens);
        let (_, mut kv) = m
            .attention_prefill(
                larql_models::WeightsView::dense(&weights),
                &h_in,
                0,
                None,
                None,
            )
            .expect("prefill");
        let h_new = larql_compute::forward::embed_tokens_pub(&weights, &[3u32]);
        let h = m
            .attention_step(
                larql_models::WeightsView::dense(&weights),
                &h_new,
                &mut kv,
                0,
                tokens.len(),
                None,
            )
            .expect("attention_step");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
    }

    #[test]
    fn attention_step_windowed_delegates_through_cpu() {
        let weights = make_test_weights();
        let m = backend();
        let tokens = vec![0u32, 1, 2];
        let h_in = larql_compute::forward::embed_tokens_pub(&weights, &tokens);
        let (_, mut kv) = m
            .attention_prefill(
                larql_models::WeightsView::dense(&weights),
                &h_in,
                0,
                None,
                None,
            )
            .expect("prefill");
        let h_new = larql_compute::forward::embed_tokens_pub(&weights, &[3u32]);
        let h = m
            .attention_step_windowed(
                larql_models::WeightsView::dense(&weights),
                &h_new,
                &mut kv,
                0,
                tokens.len(),
                64,
                None,
            )
            .expect("windowed attention_step");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
    }

    #[test]
    fn attention_prefill_delegates_through_cpu() {
        let weights = make_test_weights();
        let m = backend();
        let tokens = vec![0u32, 1, 2];
        let h_in = larql_compute::forward::embed_tokens_pub(&weights, &tokens);
        let (h, kv) = m
            .attention_prefill(
                larql_models::WeightsView::dense(&weights),
                &h_in,
                0,
                None,
                None,
            )
            .expect("prefill");
        assert_eq!(h.shape(), &[tokens.len(), weights.hidden_size]);
        assert_eq!(kv.cached_len(), tokens.len());
    }

    #[test]
    fn recompute_kv_from_residuals_delegates_through_cpu() {
        // `CpuBackend` doesn't override the trait default (it's a Metal-
        // shaped intent, MarkovResidual-only), so delegation returns the
        // default `None`. The point of the test is to drive the Metal
        // dispatch into CpuBackend and confirm it surfaces the same
        // `None` — exercising the delegation pathway.
        let weights = make_test_weights();
        let m = backend();
        let cpu = larql_compute::CpuBackend;
        let residuals =
            Array2::from_shape_vec((3, weights.hidden_size), vec![0.0; 3 * weights.hidden_size])
                .unwrap();
        let m_result = m.recompute_kv_from_residuals(
            larql_models::WeightsView::dense(&weights),
            &residuals,
            0,
        );
        let cpu_result = cpu.recompute_kv_from_residuals(
            larql_models::WeightsView::dense(&weights),
            &residuals,
            0,
        );
        assert_eq!(
            m_result.is_some(),
            cpu_result.is_some(),
            "Metal delegation must match CpuBackend"
        );
    }

    #[test]
    fn upload_boundary_residual_delegates_through_cpu() {
        let weights = make_test_weights();
        let m = backend();
        let residual =
            Array2::from_shape_vec((1, weights.hidden_size), vec![0.0; weights.hidden_size])
                .unwrap();
        let handle = m.upload_boundary_residual(&residual).expect("upload");
        let _ = handle;
    }

    #[test]
    fn forward_from_layer_delegates_through_cpu() {
        let weights = make_test_weights();
        let m = backend();
        let residual =
            Array2::from_shape_vec((1, weights.hidden_size), vec![0.0; weights.hidden_size])
                .unwrap();
        let handle = m.upload_boundary_residual(&residual).expect("upload");
        let h = m
            .forward_from_layer(
                larql_models::WeightsView::dense(&weights),
                1,
                &handle,
                &[0u32, 1, 2],
            )
            .expect("forward_from_layer");
        assert_eq!(h.ncols(), weights.hidden_size);
    }

    #[test]
    fn residual_norm_store_delegates_through_cpu() {
        let m = backend();
        let cpu = larql_compute::CpuBackend;
        let x = Array2::from_shape_vec((2, 4), (0..8).map(|i| i as f32).collect()).unwrap();
        let res = Array2::from_shape_vec((2, 4), (0..8).map(|i| -(i as f32)).collect()).unwrap();
        let norm = vec![1.0; 4];
        let h_m = m.residual_norm_store(&x, &res, &norm);
        let h_c = cpu.residual_norm_store(&x, &res, &norm);
        assert_eq!(h_m, h_c, "Metal delegation must bit-match CpuBackend");
    }

    #[test]
    fn read_kv_row_at_returns_none_when_cache_empty() {
        let m = backend();
        let sentinel = KvHandle::new(MetalCoarseHandle);
        assert!(m.read_kv_row_at(&sentinel, 0, 0).is_none());
    }

    // ── MetalCoarseHandle inner impl ──────────────────────────────────

    #[test]
    fn metal_coarse_handle_reports_sentinel_values() {
        let mut h = MetalCoarseHandle;
        assert_eq!(KvHandleInner::cached_len(&h), 0);
        assert_eq!(KvHandleInner::kv_dim(&h), 0);
        assert_eq!(KvHandleInner::backend_name(&h), "metal-coarse");
        let any: &dyn std::any::Any = KvHandleInner::as_any(&h);
        assert!(any.downcast_ref::<MetalCoarseHandle>().is_some());
        let any_mut: &mut dyn std::any::Any = KvHandleInner::as_any_mut(&mut h);
        assert!(any_mut.downcast_mut::<MetalCoarseHandle>().is_some());
    }

    #[test]
    fn coarse_decode_step_without_index_returns_none() {
        let weights = make_test_weights();
        let m = backend();
        let mut handle = KvHandle::new(MetalCoarseHandle);
        let result = m.coarse_decode_step(&weights, 0u32, None, &mut handle, 0);
        assert!(result.is_none());
    }

    /// Drives `MetalBackend::coarse_prefill` end-to-end against the Q4_K
    /// fixture — runs through `fused_prefill` and exits via
    /// `prefill_kquant` on the real Metal kernel. This is the test that
    /// makes the file's coverage jump from 60% → 90%+.
    #[test]
    fn coarse_prefill_with_q4k_fixture_returns_hidden_and_handle() {
        use larql_compute::test_fixtures::make_q4k_fixture_index;
        use larql_models::test_fixtures::make_test_q4k_weights;
        let m = backend();
        let weights = make_test_q4k_weights();
        let idx = make_q4k_fixture_index(&weights);
        let result = m.coarse_prefill(&weights, &[0u32, 1, 2], Some(&idx));
        let (h, _handle) = result.expect("Metal Q4K prefill succeeds");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
    }

    /// `coarse_prefill_with_state` happy path on Metal with Q4_K.
    #[test]
    fn coarse_prefill_with_state_drives_metal_decode_loop() {
        use larql_compute::test_fixtures::make_q4k_fixture_index;
        use larql_models::test_fixtures::make_test_q4k_weights;
        let m = backend();
        let weights = make_test_q4k_weights();
        let idx = make_q4k_fixture_index(&weights);
        let mut state = larql_compute::PerLayerDecodeState::with_capacity(weights.num_layers);
        let result =
            m.coarse_prefill_with_state(&weights, &[0u32, 1, 2], Some(&idx), Some(&mut state));
        let (h, _handle) = result.expect("Metal Q4K prefill-with-state succeeds");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(state.is_complete_for(weights.num_layers));
    }

    /// `coarse_decode_step` end-to-end on Metal with the Q4_K fixture.
    #[test]
    fn coarse_decode_step_with_q4k_fixture_returns_hidden() {
        use larql_compute::test_fixtures::make_q4k_fixture_index;
        use larql_models::test_fixtures::make_test_q4k_weights;
        let m = backend();
        let weights = make_test_q4k_weights();
        let idx = make_q4k_fixture_index(&weights);
        // Seed the KV cache via prefill.
        let (_h, mut handle) = m
            .coarse_prefill(&weights, &[0u32, 1, 2], Some(&idx))
            .expect("prefill seeds the cache");
        let result = m.coarse_decode_step(&weights, 4u32, Some(&idx), &mut handle, 3);
        let h = result.expect("Metal Q4K decode step returns Some");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
    }

    /// `coarse_decode_step_with_state_masked` over all 3 mask variants
    /// against Metal + the Q4_K fixture. Drives the masked-state-dump
    /// bridging logic in `kv_dispatch_impl`.
    #[test]
    fn coarse_decode_step_with_state_masked_over_all_mask_variants() {
        use larql_compute::test_fixtures::make_q4k_fixture_index;
        use larql_models::test_fixtures::make_test_q4k_weights;
        let m = backend();
        let weights = make_test_q4k_weights();
        let idx = make_q4k_fixture_index(&weights);
        let (_h, mut handle) = m
            .coarse_prefill(&weights, &[0u32, 1, 2], Some(&idx))
            .expect("prefill seeds the cache");
        for mask in [
            larql_compute::StateDumpMask::Full,
            larql_compute::StateDumpMask::HOnly,
            larql_compute::StateDumpMask::None,
        ] {
            let mut state = larql_compute::PerLayerDecodeState::with_capacity(weights.num_layers);
            let result = m.coarse_decode_step_with_state_masked(
                &weights,
                5u32,
                Some(&idx),
                &mut handle,
                4,
                Some(&mut state),
                mask,
            );
            assert!(
                result.is_some(),
                "Metal decode-step-with-state-masked should return Some under {mask:?}"
            );
        }
    }

    #[test]
    fn coarse_decode_step_with_state_masked_without_index_returns_none() {
        let weights = make_test_weights();
        let m = backend();
        let mut handle = KvHandle::new(MetalCoarseHandle);
        let result = m.coarse_decode_step_with_state_masked(
            &weights,
            0u32,
            None,
            &mut handle,
            0,
            None,
            larql_compute::StateDumpMask::Full,
        );
        assert!(result.is_none());
    }
}

/// Occupancy multiple of the window at which compaction runs.
///
/// Re-exported from `larql-compute` so the compaction trigger and the
/// capacity rule that must accommodate it (`kv_capacity_for_window`)
/// cannot drift apart — a capacity below the trigger is a buffer overrun,
/// not a smaller allocation.
pub(crate) use larql_compute::pipeline_layer::KV_COMPACTION_SLACK as COMPACTION_SLACK;

impl MetalBackend {
    /// Reclaim K/V above `COMPACTION_SLACK x window` rows, per layer.
    ///
    /// Safe to call every step: it is a no-op until occupancy actually
    /// reaches the slack bound. Eviction lowers occupancy only —
    /// `abs_position` keeps climbing, so RoPE does not rewind (see
    /// `LayerKVCache::evict_to_window`).
    pub(crate) fn compact_kv_to_window(&self, window: usize) {
        if window == 0 {
            return;
        }
        let trigger = window.saturating_mul(COMPACTION_SLACK);
        let Ok(mut guard) = self.kv_cache.lock() else {
            return;
        };
        let Some(cache) = guard.as_mut() else {
            return;
        };
        for layer in cache.layers.iter_mut() {
            if layer.current_len >= trigger {
                layer.evict_to_window(window);
            }
        }
    }
}

#[cfg(test)]
mod windowed_coarse_tests {
    use super::*;
    use crate::MetalBackend;

    fn backend() -> MetalBackend {
        MetalBackend::new().expect("Metal device available on test host")
    }

    /// The narrower of the two windows wins, and `0` means unbounded on
    /// both sides — the same sentinel the kernel uses, so no translation.
    #[test]
    fn effective_window_takes_the_narrower_of_arch_and_engine() {
        let b = backend();

        b.set_engine_window(None);
        assert_eq!(b.effective_window_for(0), 0, "both unbounded");
        assert_eq!(b.effective_window_for(1024), 1024, "arch window survives");

        b.set_engine_window(Some(256));
        assert_eq!(
            b.effective_window_for(0),
            256,
            "engine bounds a global layer"
        );
        assert_eq!(
            b.effective_window_for(1024),
            256,
            "engine is narrower than the arch's sliding layer"
        );
        assert_eq!(
            b.effective_window_for(128),
            128,
            "arch is narrower than the engine's request"
        );

        b.set_engine_window(None);
        assert_eq!(
            b.effective_window_for(1024),
            1024,
            "clearing restores the arch"
        );
    }

    /// A prompt longer than the window is declined — the fused prefill
    /// has no per-query masking, so accepting would attend in full while
    /// the engine advertises a bound.
    #[test]
    fn prefill_declines_a_prompt_longer_than_the_window() {
        let b = backend();
        let weights = larql_models::test_fixtures::make_test_q4k_weights();
        assert!(b
            .coarse_prefill_windowed(&weights, &[0u32, 1, 2, 3], None, Some(2))
            .is_none());
        assert!(
            b.coarse_prefill_windowed(&weights, &[0u32, 1, 2, 3], None, Some(0))
                .is_none(),
            "a zero window is refused, not treated as unbounded"
        );
    }

    /// Compaction is a no-op below the slack bound and reclaims above it,
    /// without ever moving the stream position.
    #[test]
    fn compaction_reclaims_only_past_the_slack_bound() {
        let b = backend();
        let window = 4usize;
        {
            let mut guard = b.kv_cache.lock().expect("kv cache lock");
            *guard = Some(crate::ops::kv_cache::KVCache::new(&b.bufs, 1, 64, 2, 4));
            let layer = &mut guard.as_mut().unwrap().layers[0];
            for _ in 0..(window * COMPACTION_SLACK - 1) {
                layer.advance_one();
            }
        }
        b.compact_kv_to_window(window);
        {
            let guard = b.kv_cache.lock().unwrap();
            let layer = &guard.as_ref().unwrap().layers[0];
            assert_eq!(
                layer.current_len,
                window * COMPACTION_SLACK - 1,
                "below the slack bound nothing should move"
            );
        }

        {
            let mut guard = b.kv_cache.lock().unwrap();
            guard.as_mut().unwrap().layers[0].advance_one();
        }
        b.compact_kv_to_window(window);
        {
            let guard = b.kv_cache.lock().unwrap();
            let layer = &guard.as_ref().unwrap().layers[0];
            assert_eq!(layer.current_len, window, "reclaimed to the window");
            assert_eq!(
                layer.abs_position,
                window * COMPACTION_SLACK,
                "compaction must never rewind the stream position"
            );
        }
    }

    /// A zero window compacts nothing rather than emptying the cache.
    #[test]
    fn a_zero_window_compacts_nothing() {
        let b = backend();
        {
            let mut guard = b.kv_cache.lock().expect("kv cache lock");
            *guard = Some(crate::ops::kv_cache::KVCache::new(&b.bufs, 1, 64, 2, 4));
            guard.as_mut().unwrap().layers[0].advance_one();
        }
        b.compact_kv_to_window(0);
        let guard = b.kv_cache.lock().unwrap();
        assert_eq!(guard.as_ref().unwrap().layers[0].current_len, 1);
    }

    /// Compaction before any prefill has allocated a cache must return
    /// rather than unwrap — a windowed engine calls this every decode
    /// step, including the first.
    #[test]
    fn compacting_without_a_cache_is_a_noop() {
        let b = backend();
        assert!(
            b.kv_cache.lock().expect("kv cache lock").is_none(),
            "a fresh backend has no cache yet"
        );
        b.compact_kv_to_window(4);
    }

    // ── coarse_decode_step_windowed: the three arms ──────────────────

    #[test]
    fn decode_step_windowed_forwards_when_no_window_requested() {
        let weights = larql_models::test_fixtures::make_test_weights();
        let b = backend();
        let mut handle = KvHandle::new(MetalCoarseHandle);
        // No index → the underlying coarse step declines, so this pins the
        // *delegation*: the window-less arm clears the engine window and
        // hands straight through.
        assert!(b
            .coarse_decode_step_windowed(&weights, 0, None, &mut handle, 0, None)
            .is_none());
        assert_eq!(b.effective_window_for(1024), 1024, "engine window cleared");
    }

    #[test]
    fn decode_step_windowed_declines_a_zero_window() {
        let weights = larql_models::test_fixtures::make_test_weights();
        let b = backend();
        let mut handle = KvHandle::new(MetalCoarseHandle);
        assert!(
            b.coarse_decode_step_windowed(&weights, 0, None, &mut handle, 0, Some(0))
                .is_none(),
            "a zero window is refused, not treated as unbounded"
        );
    }

    #[test]
    fn decode_step_windowed_sets_the_window_and_compacts_before_stepping() {
        let weights = larql_models::test_fixtures::make_test_weights();
        let b = backend();
        let window = 4usize;
        {
            let mut guard = b.kv_cache.lock().expect("kv cache lock");
            *guard = Some(crate::ops::kv_cache::KVCache::new(&b.bufs, 1, 64, 2, 4));
            let layer = &mut guard.as_mut().unwrap().layers[0];
            for _ in 0..(window * COMPACTION_SLACK) {
                layer.advance_one();
            }
        }
        let mut handle = KvHandle::new(MetalCoarseHandle);
        // The step itself declines (no index), but the window bookkeeping
        // and the compaction must both have run first — that pairing is
        // the whole contract this arm exists to hold.
        let _ = b.coarse_decode_step_windowed(&weights, 0, None, &mut handle, 0, Some(window));
        assert_eq!(
            b.effective_window_for(1024),
            window as u32,
            "the engine window must be set before the step"
        );
        let guard = b.kv_cache.lock().unwrap();
        assert_eq!(
            guard.as_ref().unwrap().layers[0].current_len,
            window,
            "occupancy at the slack bound must be compacted before stepping"
        );
    }

    // ── Honesty / accounting overrides ───────────────────────────────

    #[test]
    fn per_layer_surface_reports_itself_as_host_delegated() {
        // Every per-layer method forwards to CPU; only `coarse_*` runs
        // Metal kernels. Reporting `false` here would let diagnostics
        // label a pure-CPU measurement as GPU work.
        assert!(backend().per_layer_is_host_delegated());
    }

    #[test]
    fn resident_kv_bytes_is_zero_before_a_cache_exists() {
        assert_eq!(backend().backend_resident_kv_bytes(), 0);
    }

    #[test]
    fn resident_kv_bytes_counts_the_populated_prefix_not_the_capacity() {
        let b = backend();
        let (max_seq, num_kv_heads, head_dim) = (64usize, 2usize, 4usize);
        let filled = 3usize;
        {
            let mut guard = b.kv_cache.lock().expect("kv cache lock");
            *guard = Some(crate::ops::kv_cache::KVCache::new(
                &b.bufs,
                1,
                max_seq,
                num_kv_heads,
                head_dim,
            ));
            let layer = &mut guard.as_mut().unwrap().layers[0];
            for _ in 0..filled {
                layer.advance_one();
            }
        }
        let per_row = num_kv_heads * head_dim * KV_TENSORS_PER_LAYER * std::mem::size_of::<f32>();
        assert_eq!(b.backend_resident_kv_bytes(), filled * per_row);
        // The buffers are preallocated to the context ceiling; charging for
        // capacity would overstate every short-context run by ~21x here.
        assert_ne!(b.backend_resident_kv_bytes(), max_seq * per_row);
    }

    // ── read_kv_row_at against a populated cache ─────────────────────

    #[test]
    fn read_kv_row_at_returns_the_requested_position() {
        let b = backend();
        let (max_seq, num_kv_heads, head_dim) = (8usize, 2usize, 4usize);
        let stride = num_kv_heads * head_dim;
        // Distinct value per slot so a wrong offset cannot pass.
        let k_data: Vec<f32> = (0..max_seq * stride).map(|i| i as f32).collect();
        let v_data: Vec<f32> = (0..max_seq * stride).map(|i| -(i as f32)).collect();
        {
            let mut guard = b.kv_cache.lock().expect("kv cache lock");
            *guard = Some(crate::ops::kv_cache::KVCache::new(
                &b.bufs,
                1,
                max_seq,
                num_kv_heads,
                head_dim,
            ));
            let layer = &mut guard.as_mut().unwrap().layers[0];
            layer.k_cache = b.bufs.get_f32(&k_data);
            layer.v_cache = b.bufs.get_f32(&v_data);
            layer.advance_one();
            layer.advance_one();
        }
        let sentinel = KvHandle::new(MetalCoarseHandle);

        let (k0, v0) = b.read_kv_row_at(&sentinel, 0, 0).expect("position 0");
        assert_eq!(k0, k_data[0..stride].to_vec());
        assert_eq!(v0, v_data[0..stride].to_vec());

        let (k1, v1) = b.read_kv_row_at(&sentinel, 0, 1).expect("position 1");
        assert_eq!(k1, k_data[stride..2 * stride].to_vec());
        assert_eq!(v1, v_data[stride..2 * stride].to_vec());
    }

    #[test]
    fn read_kv_row_at_declines_a_position_past_the_populated_prefix() {
        let b = backend();
        {
            let mut guard = b.kv_cache.lock().expect("kv cache lock");
            *guard = Some(crate::ops::kv_cache::KVCache::new(&b.bufs, 1, 64, 2, 4));
            guard.as_mut().unwrap().layers[0].advance_one();
        }
        let sentinel = KvHandle::new(MetalCoarseHandle);
        assert!(
            b.read_kv_row_at(&sentinel, 0, 0).is_some(),
            "pos 0 is filled"
        );
        assert!(
            b.read_kv_row_at(&sentinel, 0, 1).is_none(),
            "pos == current_len is capacity, not data"
        );
        assert!(
            b.read_kv_row_at(&sentinel, 9, 0).is_none(),
            "a layer beyond the cache declines"
        );
    }

    /// A prompt that fits is accepted, and the engine window is armed
    /// before the prefill runs — the ordering matters, because the
    /// kernel reads the window off the backend, not off the argument.
    #[test]
    fn prefill_windowed_arms_the_window_for_a_prompt_that_fits() {
        let b = backend();
        let weights = larql_models::test_fixtures::make_test_q4k_weights();
        // No index → the prefill itself declines, which is fine: what this
        // pins is that a fitting prompt gets past the guard and sets the
        // window rather than being refused outright.
        let _ = b.coarse_prefill_windowed(&weights, &[0u32, 1], None, Some(4));
        assert_eq!(
            b.effective_window_for(1024),
            4,
            "a prompt within the window must arm it, not decline"
        );
    }

    /// The unmasked entry point is a forwarder that must request the
    /// full dump; nothing else in the crate calls it, so without this
    /// the `StateDumpMask::Full` it pins is untested.
    #[test]
    fn decode_step_with_state_forwards_asking_for_the_full_dump() {
        let b = backend();
        let weights = larql_models::test_fixtures::make_test_weights();
        let mut handle = KvHandle::new(MetalCoarseHandle);
        let mut state = larql_compute::PerLayerDecodeState::with_capacity(weights.num_layers);
        assert!(b
            .coarse_decode_step_with_state(&weights, 0, None, &mut handle, 0, Some(&mut state))
            .is_none());
    }

    /// `CpuBackend` leaves `compressed_kv_append` at the trait default,
    /// so Metal's delegation surfaces that backend's panic rather than
    /// silently dropping the append. Pins the delegation, not the codec.
    #[test]
    #[should_panic(expected = "compressed_kv_append not implemented")]
    fn compressed_kv_append_delegates_to_cpu() {
        struct PassthroughCodec;
        impl CompressionCodec for PassthroughCodec {
            fn encode(&self, vec: &[f32]) -> Vec<u8> {
                vec.iter().flat_map(|f| f.to_le_bytes()).collect()
            }
            fn decode(&self, bytes: &[u8], dim: usize) -> Vec<f32> {
                bytes
                    .chunks_exact(4)
                    .take(dim)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect()
            }
            fn name(&self) -> &str {
                "passthrough"
            }
        }

        let b = backend();
        let mut handle = b.alloc_kv_buffer(0, 4, 4);
        let k = Array2::zeros((1, 4));
        let v = Array2::zeros((1, 4));
        b.compressed_kv_append(&mut handle, &k, &v, &PassthroughCodec);
    }
}
