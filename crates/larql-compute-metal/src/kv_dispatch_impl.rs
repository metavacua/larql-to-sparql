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
mod tests;

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
mod windowed_coarse_tests;
