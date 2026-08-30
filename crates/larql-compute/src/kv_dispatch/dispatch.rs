//! The [`KvDispatch`] trait — engine-facing intent surface. Split from
//! `kv_dispatch/mod.rs` — see the module-level doc there.

use larql_models::ModelWeights;
use ndarray::Array2;

use super::{CompressionCodec, KvHandle, ResidualHandle};
use crate::PerLayerDecodeState;

/// Per-layer state captured during a decode step — populated by
/// [`KvDispatch::coarse_decode_step_with_state`] when the engine
/// needs per-layer intermediates that its state policy depends on.
///
/// All three vectors have length `num_layers` after a successful
/// decode. Each per-layer entry is a single-row matrix sized for
/// that layer's hidden / kv_dim respectively. Engines map these to
/// their internal state:
///
/// - `markov_residual`: `h_in_per_layer[l]` becomes the new row in
///   `stored[l]`; `k_new_per_layer[l]` / `v_new_per_layer[l]`
///   become the new row in `hot_kv[l]`.
/// - `markov_residual_codec`: same as `markov_residual`; on
///   window-overflow the evicted rows get codec-encoded into
///   `cold_encoded[l]`.
/// - `windowed_checkpoint`: `k_new_per_layer[l]` / `v_new_per_layer[l]`
///   are appended to the per-layer K/V cache; `h_in_per_layer` is
///   unused but populated for API uniformity (cheap blit).
/// - `turbo_quant`: `k_new_per_layer[l]` / `v_new_per_layer[l]`
///   feed the WHT+Lloyd-Max encoder which produces the updated
///   compressed K/V slot.
///
/// On Metal the buffers are populated via blit-encode steps inside
/// the same command buffer that runs the fused decode kernel — no
/// extra round-trip. On CPU the engine's per-layer Rust loop fills
/// them directly. Engines that don't need per-layer state pass
/// `None` and stay on the original `coarse_decode_step`
/// (one-buffer-back), so this is opt-in.
/// Engine-facing intent surface.
///
/// All methods are synchronous (return immediately with the result;
/// any GPU work is submitted and waited on internally). Async / stream-
/// graph variants live on a future `AsyncComputeBackend` trait — not
/// part of v1. See `compute-backend-redesign.md` §11.4.
///
/// Engines hold `&dyn KvDispatch` alongside
/// `&dyn crate::ComputeBackend` and [`crate::FfnBackend`].
/// The three abstractions compose orthogonally: substrate kernels +
/// engine intents + FFN routing.
pub trait KvDispatch {
    // ── Cache primitives ────────────────────────────────────────────

    /// Allocate a K/V buffer for `layer`, sized for at most `max_tokens`
    /// positions of `kv_dim`-wide K and V rows. Layout is backend-
    /// specific; engines treat the returned handle opaquely.
    fn alloc_kv_buffer(&self, layer: usize, max_tokens: usize, kv_dim: usize) -> KvHandle {
        let _ = (layer, max_tokens, kv_dim);
        unimplemented!("alloc_kv_buffer not implemented for this backend")
    }

    /// Append a single K/V row at `abs_position`. The handle must have
    /// been allocated by *this* backend; cross-backend handles panic.
    fn append_kv(&self, handle: &mut KvHandle, k_row: &[f32], v_row: &[f32], abs_position: usize) {
        let _ = (handle, k_row, v_row, abs_position);
        unimplemented!("append_kv not implemented for this backend")
    }

    /// Clip the handle's cached entries to at most `window_size` rows
    /// (keep the tail). Backends with bounded-ring-buffer K/V layouts
    /// may implement this as a no-op; backends with growing K/V apply
    /// a shift or drop.
    fn clip_kv(&self, handle: &mut KvHandle, window_size: usize) {
        let _ = (handle, window_size);
        unimplemented!("clip_kv not implemented for this backend")
    }

    /// Drop cached rows past `len`, keeping the **first** `len` in order.
    /// Returns whether the handle now holds exactly `len` rows.
    ///
    /// The inverse of an append: this rewinds a partially-applied decode
    /// step so a caller that could not finish the token leaves the cache
    /// describing the token sequence it described before. Distinct from
    /// [`Self::clip_kv`], which keeps the *tail* to enforce a sliding
    /// window — that one moves the cache forward, this one moves it back.
    ///
    /// The default answers `false` rather than panicking, because "this
    /// backend cannot rewind" is a legitimate state that a caller must
    /// handle (by invalidating the cache) rather than a programming
    /// error. Returning `true` without having rewound is the one
    /// unacceptable answer: it tells the caller a corrupt cache is sound.
    fn truncate_kv(&self, handle: &mut KvHandle, len: usize) -> bool {
        let _ = (handle, len);
        false
    }

    /// Read the full K/V back to host memory as a `(K, V)` pair.
    /// Blocking copy on GPU backends; identity on CPU. Should NOT be
    /// used in hot loops — it's the cross-backend escape hatch for
    /// fallback paths and debug inspection.
    fn read_kv_to_host(&self, handle: &KvHandle) -> Option<(Array2<f32>, Array2<f32>)> {
        let _ = handle;
        None
    }

    /// Bytes of K/V this backend holds in its own storage, i.e. K/V that
    /// is NOT reachable by measuring the [`KvHandle`]s an engine owns.
    ///
    /// Backends whose fused pipelines keep the cache internally hand out
    /// a sentinel handle that measures zero (Metal's coarse whole-model
    /// handle is the live example). An engine summing its handles then
    /// reports no K/V at all, which reads as "this engine is free" in a
    /// memory comparison when the truth is "its K/V lives one layer
    /// down". Engines add this to their own accounting so the two
    /// dispatch shapes are comparable.
    ///
    /// Default 0 = every byte this backend holds is reachable through a
    /// handle, so adding it would double-count.
    fn backend_resident_kv_bytes(&self) -> usize {
        0
    }

    /// Whether this backend implements the **per-layer** surface
    /// (`attention_step`, `attention_prefill`, `append_kv`, …) by
    /// forwarding to the host CPU rather than running native kernels.
    ///
    /// `MetalBackend` answers `true`: only its `coarse_*` family is
    /// GPU-resident, so an engine that declines the coarse path runs its
    /// whole forward on the CPU while callers still believe they
    /// selected a GPU backend. Diagnostics need to be able to say that
    /// out loud; without it a windowed engine reports `[metal (GPU)]`
    /// over a pure-CPU measurement.
    ///
    /// Default `false` — a backend is assumed to mean what it says.
    fn per_layer_is_host_delegated(&self) -> bool {
        false
    }

    // ── Attention primitives ────────────────────────────────────────

    /// Run one decode-step attention: Q (one row, pre-projection
    /// hidden) is projected internally to Q/K/V via the layer's
    /// weights, attended against K/V from `kv` PLUS the new token's
    /// K/V (the backend computes the new K/V from the query and
    /// appends it to `kv` as a side effect), and the post-O-projection
    /// hidden state is returned.
    ///
    /// `kv` is `&mut` because the backend mutates it: K and V grow by
    /// one row to include the current token. After this call the
    /// caller may invoke [`Self::clip_kv`] to enforce a sliding window.
    ///
    /// Capability gate:
    /// [`crate::Capability::FusedAttentionStep`]. Backends
    /// that don't support fused attention return `None`; callers fall
    /// back to decomposed BLAS attention via [`crate::MatMul`]
    /// + manual K/V management.
    ///
    /// `index` is `Some` when the caller has a Q4K (or other
    /// quantised) `VectorIndex` available alongside the f32 fallback
    /// in `weights.tensors`. Backends with native Q4K kernels (e.g.
    /// `MetalBackend` once A4 lands) use it directly; CPU backends
    /// today expect the caller to have already populated
    /// `weights.tensors` via
    /// [`crate::kquant_forward::ensure_attn_tensors_dequantised`] when the
    /// quantised source is present.
    ///
    /// See `docs/specs/kv-dispatch-quantization.md`.
    fn attention_step(
        &self,
        weights: larql_models::WeightsView,
        query: &Array2<f32>,
        kv: &mut KvHandle,
        layer: usize,
        abs_position: usize,
        index: Option<&dyn crate::KvIndex>,
    ) -> Option<Array2<f32>> {
        let _ = (weights, query, kv, layer, abs_position, index);
        None
    }

    /// Like [`Self::attention_step`] but with a window bound baked
    /// into the dispatch — backend may use a specialised shader variant
    /// that knows the window size at compile time. Backend may also
    /// elide the post-attention `clip_kv` since the window is known.
    ///
    /// Capability gate:
    /// [`crate::Capability::WindowedAttentionStep`]. Default
    /// runs [`Self::attention_step`] then [`Self::clip_kv`] (correct
    /// but not specialised). `index` is forwarded to the underlying
    /// `attention_step` call.
    #[allow(clippy::too_many_arguments)]
    fn attention_step_windowed(
        &self,
        weights: larql_models::WeightsView,
        query: &Array2<f32>,
        kv: &mut KvHandle,
        layer: usize,
        abs_position: usize,
        window: usize,
        index: Option<&dyn crate::KvIndex>,
    ) -> Option<Array2<f32>> {
        let h = self.attention_step(weights, query, kv, layer, abs_position, index)?;
        self.clip_kv(kv, window);
        Some(h)
    }

    /// Multi-token prefill attention: tokens have been embedded into
    /// `tokens_embedded` (shape `[seq_len, hidden]`). Backend runs full
    /// attention over the sequence, populates a fresh K/V handle, and
    /// returns `(last_hidden_1xH, populated_handle)`.
    ///
    /// `window` selects the K/V cap: `None` = unbounded growth,
    /// `Some(W)` = sliding-window K/V (older positions evicted from
    /// the cache after the prefill).
    ///
    /// `index` follows the same convention as [`Self::attention_step`].
    fn attention_prefill(
        &self,
        weights: larql_models::WeightsView,
        tokens_embedded: &Array2<f32>,
        layer: usize,
        window: Option<usize>,
        index: Option<&dyn crate::KvIndex>,
    ) -> Option<(Array2<f32>, KvHandle)> {
        let _ = (weights, tokens_embedded, layer, window, index);
        None
    }

    // ── Engine-specific primitives ──────────────────────────────────

    /// Regenerate K/V for a layer from stored pre-layer residuals.
    /// Used by `markov-rs`: residuals are the persistent state, K/V is
    /// recomputed each decode step. Backends without this intent fall
    /// back to running the Q/K/V projection through
    /// [`crate::MatMul`] directly.
    fn recompute_kv_from_residuals(
        &self,
        weights: larql_models::WeightsView,
        residuals: &Array2<f32>,
        layer: usize,
    ) -> Option<KvHandle> {
        let _ = (weights, residuals, layer);
        None
    }

    /// Append compressed K/V to a handle using the given codec.
    /// Used by `turbo-quant`. Backends with native codec kernels
    /// (Metal WHT shader) implement this; others fall back to
    /// dequant → f32 append → requant via the caller.
    fn compressed_kv_append(
        &self,
        handle: &mut KvHandle,
        k: &Array2<f32>,
        v: &Array2<f32>,
        codec: &dyn CompressionCodec,
    ) {
        let _ = (handle, k, v, codec);
        unimplemented!("compressed_kv_append not implemented for this backend")
    }

    /// Upload a boundary residual to backend-managed memory. Returns
    /// a handle the engine can use as the starting state for
    /// [`Self::forward_from_layer`]. Used by `apollo` compressed path.
    fn upload_boundary_residual(&self, residual: &Array2<f32>) -> Option<ResidualHandle> {
        let _ = residual;
        None
    }

    /// Run the forward pass starting at `start_layer` using `residuals`
    /// as the layer-`start_layer` input. Used by `apollo` to skip the
    /// pre-crystal layers when boundaries are available.
    fn forward_from_layer(
        &self,
        weights: larql_models::WeightsView,
        start_layer: usize,
        residuals: &ResidualHandle,
        token_ids: &[u32],
    ) -> Option<Array2<f32>> {
        let _ = (weights, start_layer, residuals, token_ids);
        None
    }

    // ── Coarse fused intents ────────────────────────────────────────
    //
    // Coarse-grained, **quantization-agnostic** intents for engines
    // that want backend-fastest decode without per-layer control.
    // The backend inspects `index` (or `weights.tensors`) and dispatches
    // internally to whatever native kernel matches the weight format:
    // Q4K matvec, Q6K matvec, f32 fused, future quant formats — all
    // without changing this trait surface.
    //
    // Engines that DO need per-layer control (MarkovResidual,
    // WindowedCheckpoint, TurboQuant — recompute, checkpoint, codec
    // mechanisms) continue to use the per-layer `attention_prefill` /
    // `attention_step` intents.
    //
    // Default returns `None` — engines that want a coarse path fall
    // back to per-layer dispatch when the backend doesn't support it.

    /// Coarse prefill: run the prompt through every layer using the
    /// backend's fastest available kernel, populate a backend-specific
    /// K/V cache, return last-row hidden + the populated handle.
    ///
    /// The returned `KvHandle` is opaque to the engine; pass it back to
    /// [`Self::coarse_decode_step`] for subsequent steps. Backends are
    /// free to use any internal cache shape (`CpuKvCache` on CPU,
    /// `MTLBuffer` on Metal once Step A6 lands, etc.).
    ///
    /// `weights` is `&mut` because backends with cached-streaming Q4K
    /// kernels may lazily insert dequantised f32 fallback tensors into
    /// `weights.tensors` over the lifetime of the cache. The per-layer
    /// `attention_prefill` keeps `&weights` because it can't grow
    /// shared state.
    fn coarse_prefill(
        &self,
        weights: &ModelWeights,
        token_ids: &[u32],
        index: Option<&dyn crate::KvIndex>,
    ) -> Option<(Array2<f32>, KvHandle)> {
        let _ = (weights, token_ids, index);
        None
    }

    /// One coarse decode step. `handle` must be the `KvHandle` returned
    /// by a prior [`Self::coarse_prefill`] on the same backend.
    fn coarse_decode_step(
        &self,
        weights: &ModelWeights,
        token_id: u32,
        index: Option<&dyn crate::KvIndex>,
        handle: &mut KvHandle,
        abs_position: usize,
    ) -> Option<Array2<f32>> {
        let _ = (weights, token_id, index, handle, abs_position);
        None
    }

    /// Coarse prefill under an **engine-requested sliding window** — a
    /// window the caller imposes across every layer, distinct from the
    /// architecture's own per-layer SWA (which backends read from the
    /// arch and this composes with, narrowest wins).
    ///
    /// A windowed engine promises two things: it attends at most `window`
    /// positions, AND it holds at most that much K/V. The window-less
    /// `coarse_prefill` can honour neither, so a windowed engine had to
    /// decline the fused path entirely and take the generic per-layer
    /// route — which on a host-delegating backend runs the whole forward
    /// on the CPU, costing ~2.4x. This is the entry point that lets a
    /// backend keep the fused path *and* the window.
    ///
    /// **Fail closed.** The default supports only `window: None`, and
    /// answers `None` when a real window is requested — "this backend
    /// cannot bound what you asked me to bound", which the engine reads
    /// as "take the per-layer path", exactly today's behaviour. A
    /// backend that returns `Some` for a windowed request is asserting
    /// it enforced BOTH halves of the contract.
    fn coarse_prefill_windowed(
        &self,
        weights: &ModelWeights,
        token_ids: &[u32],
        index: Option<&dyn crate::KvIndex>,
        window: Option<usize>,
    ) -> Option<(Array2<f32>, KvHandle)> {
        match window {
            None => self.coarse_prefill(weights, token_ids, index),
            Some(_) => None,
        }
    }

    /// One coarse decode step under an engine-requested sliding window.
    /// Same contract and same fail-closed default as
    /// [`Self::coarse_prefill_windowed`].
    fn coarse_decode_step_windowed(
        &self,
        weights: &ModelWeights,
        token_id: u32,
        index: Option<&dyn crate::KvIndex>,
        handle: &mut KvHandle,
        abs_position: usize,
        window: Option<usize>,
    ) -> Option<Array2<f32>> {
        match window {
            None => self.coarse_decode_step(weights, token_id, index, handle, abs_position),
            Some(_) => None,
        }
    }

    /// Coarse prefill **with per-layer state capture** — same fast
    /// path as [`Self::coarse_prefill`] but also populates `state`
    /// (when `Some`) with per-layer h_in (residual entering each
    /// layer's attention block at every prompt position) and per-
    /// layer K/V (every position's K and V row, per layer). After a
    /// successful call, each entry in `state.h_in_per_layer` has
    /// shape `[seq_len, hidden]` and each entry in
    /// `state.k_new_per_layer` / `v_new_per_layer` has shape
    /// `[seq_len, kv_dim_for_layer]`. Engines (markov_residual,
    /// windowed_checkpoint, turbo_quant) read these to seed their
    /// state policy without re-running prefill on CPU.
    ///
    /// Default impl delegates to [`Self::coarse_prefill`] and leaves
    /// `state` untouched — backends that don't yet implement
    /// per-layer dump fall back, engine falls back to its per-layer
    /// CPU walk.
    fn coarse_prefill_with_state(
        &self,
        weights: &ModelWeights,
        token_ids: &[u32],
        index: Option<&dyn crate::KvIndex>,
        state: Option<&mut PerLayerDecodeState>,
    ) -> Option<(Array2<f32>, KvHandle)> {
        let _ = state;
        self.coarse_prefill(weights, token_ids, index)
    }

    /// One coarse decode step **with per-layer state capture** — the
    /// same fast path as [`Self::coarse_decode_step`] but also
    /// populates `state` (when `Some`) with per-layer h_in (residual
    /// entering each layer's attention block) and per-layer K_new /
    /// V_new (the new K/V row appended to that layer this step).
    ///
    /// Engines that need per-layer state to enforce their state
    /// policy — `markov_residual` (stores h_in per layer),
    /// `turbo_quant` (compresses per-layer K/V), `windowed_checkpoint`
    /// (snapshots K/V at window boundaries) — pass `Some(&mut state)`
    /// to extract per-layer state without re-running compute on CPU.
    ///
    /// On GPU backends the per-layer state is blit-copied from the
    /// Metal kernel's internal scratch buffers into CPU-visible
    /// buffers as part of the same command buffer that runs the
    /// decode — near-zero per-blit cost vs CPU per-layer re-walk.
    ///
    /// Default impl delegates to [`Self::coarse_decode_step`] and
    /// leaves `state` untouched, so backends that don't yet implement
    /// per-layer dump fall back to the per-layer CPU walk in the
    /// engine.
    fn coarse_decode_step_with_state(
        &self,
        weights: &ModelWeights,
        token_id: u32,
        index: Option<&dyn crate::KvIndex>,
        handle: &mut KvHandle,
        abs_position: usize,
        state: Option<&mut PerLayerDecodeState>,
    ) -> Option<Array2<f32>> {
        let _ = state;
        self.coarse_decode_step(weights, token_id, index, handle, abs_position)
    }

    /// Mask-aware variant of [`Self::coarse_decode_step_with_state`].
    ///
    /// Engines that treat K/V as **derivative** state can pass
    /// [`crate::StateDumpMask::HOnly`] to request only the h_in
    /// capture, skipping the K/V staging buffer alloc + GPU→CPU
    /// readback on backends that support it. The default impl
    /// ignores the mask and falls back to the full-capture path —
    /// correct on every backend, only Metal gains the perf saving
    /// today. See `crates/larql-kv/docs/state-policy.md` for the
    /// canonical vs derivative cut.
    #[allow(clippy::too_many_arguments)]
    fn coarse_decode_step_with_state_masked(
        &self,
        weights: &ModelWeights,
        token_id: u32,
        index: Option<&dyn crate::KvIndex>,
        handle: &mut KvHandle,
        abs_position: usize,
        state: Option<&mut PerLayerDecodeState>,
        mask: crate::StateDumpMask,
    ) -> Option<Array2<f32>> {
        let _ = mask;
        self.coarse_decode_step_with_state(weights, token_id, index, handle, abs_position, state)
    }

    /// Read K/V at `pos` for `layer` from the backend's internal kv
    /// cache. Returns `(k_row, v_row)` as flat `Vec<f32>` of length
    /// `kv_dim_for_layer`. Used by engines running under
    /// [`crate::StateDumpMask::HOnly`] that need to snapshot specific
    /// K/V positions on demand (e.g. `WindowedCheckpointEngine`'s
    /// `close_window` checkpoint emission).
    ///
    /// Default returns `None` — backends without an internal kv cache
    /// (CPU) or without the readback affordance (early-stage Metal)
    /// don't support it, and the engine falls back to its own shadow.
    /// `MetalBackend` overrides to read from `KVCache.layers[layer]`.
    fn read_kv_row_at(
        &self,
        handle: &KvHandle,
        layer: usize,
        pos: usize,
    ) -> Option<(Vec<f32>, Vec<f32>)> {
        let _ = (handle, layer, pos);
        None
    }

    // ── Norm + residual primitives ──────────────────────────────────

    /// Fused `residual_add + rmsnorm` for the post-attention or
    /// post-FFN residual write. Target for D-RMS-FUSE phase 2 work.
    ///
    /// Capability gate:
    /// [`crate::Capability::FusedResidualNorm`]. Default
    /// decomposes into separate add + rmsnorm calls on host (correct
    /// but slow); backends with fused kernels override.
    fn residual_norm_store(
        &self,
        x: &Array2<f32>,
        residual: &Array2<f32>,
        norm_weights: &[f32],
    ) -> Array2<f32> {
        // Default: decompose. add then rmsnorm.
        let added = x + residual;
        let mut out = Array2::<f32>::zeros(added.raw_dim());
        for (i, row) in added.rows().into_iter().enumerate() {
            let row_slice = row.as_slice().expect("non-contiguous row");
            let mean_sq: f32 =
                row_slice.iter().map(|v| v * v).sum::<f32>() / row_slice.len() as f32;
            let scale = (mean_sq + 1e-6).sqrt().recip();
            for (j, (val, w)) in row_slice.iter().zip(norm_weights.iter()).enumerate() {
                out[[i, j]] = val * scale * w;
            }
        }
        out
    }
}
