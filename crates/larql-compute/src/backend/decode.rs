//! `DecodeBackend` — full-pipeline KV-cached decode + prefill.
//!
//! These methods cover the autoregressive inference loop: prefill
//! (multi-position with KV-cache population), decode (single token
//! against the cache), MoE-aware decode, and per-stage timing.
//!
//! All methods default to `None` / no-op; only the GPU backend
//! implements them today (CPU runs decode through the higher-level
//! `larql-inference` path, not through `ComputeBackend`).
//!
//! All attention geometry (head_dim, num_q_heads, num_kv_heads,
//! rope_base, sliding_window, etc.) is read per-layer from
//! `FullPipelineLayer`. The trait surface intentionally does **not**
//! take scalar geometry parameters — passing them would invite a
//! single-layer fallback to silently corrupt heterogeneous models
//! like Gemma 4 31B (50 sliding-attention layers + 10 global-attention
//! layers, with different head_dim and num_kv_heads on each class).

/// Per-layer state captured during a fused decode step. Engines
/// (`markov_residual`, `markov_residual_codec`, `turbo_quant`) read
/// this to enforce their state policy without re-running the per-
/// layer compute on CPU. See
/// [`DecodeBackend::decode_token_with_state_dump`].
///
/// All three vectors have length `num_layers` after a successful
/// decode. Each per-layer entry is a flat `Vec<f32>` sized to the
/// layer's hidden / kv_dim respectively.
#[derive(Debug, Default, Clone)]
pub struct DecodeStateDump {
    /// Pre-attention residual entering each layer's attention block.
    /// Length: `num_layers`; each inner `Vec<f32>` is `hidden_size`.
    pub h_in_per_layer: Vec<Vec<f32>>,
    /// New K row appended this step, per layer.
    /// Length: `num_layers`; each inner `Vec<f32>` is `kv_dim_for_layer`.
    pub k_new_per_layer: Vec<Vec<f32>>,
    /// New V row appended this step, per layer.
    /// Length: `num_layers`; each inner `Vec<f32>` is `kv_dim_for_layer`.
    pub v_new_per_layer: Vec<Vec<f32>>,
}

impl DecodeStateDump {
    pub fn with_capacity(num_layers: usize) -> Self {
        Self {
            h_in_per_layer: Vec::with_capacity(num_layers),
            k_new_per_layer: Vec::with_capacity(num_layers),
            v_new_per_layer: Vec::with_capacity(num_layers),
        }
    }

    pub fn is_complete_for(&self, num_layers: usize) -> bool {
        self.h_in_per_layer.len() == num_layers
            && self.k_new_per_layer.len() == num_layers
            && self.v_new_per_layer.len() == num_layers
    }

    /// `is_complete_for` variant that respects the capture mask.
    /// Under `HOnly`, only `h_in_per_layer` is required to be
    /// populated; under `None`, the state dump is intentionally empty
    /// and the check trivially holds.
    pub fn is_complete_under(&self, num_layers: usize, mask: StateDumpMask) -> bool {
        match mask {
            StateDumpMask::Full => self.is_complete_for(num_layers),
            StateDumpMask::HOnly => self.h_in_per_layer.len() == num_layers,
            StateDumpMask::None => true,
        }
    }
}

/// Capture mask for [`DecodeBackend::decode_token_with_state_dump_masked`].
///
/// Engines that treat K/V as **derivative** state (see
/// `crates/larql-kv/docs/state-policy.md`) can request `HOnly` to
/// skip the GPU → CPU readback of K/V. The kernel does not blit K/V
/// to staging buffers under `HOnly`, eliminating both the transfer
/// and the bridge-layer wrap. The engine relies on the backend's own
/// kv cache for attention; on eviction it re-projects K/V from
/// canonical residual state (`MarkovResidualEngine::recompute_kv`).
///
/// Engines that treat K/V as **canonical** (e.g. `TurboQuantEngine`'s
/// compressed K/V, `WindowedCheckpointEngine`'s in-window K/V) must
/// use `Full`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum StateDumpMask {
    /// Capture h_in, k_new, v_new for every layer (today's default).
    #[default]
    Full,
    /// Capture h_in only. Skip the K/V staging buffer alloc, blit,
    /// and GPU → CPU readback. Backends without an optimised
    /// h-only path fall through to `Full` via the trait's default
    /// impl — engines get correct behavior, just no perf saving.
    HOnly,
    /// Capture nothing. Both h_in AND K/V skip staging + readback.
    /// W10 Phase C: only valid when the engine has no use for h_in
    /// on the dispatch path — e.g. `MarkovResidualEngine` configured
    /// with `window_size = None`, where `rs.stored` is never read
    /// after prefill (no cold-tier eviction, no recompute_kv). On
    /// backends without an optimised path, falls through to `Full`
    /// via the default trait impl — correct, no perf saving.
    None,
}

/// Per-stage wall-clock decode timings in milliseconds.
///
/// Filled by a backend's `decode_token_split_profile` path when the
/// caller sets `LARQL_PROFILE_SPLIT=1`. The shape is backend-agnostic
/// — Metal records GPU command-buffer timestamps, future Vulkan/CUDA
/// backends record their own equivalents. Engines read these back via
/// [`crate::ComputeBackend::take_split_timings`] after each decode call.
///
/// Granularity today (set by Metal) is **attention vs full FFN block**:
/// - `attn_ms` — Steps 1.5–5: QK-norm + RoPE + V-norm + KV append/attend
///   + O proj + post-attn residual + ffn-input norm.
/// - `gate_up_ms` — the **entire FFN block**: gate + up + activation
///   (GEGLU/SiLU) + down + post-FFN residual.
/// - `down_ms` — reserved for the next-finer split that breaks
///   `encode_ffn_step` into separate `gate_up` and `down` phases.
#[derive(Debug, Default, Clone, Copy)]
pub struct ProfileTimings {
    /// Wall time for the attention side of the layer:
    /// input norm → QKV proj → QK-norm → RoPE → KV-attend → O proj.
    pub attn_ms: f64,
    /// Wall time for the FFN gate + up + activation.
    pub gate_up_ms: f64,
    /// Wall time for the FFN down projection + post-FFN residual + scalar.
    pub down_ms: f64,
    /// Total time the GPU was actually executing, summed from
    /// `MTLCommandBuffer` start/end windows. Zero on CPU backends.
    ///
    /// Reported alongside [`Self::wall_ms`] so a reader can see that "GPU
    /// fwd" is wall time *around* the dispatch rather than GPU time.
    ///
    /// **The split is only valid against itself.** Turning profiling on
    /// costs 2.5× — qwen3-0.6b runs 94.8 tok/s unprofiled and 38.2 tok/s
    /// with `LARQL_PROFILE_SPLIT=1`, with `gpu_ms` steady near 2.8ms while
    /// the wall around it goes 4.3ms → 20ms. Almost all of the CPU-side
    /// share this reports is the instrument's own per-command-buffer
    /// synchronisation, not production cost. Use it to compare stages within
    /// one profiled run; do not quote the CPU/GPU ratio as a property of the
    /// unprofiled path.
    pub gpu_ms: f64,
    /// Wall time around the whole token, CPU encoding and readback included.
    pub wall_ms: f64,
    /// Command buffers submitted for this token. A large count is dispatch
    /// overhead the per-stage numbers do not show.
    pub cmd_buffers: u32,
}

impl ProfileTimings {
    /// Sum across the three buckets — the whole-token cost.
    pub fn total_ms(&self) -> f64 {
        self.attn_ms + self.gate_up_ms + self.down_ms
    }

    /// Format a `[profile-split] …` line for stderr instrumentation.
    pub fn format_summary(&self, num_layers: usize) -> String {
        let total = self.total_ms();
        let pct = |v: f64| if total > 0.0 { v / total * 100.0 } else { 0.0 };
        let per_layer = if num_layers > 0 {
            total / num_layers as f64
        } else {
            0.0
        };
        format!(
            "[profile-split] {num_layers} layers — \
             attn={:.2}ms ({:.0}%)  gate+up={:.2}ms ({:.0}%)  \
             down={:.2}ms ({:.0}%)  total={:.2}ms ({per_layer:.3}ms/layer)",
            self.attn_ms,
            pct(self.attn_ms),
            self.gate_up_ms,
            pct(self.gate_up_ms),
            self.down_ms,
            pct(self.down_ms),
            total,
        )
    }
}

/// What a backend needs in order to finish the token itself: final norm,
/// the lm_head matvec, and the partial top-K.
///
/// Handing this to [`DecodeBackend::decode_token_q4k_moe_head`] moves the
/// head *inside* the decode submission. The token then costs one
/// commit + wait instead of two, and the only thing crossing the host
/// boundary is the partial top-K — the hidden state never comes back.
///
/// The plan states the store's geometry rather than deriving it: `cols`
/// is the width the Q4_K rows are actually stored at (`hidden` rounded up
/// to whole super-blocks), and the query is zero-padded to it. A backend
/// that cannot honour some part of the plan returns `None` and the caller
/// stays on the unfused path — never a silently different head.
pub struct DecodeHeadPlan<'a> {
    /// Which norm the head applies. Stated rather than assumed: a
    /// backend that implements only one of them must refuse the other,
    /// because "applied the wrong normaliser" and "did not run" are the
    /// same `None` to the caller only if the refusal actually happens.
    pub norm_type: crate::NormType,
    /// Final norm weight, `[hidden]`.
    pub final_norm_weight: &'a [f32],
    /// Variance epsilon for the final norm.
    pub norm_eps: f32,
    /// Added to each norm weight before scaling (Gemma's `+ 1`).
    pub norm_offset: f32,
    /// Q4_K lm_head bytes, `[vocab, cols]` row-major.
    pub lm_head_q4k: &'a [u8],
    /// Rows in the lm_head store.
    pub vocab: usize,
    /// Stored row width in elements. Equals `hidden` when the hidden size
    /// is already a whole number of Q4_K super-blocks.
    pub cols: usize,
    /// How many `(token_id, score)` pairs to return, descending.
    pub top_k: usize,
}

/// KV-cached generation primitives.
///
/// "Backend supports decode" means the backend can run a full forward
/// pass internally — attention + FFN + KV cache update — without
/// returning intermediate residuals to the caller.
pub trait DecodeBackend {
    /// Full pipeline: ALL Q4 (attention + FFN) for all layers in ONE
    /// command buffer. Each layer: Q4 Q/K/V proj → fused attention →
    /// Q4 O proj → Q4 FFN. No CPU-GPU round-trips between layers.
    #[allow(clippy::too_many_arguments)]
    fn full_pipeline_q4(
        &self,
        _layers: &[crate::FullPipelineLayer<'_>],
        _x: &[f32],
        _hidden: usize,
        _inter: usize,
        _seq_len: usize,
        _use_qk_norm: bool,
        _softcap: f32,
    ) -> Option<Vec<f32>> {
        None
    }

    /// Like `full_pipeline_q4` but replaces one attention head's residual
    /// contribution at `target_layer` with `replacement_delta`.
    ///
    /// This is the Metal-accelerated path for Mode D head injection used by
    /// the AHORD CEGIS loop. Default delegates to `full_pipeline_q4` (no
    /// intervention — callers must fall back to the CPU path if this returns
    /// `None`).
    #[allow(clippy::too_many_arguments)]
    fn full_pipeline_q4_with_head_replacement(
        &self,
        layers: &[crate::FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize,
        inter: usize,
        seq_len: usize,
        use_qk_norm: bool,
        softcap: f32,
        target_layer: usize,
        target_head: usize,
        replacement_delta: &[f32],
    ) -> Option<Vec<f32>> {
        // Default: fall back to full pipeline without intervention.
        // Metal backend overrides this with the intervention-aware path.
        let _ = (target_layer, target_head, replacement_delta);
        self.full_pipeline_q4(layers, x, hidden, inter, seq_len, use_qk_norm, softcap)
    }

    /// Multi-layer Q4 FFN in one submission: gate → up → GEGLU → down.
    fn multi_layer_q4_ffn(
        &self,
        _layers_q4: &[(&[u8], &[u8], &[u8])],
        _x: &[f32],
        _inter: usize,
        _hidden: usize,
    ) -> Option<Vec<f32>> {
        None
    }

    /// Whether this backend supports KV-cache decode operations.
    fn has_kv_cache(&self) -> bool {
        false
    }

    /// Populate KV cache with prefill K/V data for one layer.
    ///
    /// `(num_kv_heads, head_dim)` here are this specific layer's
    /// geometry — not vestigial scalars. The caller must pass per-layer
    /// values (e.g. via `arch.num_kv_heads_for_layer(layer)`); a
    /// uniform-from-layer-0 fallback would corrupt heterogeneous models.
    fn populate_kv_layer(
        &self,
        _layer: usize,
        _k_data: &[f32],
        _v_data: &[f32],
        _seq_len: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
    ) {
    }

    /// Reset KV cache (for new prompt).
    fn reset_kv_cache(&self) {}

    /// Return the number of token positions currently committed to the KV cache.
    fn kv_cache_len(&self) -> usize {
        0
    }

    /// Roll back the KV cache to a previously saved length.  Safe to call with
    /// any `len ≤ current_len`; the physical K/V data below `len` is preserved
    /// (positions 0..len are not zeroed), so a subsequent decode pass starting
    /// from position `len` will produce correct attention over the prior tokens.
    ///
    /// Used by iterative predispatch: all but the final Metal pass call
    /// `truncate_kv_cache(saved_len)` so that only the last pass permanently
    /// advances the sequence length.
    fn truncate_kv_cache(&self, _len: usize) {}

    /// Pre-allocate the KV cache with per-layer shapes. Required for
    /// asymmetric attention geometry (Gemma 4 alternates sliding/global).
    fn preallocate_kv_cache_per_layer(&self, _shapes: &[(usize, usize)], _max_seq: usize) {}

    /// Pre-allocate with a per-layer *capacity* as well as a per-layer
    /// shape, so a sliding layer is not sized for history it can never
    /// reach.
    ///
    /// `capacities[i]` is the row count for layer `i`, from
    /// [`crate::pipeline_layer::kv_capacities_for_arch`]. The default impl
    /// ignores them and falls back to the uniform allocation, so a backend
    /// that has not opted in keeps its previous behaviour — over-allocating
    /// is wasteful, never wrong.
    fn preallocate_kv_cache_per_layer_with_capacity(
        &self,
        shapes: &[(usize, usize)],
        capacities: &[usize],
    ) {
        let max_seq = capacities.iter().copied().max().unwrap_or(0);
        self.preallocate_kv_cache_per_layer(shapes, max_seq);
    }

    /// Decode one token through all layers with KV cache.
    fn decode_token(
        &self,
        _layers: &[crate::FullPipelineLayer<'_>],
        _x: &[f32],
        _hidden: usize,
        _inter: usize,
    ) -> Option<Vec<f32>> {
        None
    }

    /// Decode one token with optional per-layer state capture
    /// (W1-GPU step 2). When `state` is `Some`, on success the
    /// backend populates each per-layer entry with the layer's
    /// `h_in` (pre-attention residual, shape `hidden`), `k_new` and
    /// `v_new` (newly-projected K/V row, shape `kv_dim_for_layer`).
    ///
    /// Default impl calls `decode_token` and leaves `state`
    /// untouched. Backends with a fused per-token kernel
    /// (`MetalBackend`) override to capture per-layer state via
    /// blit encodes inside the same command buffer — near-zero
    /// extra cost vs a CPU per-layer walk. Engines that need this
    /// (markov_residual, codec, turbo_quant) route through the
    /// trait method on `KvDispatch` which calls this in turn.
    fn decode_token_with_state_dump(
        &self,
        layers: &[crate::FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize,
        inter: usize,
        state: Option<&mut DecodeStateDump>,
    ) -> Option<Vec<f32>> {
        let _ = state;
        self.decode_token(layers, x, hidden, inter)
    }

    /// [`decode_token_with_state_dump`] variant that respects a
    /// capture [`StateDumpMask`].
    ///
    /// Under `StateDumpMask::HOnly`, backends with an optimised
    /// h-only path skip the K/V staging buffer alloc, blit, and
    /// readback. The default impl ignores the mask and falls through
    /// to the full-capture path — correct for any backend, no perf
    /// saving. `MetalBackend` overrides to take the optimised path.
    fn decode_token_with_state_dump_masked(
        &self,
        layers: &[crate::FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize,
        inter: usize,
        state: Option<&mut DecodeStateDump>,
        mask: StateDumpMask,
    ) -> Option<Vec<f32>> {
        let _ = mask;
        self.decode_token_with_state_dump(layers, x, hidden, inter, state)
    }

    /// Like `decode_token` but calls `moe_fn(layer, h_post_attn)` for
    /// MoE layers (enables remote expert dispatch). Default delegates
    /// to `decode_token` and ignores the hook.
    fn decode_token_with_moe(
        &self,
        layers: &[crate::FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize,
        inter: usize,
        _moe_fn: &mut dyn FnMut(usize, &[f32]) -> Vec<f32>,
    ) -> Option<Vec<f32>> {
        self.decode_token(layers, x, hidden, inter)
    }

    /// Decode one token while dispatching Q4_K per-layer expert tensors on
    /// the backend. The expert callback returns borrowed `(gate_up, down)`
    /// byte slices for the requested `(layer, expert)` pair.
    fn decode_token_q4k_moe<'w>(
        &self,
        _layers: &[crate::FullPipelineLayer<'_>],
        _x: &[f32],
        _hidden: usize,
        _inter: usize,
        _norm_eps: f32,
        _get_expert: &dyn Fn(usize, usize) -> Option<(&'w [u8], &'w [u8])>,
    ) -> Option<Vec<f32>> {
        None
    }

    /// [`Self::decode_token_q4k_moe`] with the LM head encoded into the
    /// same command buffer: final norm → lm_head matvec → partial top-K,
    /// one commit + wait, partials-only readback.
    ///
    /// Returns the `top_k` `(token_id, score)` pairs descending, or `None`
    /// when this backend cannot serve the plan as stated — in which case
    /// the caller runs the unfused path (readback → CPU norm → lm_head),
    /// which stays the reference the fused result is pinned against.
    #[allow(clippy::too_many_arguments)]
    fn decode_token_q4k_moe_head<'w>(
        &self,
        _layers: &[crate::FullPipelineLayer<'_>],
        _x: &[f32],
        _hidden: usize,
        _inter: usize,
        _norm_eps: f32,
        _get_expert: &dyn Fn(usize, usize) -> Option<(&'w [u8], &'w [u8])>,
        _head: &DecodeHeadPlan<'_>,
    ) -> Option<Vec<(u32, f32)>> {
        None
    }

    /// Split fire / collect variant of `decode_token_with_moe`.  At each MoE
    /// layer the implementation calls `moe_fire_fn(layer, h_post_attn)` once
    /// `h_post_attn` is computed, encodes dense FFN + post-FFN residual on a
    /// fresh command buffer, commits without waiting, then calls
    /// `moe_collect_fn(layer)` to retrieve the expert weighted-sum vector
    /// while the GPU runs the dense FFN in parallel.
    ///
    /// Default impl combines the two callbacks into a single synchronous
    /// closure and forwards to `decode_token_with_moe` — backends that don't
    /// support encoder splitting see no behaviour change.
    fn decode_token_with_moe_split(
        &self,
        layers: &[crate::FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize,
        inter: usize,
        moe_fire_fn: &mut dyn FnMut(usize, &[f32]),
        moe_collect_fn: &mut dyn FnMut(usize) -> Vec<f32>,
    ) -> Option<Vec<f32>> {
        // Default: synthesise a single synchronous moe_fn from the pair.
        let mut combined = |layer: usize, h: &[f32]| -> Vec<f32> {
            moe_fire_fn(layer, h);
            moe_collect_fn(layer)
        };
        self.decode_token_with_moe(layers, x, hidden, inter, &mut combined)
    }

    /// Like `decode_token` but splits each layer into attn / gate+up /
    /// down command buffers and times each. Returns `(result, attn_ms,
    /// gate_up_ms, down_ms)`. Default delegates to `decode_token` with
    /// zero timings.
    fn decode_token_split_profile(
        &self,
        layers: &[crate::FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize,
        inter: usize,
    ) -> (Option<Vec<f32>>, f64, f64, f64) {
        (self.decode_token(layers, x, hidden, inter), 0.0, 0.0, 0.0)
    }

    /// Multi-position prefill with KV-cache population. Stores
    /// post-RoPE K/V in the cache; returns the final hidden state
    /// `[seq_len * hidden]` for all positions.
    #[allow(clippy::too_many_arguments)]
    fn prefill_kquant(
        &self,
        _layers: &[crate::FullPipelineLayer<'_>],
        _x: &[f32],
        _hidden: usize,
        _inter: usize,
        _seq_len: usize,
        _use_qk_norm: bool,
        _softcap: f32,
    ) -> Option<Vec<f32>> {
        None
    }

    /// Capture the target head's pre-W_O output at `target_layer` via GPU,
    /// then stop. Returns `[seq_len × head_dim]` f32 — the raw attention output
    /// for `target_head` before W_O projection.
    ///
    /// For AHORD oracle code computation: runs only layers 0..=target_layer on GPU
    /// (not all 34 layers), giving ~34× speedup for target_layer=0 over CPU.
    #[allow(clippy::too_many_arguments)]
    fn full_pipeline_kquant_capture_pre_wo(
        &self,
        _layers: &[crate::FullPipelineLayer<'_>],
        _x: &[f32],
        _hidden: usize,
        _inter: usize,
        _seq_len: usize,
        _use_qk_norm: bool,
        _softcap: f32,
        _target_layer: usize,
        _target_head: usize,
    ) -> Option<Vec<f32>> {
        None
    }

    /// Like `prefill_kquant` but replaces one attention head's residual contribution
    /// at `target_layer` with `replacement_delta` — the AHORD Mode D injection path.
    ///
    /// Uses the same KV cache + per-position RoPE setup as `prefill_kquant`, so positional
    /// encodings are correct for all seq_len positions. Default returns `None`; the
    /// Metal backend overrides with the intervention-aware dispatch.
    #[allow(clippy::too_many_arguments)]
    fn prefill_kquant_with_head_replacement(
        &self,
        layers: &[crate::FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize,
        inter: usize,
        seq_len: usize,
        use_qk_norm: bool,
        softcap: f32,
        target_layer: usize,
        target_head: usize,
        replacement_delta: &[f32],
    ) -> Option<Vec<f32>> {
        let _ = (target_layer, target_head, replacement_delta);
        self.prefill_kquant(layers, x, hidden, inter, seq_len, use_qk_norm, softcap)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_total_ms_sums_buckets() {
        let p = ProfileTimings {
            attn_ms: 1.5,
            gate_up_ms: 2.5,
            down_ms: 1.0,
            ..Default::default()
        };
        assert!((p.total_ms() - 5.0).abs() < 1e-9);
    }

    #[test]
    fn profile_format_summary_handles_zero_total() {
        let p = ProfileTimings::default();
        let s = p.format_summary(34);
        // No NaN-percent panics, total prints as 0.00.
        assert!(s.contains("total=0.00ms"));
        assert!(s.contains("34 layers"));
    }

    #[test]
    fn profile_format_summary_includes_per_layer_average() {
        let p = ProfileTimings {
            attn_ms: 6.0,
            gate_up_ms: 3.0,
            down_ms: 1.0,
            ..Default::default()
        };
        let s = p.format_summary(10);
        assert!(s.contains("total=10.00ms"));
        assert!(s.contains("1.000ms/layer"));
    }

    /// `format_summary(0)` takes the `else` branch (per_layer = 0.0).
    #[test]
    fn profile_format_summary_zero_layers_uses_zero_per_layer() {
        let p = ProfileTimings {
            attn_ms: 1.0,
            gate_up_ms: 1.0,
            down_ms: 1.0,
            ..Default::default()
        };
        let s = p.format_summary(0);
        assert!(s.contains("0 layers"));
        assert!(s.contains("0.000ms/layer"));
    }

    // ── DecodeStateDump ────────────────────────────────────────────

    #[test]
    fn state_dump_with_capacity_preallocates_three_slots() {
        let dump = DecodeStateDump::with_capacity(8);
        assert_eq!(dump.h_in_per_layer.capacity(), 8);
        assert_eq!(dump.k_new_per_layer.capacity(), 8);
        assert_eq!(dump.v_new_per_layer.capacity(), 8);
        // The buffers are empty until the backend fills them.
        assert!(dump.h_in_per_layer.is_empty());
    }

    #[test]
    fn state_dump_is_complete_for_requires_every_layer_populated() {
        let mut dump = DecodeStateDump::with_capacity(2);
        assert!(!dump.is_complete_for(2));
        dump.h_in_per_layer.push(vec![0.0; 4]);
        dump.k_new_per_layer.push(vec![0.0; 4]);
        dump.v_new_per_layer.push(vec![0.0; 4]);
        assert!(!dump.is_complete_for(2)); // only 1 layer
        dump.h_in_per_layer.push(vec![0.0; 4]);
        dump.k_new_per_layer.push(vec![0.0; 4]);
        dump.v_new_per_layer.push(vec![0.0; 4]);
        assert!(dump.is_complete_for(2));
    }

    #[test]
    fn state_dump_is_complete_under_full_matches_is_complete_for() {
        let mut dump = DecodeStateDump::with_capacity(1);
        dump.h_in_per_layer.push(vec![0.0; 4]);
        // Under Full: K/V also required.
        assert!(!dump.is_complete_under(1, StateDumpMask::Full));
        dump.k_new_per_layer.push(vec![0.0; 4]);
        dump.v_new_per_layer.push(vec![0.0; 4]);
        assert!(dump.is_complete_under(1, StateDumpMask::Full));
    }

    #[test]
    fn state_dump_is_complete_under_h_only_ignores_kv() {
        let mut dump = DecodeStateDump::with_capacity(1);
        dump.h_in_per_layer.push(vec![0.0; 4]);
        // K/V never populated, but HOnly is satisfied by h_in alone.
        assert!(dump.is_complete_under(1, StateDumpMask::HOnly));
        // But still not Full.
        assert!(!dump.is_complete_under(1, StateDumpMask::Full));
    }

    #[test]
    fn state_dump_is_complete_under_none_is_trivially_true() {
        let dump = DecodeStateDump::default();
        // Nothing populated; None mask doesn't require anything.
        assert!(dump.is_complete_under(8, StateDumpMask::None));
    }

    #[test]
    fn state_dump_mask_default_is_full() {
        assert_eq!(StateDumpMask::default(), StateDumpMask::Full);
    }

    // ── DecodeBackend trait defaults ──────────────────────────────────
    //
    // A minimal stub that uses every default. Covers the trait's
    // default bodies — `None` returns, no-op writes, the delegating
    // defaults (e.g. `full_pipeline_q4_with_head_replacement` →
    // `full_pipeline_q4`).

    struct StubDecode;
    impl DecodeBackend for StubDecode {}

    fn stub_layers() -> Vec<crate::FullPipelineLayer<'static>> {
        vec![crate::FullPipelineLayer::default()]
    }

    #[test]
    fn default_full_pipeline_q4_returns_none() {
        let b = StubDecode;
        let layers = stub_layers();
        let r = b.full_pipeline_q4(&layers, &[0.0; 4], 4, 4, 1, false, 0.0);
        assert!(r.is_none());
    }

    #[test]
    fn default_full_pipeline_q4_with_head_replacement_delegates_to_no_intervention() {
        let b = StubDecode;
        let layers = stub_layers();
        // Default delegates to `full_pipeline_q4` (which is `None`).
        let r = b.full_pipeline_q4_with_head_replacement(
            &layers, &[0.0; 4], 4, 4, 1, false, 0.0, 0, 0, &[0.0; 2],
        );
        assert!(r.is_none());
    }

    #[test]
    fn default_multi_layer_q4_ffn_returns_none() {
        let b = StubDecode;
        let r = b.multi_layer_q4_ffn(&[], &[0.0; 4], 4, 4);
        assert!(r.is_none());
    }

    #[test]
    fn default_has_kv_cache_returns_false() {
        let b = StubDecode;
        assert!(!b.has_kv_cache());
    }

    #[test]
    fn default_kv_cache_helpers_are_no_ops() {
        let b = StubDecode;
        // None of these should panic; nothing observable changes.
        b.populate_kv_layer(0, &[0.0; 4], &[0.0; 4], 1, 2, 2);
        b.reset_kv_cache();
        b.truncate_kv_cache(0);
        b.preallocate_kv_cache_per_layer(&[(2, 4)], 16);
        assert_eq!(b.kv_cache_len(), 0);
    }

    /// The capacity-aware variant falls back to the uniform allocation
    /// using the *largest* requested capacity. Taking the max rather than
    /// the min matters: a backend that ignores per-layer capacity must
    /// over-allocate, never under-allocate — too small is an overrun,
    /// too large is only waste.
    #[test]
    fn default_capacity_preallocate_falls_back_to_the_largest_capacity() {
        let b = StubDecode;
        // Mixed sliding/global capacities, as a real model produces.
        b.preallocate_kv_cache_per_layer_with_capacity(&[(2, 4), (2, 4)], &[2048, 4096]);
        // Empty input must not panic on the `max` of an empty iterator.
        b.preallocate_kv_cache_per_layer_with_capacity(&[], &[]);
        assert_eq!(b.kv_cache_len(), 0);
    }

    #[test]
    fn default_decode_token_returns_none() {
        let b = StubDecode;
        let layers = stub_layers();
        let r = b.decode_token(&layers, &[0.0; 4], 4, 4);
        assert!(r.is_none());
    }

    #[test]
    fn default_decode_token_with_state_dump_delegates_to_decode_token() {
        let b = StubDecode;
        let layers = stub_layers();
        let mut dump = DecodeStateDump::default();
        let r = b.decode_token_with_state_dump(&layers, &[0.0; 4], 4, 4, Some(&mut dump));
        assert!(r.is_none());
        // Default doesn't populate the dump.
        assert!(dump.h_in_per_layer.is_empty());
    }

    #[test]
    fn default_decode_token_with_state_dump_masked_delegates_through() {
        let b = StubDecode;
        let layers = stub_layers();
        for mask in [
            StateDumpMask::Full,
            StateDumpMask::HOnly,
            StateDumpMask::None,
        ] {
            let r = b.decode_token_with_state_dump_masked(&layers, &[0.0; 4], 4, 4, None, mask);
            assert!(r.is_none(), "mask {mask:?} should produce None");
        }
    }

    #[test]
    fn default_decode_token_with_moe_ignores_hook() {
        let b = StubDecode;
        let layers = stub_layers();
        let mut hook_called = 0;
        let mut moe_fn = |_l: usize, _h: &[f32]| {
            hook_called += 1;
            vec![0.0; 4]
        };
        let r = b.decode_token_with_moe(&layers, &[0.0; 4], 4, 4, &mut moe_fn);
        assert!(r.is_none());
        // Default delegates to `decode_token` (None) before reaching layers,
        // so the MoE hook is never called.
        assert_eq!(hook_called, 0);
    }

    #[test]
    fn default_decode_token_q4k_moe_returns_none() {
        let b = StubDecode;
        let layers = stub_layers();
        let get_expert = |_l: usize, _e: usize| -> Option<(&[u8], &[u8])> { None };
        let r = b.decode_token_q4k_moe(&layers, &[0.0; 4], 4, 4, 1e-6, &get_expert);
        assert!(r.is_none());
    }

    #[test]
    fn default_decode_token_with_moe_split_combines_pair() {
        let b = StubDecode;
        let layers = stub_layers();
        let mut fire = |_l: usize, _h: &[f32]| {};
        let mut collect = |_l: usize| vec![0.0; 4];
        let r = b.decode_token_with_moe_split(&layers, &[0.0; 4], 4, 4, &mut fire, &mut collect);
        assert!(r.is_none());
    }

    #[test]
    fn default_decode_token_split_profile_returns_zero_timings() {
        let b = StubDecode;
        let layers = stub_layers();
        let (r, attn, gu, dn) = b.decode_token_split_profile(&layers, &[0.0; 4], 4, 4);
        assert!(r.is_none());
        assert_eq!(attn, 0.0);
        assert_eq!(gu, 0.0);
        assert_eq!(dn, 0.0);
    }

    #[test]
    fn default_prefill_kquant_returns_none() {
        let b = StubDecode;
        let layers = stub_layers();
        let r = b.prefill_kquant(&layers, &[0.0; 4], 4, 4, 1, false, 0.0);
        assert!(r.is_none());
    }

    #[test]
    fn default_full_pipeline_kquant_capture_pre_wo_returns_none() {
        let b = StubDecode;
        let layers = stub_layers();
        let r =
            b.full_pipeline_kquant_capture_pre_wo(&layers, &[0.0; 4], 4, 4, 1, false, 0.0, 0, 0);
        assert!(r.is_none());
    }

    #[test]
    fn default_prefill_kquant_with_head_replacement_delegates_to_no_intervention() {
        let b = StubDecode;
        let layers = stub_layers();
        let r = b.prefill_kquant_with_head_replacement(
            &layers, &[0.0; 4], 4, 4, 1, false, 0.0, 0, 0, &[0.0; 2],
        );
        // Default delegates to `prefill_kquant` (None).
        assert!(r.is_none());
    }
}
