//! `DecodeBackend` — full-pipeline KV-cached decode + prefill.
//!
//! These methods cover the autoregressive inference loop: prefill
//! (multi-position with KV-cache population), decode (single token
//! against the cache), MoE-aware decode, and per-stage timing.
//!
//! All methods default to `None` / no-op; only the GPU backend
//! implements them today (CPU runs decode through the higher-level
//! `larql-inference` path, not through `ComputeBackend`).

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
/// compressed K/V, `UnlimitedContextEngine`'s in-window K/V) must
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
        _q_dim: usize,
        _kv_dim: usize,
        _seq_len: usize,
        _num_q_heads: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        _rope_base: f32,
        _use_qk_norm: bool,
        _softcap: f32,
    ) -> Option<Vec<f32>> {
        None
    }

    /// Variant of [`Self::full_pipeline_q4`] that replaces head `target_head`
    /// at `target_layer` with `replacement_delta` (a `[seq_len * head_dim]`
    /// f32 slice). The kernel zeros that head's pre-W_O output then adds
    /// the supplied delta in its place, so the rest of the forward pass
    /// sees the intervened residual.
    ///
    /// Used by the `dev ov_rd` circuit-intervention CLI (eval-program /
    /// induce-program) to probe head contributions on the GPU.
    ///
    /// Default impl returns `None` so backends without the intervention
    /// hook fall back to the CPU path.
    #[allow(clippy::too_many_arguments)]
    fn full_pipeline_q4_with_head_replacement(
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
        _replacement_delta: &[f32],
    ) -> Option<Vec<f32>> {
        None
    }

    /// Variant of [`Self::full_pipeline_q4`] that captures the pre-W_O
    /// output of head `target_head` at `target_layer` (i.e. the head's
    /// contribution to the residual stream before the attention output
    /// projection mixes heads together). After capturing, the dispatcher
    /// stops short of running the rest of the layers; the returned vec
    /// is `[seq_len * head_dim]` f32.
    ///
    /// Used by the `dev ov_rd` oracle PQ flow to compute residual
    /// fingerprints from one head's output without paying the cost of
    /// finishing the forward pass.
    ///
    /// Default impl returns `None` so backends without the capture hook
    /// fall back to the CPU path.
    #[allow(clippy::too_many_arguments)]
    fn full_pipeline_q4_capture_pre_wo(
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

    /// Decode one token through all layers with KV cache.
    #[allow(clippy::too_many_arguments)]
    fn decode_token(
        &self,
        _layers: &[crate::FullPipelineLayer<'_>],
        _x: &[f32],
        _hidden: usize,
        _inter: usize,
        _q_dim: usize,
        _kv_dim: usize,
        _num_q_heads: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        _rope_base: f32,
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
    #[allow(clippy::too_many_arguments)]
    fn decode_token_with_moe(
        &self,
        layers: &[crate::FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize,
        inter: usize,
        q_dim: usize,
        kv_dim: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope_base: f32,
        _moe_fn: &mut dyn FnMut(usize, &[f32]) -> Vec<f32>,
    ) -> Option<Vec<f32>> {
        self.decode_token(
            layers,
            x,
            hidden,
            inter,
            q_dim,
            kv_dim,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rope_base,
        )
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
    #[allow(clippy::too_many_arguments)]
    fn decode_token_with_moe_split(
        &self,
        layers: &[crate::FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize,
        inter: usize,
        q_dim: usize,
        kv_dim: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope_base: f32,
        moe_fire_fn: &mut dyn FnMut(usize, &[f32]),
        moe_collect_fn: &mut dyn FnMut(usize) -> Vec<f32>,
    ) -> Option<Vec<f32>> {
        // Default: synthesise a single synchronous moe_fn from the pair.
        let mut combined = |layer: usize, h: &[f32]| -> Vec<f32> {
            moe_fire_fn(layer, h);
            moe_collect_fn(layer)
        };
        self.decode_token_with_moe(
            layers,
            x,
            hidden,
            inter,
            q_dim,
            kv_dim,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rope_base,
            &mut combined,
        )
    }

    /// Like `decode_token` but splits each layer into attn / gate+up /
    /// down command buffers and times each. Returns `(result, attn_ms,
    /// gate_up_ms, down_ms)`. Default delegates to `decode_token` with
    /// zero timings.
    #[allow(clippy::too_many_arguments)]
    fn decode_token_split_profile(
        &self,
        layers: &[crate::FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize,
        inter: usize,
        q_dim: usize,
        kv_dim: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope_base: f32,
    ) -> (Option<Vec<f32>>, f64, f64, f64) {
        (
            self.decode_token(
                layers,
                x,
                hidden,
                inter,
                q_dim,
                kv_dim,
                num_q_heads,
                num_kv_heads,
                head_dim,
                rope_base,
            ),
            0.0,
            0.0,
            0.0,
        )
    }

    /// Speculative-decode forward: process `x_per_token.len()` tokens
    /// through all layers with KV cache, returning the per-position
    /// hidden state for each.
    ///
    /// **Cache semantics**: this method advances the KV cache during
    /// processing then RESTORES it to its pre-call length. The
    /// caller (typically `larql_inference::speculative`) explores N
    /// candidate tokens without committing them; on acceptance, the
    /// caller separately calls `decode_token` for each accepted
    /// token to commit it permanently to the cache.
    ///
    /// Returns `Some(hiddens)` on success — `hiddens.len() ==
    /// x_per_token.len()`, with `hiddens[k]` the `[hidden]`-shaped
    /// output of the target's forward pass after consuming
    /// `x_per_token[..=k]`. Returns `None` if any per-token
    /// `decode_token` call fails (cache state restored regardless).
    ///
    /// **Default impl**: sequential `decode_token` calls + cache
    /// rollback via `truncate_kv_cache`. Backends with batched
    /// kernels (`cuda::q4k_batched` + `cuda::attn_tree`) MAY
    /// override for the perf win — phase 4c task C.2 implements
    /// this for `CudaBackend`.
    #[allow(clippy::too_many_arguments)]
    fn decode_tokens_speculative(
        &self,
        layers: &[crate::FullPipelineLayer<'_>],
        x_per_token: &[Vec<f32>],
        hidden: usize,
        inter: usize,
        q_dim: usize,
        kv_dim: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope_base: f32,
    ) -> Option<Vec<Vec<f32>>> {
        if !self.has_kv_cache() {
            return None;
        }
        let pre_len = self.kv_cache_len();
        let mut hiddens: Vec<Vec<f32>> = Vec::with_capacity(x_per_token.len());
        for x in x_per_token {
            match self.decode_token(
                layers,
                x,
                hidden,
                inter,
                q_dim,
                kv_dim,
                num_q_heads,
                num_kv_heads,
                head_dim,
                rope_base,
            ) {
                Some(h) => hiddens.push(h),
                None => {
                    // Restore cache before bailing.
                    self.truncate_kv_cache(pre_len);
                    return None;
                }
            }
        }
        self.truncate_kv_cache(pre_len);
        Some(hiddens)
    }

    /// `cuda-spec-branching-tree` T3.2: tree-shaped variant of
    /// [`Self::decode_tokens_speculative_keep_cache`]. Processes a
    /// branching draft tree in one batched forward, masking attention
    /// per-node so siblings don't leak.
    ///
    /// `ancestors[n]` SHALL be the u64 bitset for node `n` (per
    /// `DraftTree::ancestor_bitsets()`). `ancestors.len() ==
    /// x_per_node.len() <= 64`.
    ///
    /// On `None` (backend lacks tree-mask kernel, or tree size
    /// exceeds the kernel's cap), the caller SHALL fall back to the
    /// per-node `decode_token` path. Cache is restored to `pre_len`
    /// on any error.
    ///
    /// Default impl returns `None` so backends without the tree-mask
    /// kernel route to the conservative per-node fallback.
    #[allow(clippy::too_many_arguments)]
    fn decode_tokens_speculative_tree_keep_cache(
        &self,
        _layers: &[crate::FullPipelineLayer<'_>],
        _x_per_node: &[Vec<f32>],
        _ancestors: &[u64],
        _hidden: usize,
        _inter: usize,
        _q_dim: usize,
        _kv_dim: usize,
        _num_q_heads: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        _rope_base: f32,
    ) -> Option<Vec<Vec<f32>>> {
        None
    }

    /// Variant of [`Self::decode_tokens_speculative`] that does NOT
    /// truncate the KV cache on success — the cache is left advanced
    /// by `x_per_token.len()` positions. The CALLER is responsible for
    /// truncating to the desired position (typically `pre_len + R`
    /// where R is the number of accepted tokens after `verify_tree`)
    /// and re-decoding the bonus token at position `pre_len + R`.
    ///
    /// **Why this exists** (phase 4c skip-redundant-commit): the
    /// previous flow ran the helper, truncated cache back to pre_len,
    /// then re-decoded ALL R+1 emitted tokens to commit them. The
    /// first R of those re-decodes were redundant — the helper had
    /// just decoded the same R tokens (drafts[0..R-1]). With this
    /// keep-cache variant, the helper's chain decode commits drafts
    /// to cache; the caller then truncates to drop drafts[R..N-1]
    /// and re-decodes only the bonus (which is a resampled token,
    /// not equal to drafts[R]).
    ///
    /// On error, cache IS restored to pre_len (matches the truncate
    /// variant's error semantics).
    ///
    /// Returns `Some(hiddens)` with `hiddens[k]` = post-token-k hidden,
    /// same shape as [`Self::decode_tokens_speculative`].
    #[allow(clippy::too_many_arguments)]
    fn decode_tokens_speculative_keep_cache(
        &self,
        layers: &[crate::FullPipelineLayer<'_>],
        x_per_token: &[Vec<f32>],
        hidden: usize,
        inter: usize,
        q_dim: usize,
        kv_dim: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope_base: f32,
    ) -> Option<Vec<Vec<f32>>> {
        if !self.has_kv_cache() {
            return None;
        }
        let pre_len = self.kv_cache_len();
        let mut hiddens: Vec<Vec<f32>> = Vec::with_capacity(x_per_token.len());
        for x in x_per_token {
            match self.decode_token(
                layers,
                x,
                hidden,
                inter,
                q_dim,
                kv_dim,
                num_q_heads,
                num_kv_heads,
                head_dim,
                rope_base,
            ) {
                Some(h) => hiddens.push(h),
                None => {
                    // Restore cache before bailing — same as the
                    // truncate variant's error semantics so callers
                    // don't have to handle a partial-advance state.
                    self.truncate_kv_cache(pre_len);
                    return None;
                }
            }
        }
        // NO truncate on success — cache is left at pre_len + n.
        Some(hiddens)
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
        _q_dim: usize,
        _kv_dim: usize,
        _seq_len: usize,
        _num_q_heads: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        _rope_base: f32,
        _use_qk_norm: bool,
        _softcap: f32,
    ) -> Option<Vec<f32>> {
        None
    }
<<<<<<< HEAD

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
=======
>>>>>>> ianblenke/main
}

#[cfg(test)]
mod tests {
    use super::*;
<<<<<<< HEAD

    #[test]
    fn profile_total_ms_sums_buckets() {
        let p = ProfileTimings {
            attn_ms: 1.5,
            gate_up_ms: 2.5,
            down_ms: 1.0,
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
=======
    use crate::cpu::CpuBackend;

    #[test]
    fn cpu_full_pipeline_q4_with_head_replacement_returns_none() {
        // CPU has no GPU intervention hook — default impl returns None,
        // so callers fall back to a CPU intervention path implemented at
        // the larql-inference layer.
        let backend = CpuBackend;
        let layers: Vec<crate::FullPipelineLayer<'_>> = Vec::new();
        let out = backend.full_pipeline_q4_with_head_replacement(
            &layers,
            &[],
            16,
            32,
            1,
            false,
            0.0,
            0,
            0,
            &[],
        );
        assert!(
            out.is_none(),
            "CPU backend must NOT support head replacement"
        );
    }

    #[test]
    fn cpu_full_pipeline_q4_capture_pre_wo_returns_none() {
        let backend = CpuBackend;
        let layers: Vec<crate::FullPipelineLayer<'_>> = Vec::new();
        let out =
            backend.full_pipeline_q4_capture_pre_wo(&layers, &[], 16, 32, 1, false, 0.0, 0, 0);
        assert!(
            out.is_none(),
            "CPU backend must NOT support pre-W_O capture"
        );
>>>>>>> ianblenke/main
    }
}
