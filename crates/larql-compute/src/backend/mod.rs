//! Compute backend interface.
//!
//! `ComputeBackend` is the umbrella trait every caller takes as
//! `&dyn ComputeBackend`. It supertraits four narrower traits, each in
//! its own module so it's easy to read what a backend has to provide:
//!
//! | Sub-trait                     | What's there                                  |
//! |-------------------------------|-----------------------------------------------|
//! | [`MatMul`]                    | f32 / f16 matmul, gemv, batch matmul          |
//! | [`QuantMatVec`]               | unified `quant_matvec` + per-format helpers   |
//! | [`DecodeBackend`]             | KV-cached decode + prefill + MoE hook (Metal-shaped) |
//! | (umbrella) `ComputeBackend`   | `name`, `device_info`, [`Capability`] probe   |
//!
//! The engine-facing intent surface (`KvDispatch`) is a *sibling* of
//! `ComputeBackend`, not a sub-trait. It lives in `larql-inference`
//! (sibling to `FfnBackend`) so its CPU and Metal impls can call into
//! the inference-side forward-pass functions without inducing a dep
//! cycle on `larql-compute`. New [`Capability`] flags
//! (`FusedAttentionStep`, `WindowedAttentionStep`, `NativeKvCodec`,
//! `PipelinedBoundaryUpload`, `FusedResidualNorm`, `KvHandleNative`)
//! stay here — they describe what the *substrate* supports, regardless
//! of where the dispatch trait lives. See
//! `crates/larql-inference/docs/specs/compute-backend-redesign.md` §10.2.
//!
//! Most callers stay typed against `&dyn ComputeBackend`; the
//! sub-trait split is mainly an implementation-side organising
//! principle. Callers that want to branch on a specific accelerator
//! (e.g. "use f32_gemv if the backend has it, otherwise fall back to
//! matmul_transb") should use [`Capability`] + [`ComputeBackend::supports`]
//! instead of probing for `None` returns.

pub mod capability;
pub mod decode;
pub mod helpers;
pub mod matmul;
pub mod quant_matvec;

pub use capability::Capability;
pub use decode::{DecodeBackend, DecodeStateDump, ProfileTimings, StateDumpMask};
pub use helpers::{dot_proj_gpu, matmul_gpu};
pub use matmul::{MatMul, MatMulOp};
pub use quant_matvec::QuantMatVec;

/// `cuda-moe-multistream`: per-expert weight spec passed to
/// [`ComputeBackend::qwen35_moe_ffn_batch`]. Borrowed bytes — the
/// underlying `QuantTensor::raw_bytes()` views live for the duration
/// of the request via the model's Arc'd weights.
pub struct MoeFfnExpert<'a> {
    pub gate_data: &'a [u8],
    pub gate_format: crate::QuantFormat,
    pub up_data: &'a [u8],
    pub up_format: crate::QuantFormat,
    pub down_data: &'a [u8],
    pub down_format: crate::QuantFormat,
}

/// Hardware compute backend — the umbrella trait every caller binds.
///
/// Combines [`MatMul`] + [`QuantMatVec`] + [`DecodeBackend`] plus
/// metadata (`name`, `device_info`) and an explicit
/// [`Capability::supports`](Self::supports) probe. Most callers
/// shouldn't care which sub-trait a method comes from.
pub trait ComputeBackend: MatMul + QuantMatVec + DecodeBackend + Send + Sync {
    /// Human-readable backend name.
    fn name(&self) -> &str;

    /// Device info string (for logging/diagnostics).
    fn device_info(&self) -> String {
        self.name().to_string()
    }

    /// Whether this backend accelerates `cap`. Callers can branch on
    /// this *before* calling, instead of pattern-matching on `None`
    /// returns from probe methods.
    ///
    /// Default returns `false` for everything; backends override to
    /// enable. See [`Capability`] for the menu.
    fn supports(&self, _cap: Capability) -> bool {
        false
    }

    /// Optional Qwen3.6 Gated DeltaNet causal Conv1D step.
    ///
    /// Implementations mutate `state` in place and return the
    /// convolution output. Shape convention is row-major:
    /// `weight[d_conv, conv_dim]`, `state[d_conv - 1, conv_dim]`,
    /// `new[conv_dim]`.
    fn qwen35_causal_conv1d_step(
        &self,
        _weight: &[f32],
        _state: &mut [f32],
        _new: &[f32],
        _d_conv: usize,
        _conv_dim: usize,
        _sequence_pos: usize,
    ) -> Option<Vec<f32>> {
        None
    }

    /// Optional Qwen3.6 Gated DeltaNet recurrence step.
    ///
    /// Shape convention is row-major for the ndarray layouts used by
    /// `larql-inference`: `q[s, h_k]`, `k[s, h_k]`, `v[s, h_v]`, and
    /// `state[s, s, h_v]` with `h_v` as the fastest-moving axis.
    /// Implementations mutate `state` in place and return
    /// `out[s, h_v]`.
    #[allow(clippy::too_many_arguments)]
    fn qwen35_deltanet_step(
        &self,
        _q: &[f32],
        _k: &[f32],
        _v: &[f32],
        _log_g: &[f32],
        _beta: &[f32],
        _state: &mut [f32],
        _s: usize,
        _h_k: usize,
        _h_v: usize,
        _sequence_pos: usize,
        _block_gqa: bool,
    ) -> Option<Vec<f32>> {
        None
    }

    /// Optional Qwen3.6 per-head L2 normalisation.
    ///
    /// Shape convention is row-major for a logical
    /// `[head_dim, n_heads]` array, i.e. `x[d * n_heads + h]`.
    /// Implementations return the same layout.
    fn qwen35_l2_normalize_per_head(
        &self,
        _x: &[f32],
        _head_dim: usize,
        _n_heads: usize,
        _eps: f32,
    ) -> Option<Vec<f32>> {
        None
    }

    /// Optional Qwen3.6 per-head RMSNorm.
    ///
    /// Shape convention is head-major flat layout,
    /// `x[h * head_dim + d]`; `weight[d]` is shared across heads.
    fn qwen35_rms_norm_heads(
        &self,
        _x: &[f32],
        _weight: &[f32],
        _num_heads: usize,
        _head_dim: usize,
        _eps: f32,
    ) -> Option<Vec<f32>> {
        None
    }

    /// Fused Qwen3.6 DeltaNet recurrence block: L2 normalise Q and K,
    /// run the delta-rule recurrence with the cached device-resident
    /// state, then RMSNorm the head-major output. Four GPU kernels
    /// chained on the same stream with a single sync at exit, saving
    /// ~3 per-call sync round-trips per linear DeltaNet layer × 48
    /// layers ≈ 10-15 ms / token at the current decode rate.
    ///
    /// Inputs `q` / `k` (`[head_v_dim * n_k_heads]` dim-major flat),
    /// `v` (`[head_v_dim * n_v_heads]` dim-major flat), `log_g` /
    /// `beta` (`[n_v_heads]`) are produced by the unchanged host code
    /// (conv1d → silu → split/reshape on host), so the
    /// `silu(qkv_conv)` step that drove the E.6.A multi-position
    /// drift remains CPU. `ssm_norm_weight` is the per-head RMSNorm
    /// weight `[head_v_dim]`. `recurrent_state` is the per-layer
    /// host slice that the backend caches device-side, keyed by
    /// pointer; the kernel mutates the device buffer in place and
    /// leaves the host slice stale until a `sequence_pos == 0` reset.
    /// Returns the `[n_v_heads * head_v_dim]` head-major output ready
    /// for the caller to apply `silu(z) *` and the `ssm_out` matvec.
    #[allow(clippy::too_many_arguments)]
    fn qwen35_deltanet_recurrence_block(
        &self,
        _q: &[f32],
        _k: &[f32],
        _v: &[f32],
        _log_g: &[f32],
        _beta: &[f32],
        _ssm_norm_weight: &[f32],
        _recurrent_state: &mut [f32],
        _head_v_dim: usize,
        _n_v_heads: usize,
        _n_k_heads: usize,
        _eps: f32,
        _sequence_pos: usize,
        _block_gqa: bool,
    ) -> Option<Vec<f32>> {
        None
    }

    /// Fused Qwen3.6 SwiGLU FFN block on the GPU. Chains all three
    /// projection matvecs (`gate`, `up`, `down`) on the same stream
    /// with `silu(gate) * up` computed device-resident between them,
    /// so the host only ever sees the final `[hidden]` output.
    ///
    /// `gate_data` / `up_data` are the residual stream after the
    /// post-attn norm; both share the same `x` input. `down_data`
    /// projects the gated intermediate back to `hidden`. Formats are
    /// per-tensor (Qwen3.6-27B-Q4_K_S has `gate`/`up` as Q4_K and
    /// `down` as Q5_K), so the implementation must accept a mix.
    ///
    /// Saves the per-call htod x + dtoh intermediate transfers and 2
    /// stream syncs vs the host-bouncing path through
    /// `qwen35_paired_q4k_matvec` + CPU silu loop + `quant_matvec`.
    /// Returns `None` if the backend can't handle the format combo
    /// (caller falls back to per-call dispatch).
    #[allow(clippy::too_many_arguments)]
    fn qwen35_ffn_lazy_block(
        &self,
        _x: &[f32],
        _gate_data: &[u8],
        _gate_format: crate::QuantFormat,
        _gate_rows: usize,
        _up_data: &[u8],
        _up_format: crate::QuantFormat,
        _up_rows: usize,
        _down_data: &[u8],
        _down_format: crate::QuantFormat,
        _down_rows: usize,
        _hidden: usize,
    ) -> Option<Vec<f32>> {
        None
    }

    /// Paired Q4_K matvec sharing one `x` input. Two weight matrices
    /// (`a_rows × hidden` and `b_rows × hidden`) feed off the same
    /// activation; backend uploads `x` once, runs both kernels on the
    /// same stream, syncs once, returns `(a_out, b_out)`.
    ///
    /// Qwen3.6's DeltaNet block runs `attn_qkv` (10240 rows) and
    /// `attn_gate` (6144 rows) back-to-back on `x_norm`. The full-attn
    /// block similarly chains its Q/K/V projections. Pairing them
    /// halves the per-call sync overhead. `None` falls back to two
    /// independent `q4k_matvec` calls.
    #[allow(clippy::too_many_arguments)]
    fn qwen35_paired_q4k_matvec(
        &self,
        _a_data: &[u8],
        _a_rows: usize,
        _b_data: &[u8],
        _b_rows: usize,
        _x: &[f32],
        _hidden: usize,
    ) -> Option<(Vec<f32>, Vec<f32>)> {
        None
    }

    /// Optional fused Qwen3.6 DeltaNet post-projection chain (Phase
    /// E.6.A). Composes conv1d → silu → split + reshape → L2 Q/K →
    /// recurrence → reshape → rms_norm_heads → silu(z) * o on the
    /// device with a single sync at the end. Caller has already
    /// computed the projection outputs (`qkv_mixed`, `z`, `log_g`,
    /// `beta`) and supplies them as host slices.
    ///
    /// Returns the post-rms-norm, post-silu-z output `[value_dim]` in
    /// head-major layout, ready for the `ssm_out` projection. `None`
    /// means the backend declined to handle this call (CPU fallback).
    #[allow(clippy::too_many_arguments)]
    fn qwen35_deltanet_postproj_step(
        &self,
        _qkv_mixed: &[f32],
        _ssm_conv1d_weight: &[f32],
        _log_g: &[f32],
        _beta: &[f32],
        _z: &[f32],
        _ssm_norm_weight: &[f32],
        _conv_state: &mut [f32],
        _recurrent_state: &mut [f32],
        _head_v_dim: usize,
        _n_v_heads: usize,
        _n_k_heads: usize,
        _d_conv: usize,
        _eps: f32,
        _sequence_pos: usize,
        _block_gqa: bool,
    ) -> Option<Vec<f32>> {
        None
    }

    /// Optional Qwen 3.6 single-step GQA softmax attention.
    ///
    /// Caller has already projected + normed + roped a single new Q
    /// (`q[num_q_heads * head_dim]`) and appended the new K/V row to
    /// the cumulative cache slabs (`k_cache[seq_len, num_kv_heads *
    /// head_dim]`, `v_cache[seq_len, ...]`). The kernel computes
    /// stable softmax(Q·K^T) · V per head with GQA repeat-interleave
    /// (`Q head h` reads `KV head h / (num_q_heads / num_kv_heads)`).
    ///
    /// Returns `out[num_q_heads * head_dim]` head-major flat, or
    /// `None` to fall back to the host-side `gqa_decode_step` loop.
    ///
    /// `cache_id` identifies the per-layer device-resident KV slab the
    /// backend may keep across calls — the host slab `Array2<f32>`
    /// reallocates on every append (`Array2::zeros((N+1, dim))` +
    /// copy), so the pointer changes per token and can't itself be the
    /// key. Callers pass a stable id (e.g. the layer index, optionally
    /// combined with a per-sequence salt). `max_seq` sizes the device
    /// cache on first allocation; subsequent calls reuse it.
    ///
    /// The kernel writes the *new* row (last row of the host slab) to
    /// the device cache at `pos = seq_len - 1`, so previous tokens'
    /// device state survives across calls — no full htod per token in
    /// steady state.
    #[allow(clippy::too_many_arguments)]
    fn qwen35_gqa_decode_step(
        &self,
        _cache_id: u64,
        _max_seq: usize,
        _q: &[f32],
        _k_cache: &[f32],
        _v_cache: &[f32],
        _num_q_heads: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        _seq_len: usize,
    ) -> Option<Vec<f32>> {
        None
    }

    /// Drop the device-resident KV slab for `cache_id` (if any).
    ///
    /// Called from `DeltaNetHybridCache::reset` so per-sequence resets
    /// don't leak device memory across requests. Default no-op for
    /// backends that don't keep device-resident KV.
    fn qwen35_gqa_decode_reset(&self, _cache_id: u64) {}

    /// Phase 4b — batched-prefill GQA softmax attention over
    /// `seq_len` positions in a single kernel call.
    ///
    /// Inputs are host-RoPE'd / QK-normed per position by the caller
    /// (`qwen35_attention_block_step` host helpers — same convention
    /// as `qwen35_gqa_decode_step`). The kernel writes K/V rows
    /// `base_pos..base_pos + seq_len` into the device-resident
    /// cache identified by `cache_id` (the same slab the single-step
    /// method uses, so a subsequent decode-step call finds the
    /// populated cache). Returns `[seq_len * num_q_heads * head_dim]`
    /// row-major outputs.
    ///
    /// Default impl returns `None`. CUDA backend wraps
    /// `cuda::attn::fused_prefill_attention_seq_device_into`. Phase 4b
    /// ships the trait surface + CudaBackend impl + hardware smoke
    /// test; wiring this into the qwen35 attention block's prefill
    /// path is gated on Phase 4c/d (per-position projections / MoE /
    /// DeltaNet recurrence batched siblings) and lands in a later PR.
    #[allow(clippy::too_many_arguments)]
    fn qwen35_attention_prefill_batch(
        &self,
        _cache_id: u64,
        _max_seq: usize,
        _base_pos: usize,
        _q_seq: &[f32],
        _k_seq: &[f32],
        _v_seq: &[f32],
        _num_q_heads: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        _seq_len: usize,
    ) -> Option<Vec<f32>> {
        None
    }

    /// `cuda-moe-multistream`: parallel-expert MoE FFN dispatch.
    ///
    /// Issues all `experts.len()` (top_k, typically 8) expert chains
    /// concurrently on a pool of worker streams instead of serialising
    /// them on the default stream. Each expert is the standard
    /// SwiGLU triple — `gate @ x → silu(gate) * (up @ x) → down @ inter`
    /// — but the per-expert kernels overlap on the GPU when SM
    /// occupancy permits.
    ///
    /// Each `experts[i]` carries the gate/up/down quantised bytes +
    /// format + row count for one expert. All experts must share the
    /// same `hidden` (== input dim) and `ffn_dim` (gate/up output
    /// rows; down's input dim). The kernel does NOT apply the routing
    /// weights — the caller multiplies each returned output by its
    /// router-weight before summing.
    ///
    /// Returns one `Vec<f32>` of length `hidden` per expert (the
    /// `down @ inter` result, ready for weighted accumulation). `None`
    /// means the backend can't serve this shape; callers fall back to
    /// per-expert sequential dispatch via `swiglu_ffn_lazy`.
    fn qwen35_moe_ffn_batch(
        &self,
        _x: &[f32],
        _hidden: usize,
        _ffn_dim: usize,
        _experts: &[crate::backend::MoeFfnExpert<'_>],
    ) -> Option<Vec<Vec<f32>>> {
        None
    }

    /// Expose the concrete type for safe downcasting.
    fn as_any(&self) -> &dyn std::any::Any;

    /// Upload a Per-Layer Embeddings input table for the next
    /// `decode_token*` / `prefill_*` call.
    ///
    /// `flat` is laid out as `[(position * num_layers + layer) * ple_dim]`
    /// f32 elements. Backends that don't implement PLE do nothing — the
    /// upload is a no-op and the engine must skip the precompute work
    /// instead of relying on this method to validate inputs. Probe
    /// [`Capability::PerLayerEmbeddings`] to decide whether to call at all.
    fn prepare_ple_inputs(&self, _flat: &[f32], _num_layers: usize, _ple_dim: usize) {}

    /// Consume the per-stage timing recorded by the most recent
    /// `decode_token_split_profile` call on the current thread.
    ///
    /// Backends that record split-stage timings (Metal via paired
    /// commit/wait boundaries; future Vulkan/CUDA via their own
    /// timestamp APIs) return `Some(_)` when `LARQL_PROFILE_SPLIT=1`
    /// was honoured on the preceding call. Default returns `None` —
    /// engines treat that as "no instrumentation available."
    fn take_split_timings(&self) -> Option<ProfileTimings> {
        None
    }

    /// Run ONE layer of attention on the GPU and return the
    /// post-attention hidden state. The caller runs FFN externally
    /// (typically a `vindex` walk) and feeds the result back for the
    /// next layer.
    ///
    /// Steps the backend performs: input norm → QKV projection → RoPE →
    /// V-norm → KV append → KV attend → O projection → post-attention
    /// residual + post-attn norm.
    ///
    /// The backing KV cache is owned by the backend and ensured to
    /// match `kv_shapes` (one `(num_kv_heads, head_dim)` per absolute
    /// layer) before the dispatch. Default returns `None` — backends
    /// without GPU attention drop through to the engine's CPU
    /// fallback path. Probe [`Capability::HybridAttention`] before
    /// calling.
    #[allow(clippy::too_many_arguments)]
    fn hybrid_decode_attention_layer(
        &self,
        _layer: &crate::FullPipelineLayer<'_>,
        _layer_idx: usize,
        _x: &[f32],
        _hidden: usize,
        _q_dim: usize,
        _kv_dim: usize,
        _kv_shapes: &[(usize, usize)],
    ) -> Option<Vec<f32>> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, ArrayView2};

    /// Minimal `ComputeBackend` that uses every default impl. Lets the
    /// tests below exercise the trait's default bodies (which `CpuBackend`
    /// and `MetalBackend` would otherwise shadow with their overrides).
    struct StubBackend;

    impl MatMul for StubBackend {
        fn matmul(&self, _a: ArrayView2<f32>, _b: ArrayView2<f32>) -> Array2<f32> {
            unreachable!("stub backend MatMul never called")
        }
        fn matmul_transb(&self, _a: ArrayView2<f32>, _b: ArrayView2<f32>) -> Array2<f32> {
            unreachable!("stub backend MatMul never called")
        }
    }
    impl QuantMatVec for StubBackend {}
    impl DecodeBackend for StubBackend {}
    impl ComputeBackend for StubBackend {
        fn name(&self) -> &str {
            "stub"
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    #[test]
    fn default_device_info_falls_back_to_name() {
        let b = StubBackend;
        assert_eq!(b.device_info(), "stub");
    }

    #[test]
    fn default_supports_returns_false_for_every_capability() {
        let b = StubBackend;
        for cap in [
            Capability::F32Gemv,
            Capability::F16Gemv,
            Capability::QuantMatVec,
            Capability::Q4VecMat,
            Capability::Q4PairBatch,
            Capability::FullPipelineQ4,
            Capability::MultiLayerQ4Ffn,
            Capability::DecodeToken,
            Capability::DecodeMoe,
            Capability::DecodeQ4KMoe,
            Capability::DecodeProfile,
            Capability::PrefillQ4,
            Capability::HeterogeneousAttention,
            Capability::FusedAttentionStep,
            Capability::WindowedAttentionStep,
            Capability::NativeKvCodec,
            Capability::PipelinedBoundaryUpload,
            Capability::FusedResidualNorm,
            Capability::KvHandleNative,
            Capability::PerLayerEmbeddings,
            Capability::HybridAttention,
        ] {
            assert!(!b.supports(cap), "default supports must reject {cap:?}");
        }
    }

    #[test]
    fn default_prepare_ple_inputs_is_a_no_op() {
        let b = StubBackend;
        // No state to read back — we're just verifying the call returns
        // without panic. The default's job is to be a safe no-op so engines
        // that probe `Capability::PerLayerEmbeddings` and skip can still
        // safely call through.
        b.prepare_ple_inputs(&[0.0; 8], 2, 4);
        b.prepare_ple_inputs(&[], 0, 0);
    }

    #[test]
    fn default_take_split_timings_returns_none() {
        let b = StubBackend;
        assert!(b.take_split_timings().is_none());
    }

    #[test]
    fn default_hybrid_decode_attention_layer_returns_none() {
        let b = StubBackend;
        // The trait default ignores every argument; `FullPipelineLayer::default()`
        // provides the minimal layer fixture.
        let layer = crate::FullPipelineLayer::default();
        let kv_shapes = [(2usize, 4usize)];
        let result = b.hybrid_decode_attention_layer(&layer, 0, &[0.0; 8], 8, 8, 8, &kv_shapes);
        assert!(result.is_none());
    }

    /// `as_any` must downcast back to the concrete type.
    #[test]
    fn as_any_downcasts_to_concrete() {
        let b = StubBackend;
        let any = b.as_any();
        assert!(any.downcast_ref::<StubBackend>().is_some());
    }
}
