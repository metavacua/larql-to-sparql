//! The decode entry points callers actually name.
//!
//! Each is a thin arrangement of arguments over
//! [`MetalBackend::decode_token_with_moe_split_fn`] in `token.rs`, which
//! holds the single implementation. They exist so a caller states its
//! intent — plain decode, decode with an MoE hook, decode that also
//! captures per-layer state — rather than passing a row of `None`s.

use super::*;

impl MetalBackend {
    /// Decode one token through all layers with KV cache.
    ///
    /// **Single command buffer**, one encoder per layer, no explicit barriers
    /// (Apple Silicon serialises compute dispatches within an encoder).
    ///
    /// Per-layer pipeline (~10 dispatches):
    ///   1. Input norm
    ///   2. Fused QKV projection (Q4_K or Q4_KF)
    ///   3. Batched RoPE (all Q heads), batched RoPE (all K heads)
    ///   4. Batched V-norm (optional, Gemma 4)
    ///   5. KV cache append + KV attend
    ///   6. O projection
    ///   7. Residual + norm (f32 for Q4_K/Q4_KF, +Q8 for Q4_0)
    ///   8. FFN: fused gate+up (Q4_K) or separate gate/up (Q4_KF) + GEGLU + down
    ///   9. Post-FFN residual + optional layer scalar
    ///
    /// Format-aware FFN routing:
    ///   - Q4_KF: llama.cpp-exact kernel (q4kf_proj) with register-cached input
    ///   - Q4_K:  fused gate+up kernel + q4k_matvec (uint4, 8 rows/TG, nr0=2)
    ///   - Q4_0:  legacy Q8-input path
    ///
    /// Decode one token with an optional MoE override function.
    ///
    /// When `moe_fn` is `Some`, it is called instead of `cpu_moe_forward` for
    /// every MoE layer.  Signature: `moe_fn(layer_idx, h_post_attn) -> Vec<f32>`.
    /// The returned vec must have length == `hidden`.  Pass `None` for the
    /// normal local-expert path.
    ///
    /// When `moe_collect_fn` is also `Some` the per-layer pipeline switches to
    /// the split-encoder layout: attention is committed and waited, `moe_fn`
    /// is invoked as a non-blocking *fire* (return value discarded), dense
    /// FFN + post-FFN residual are encoded on a fresh command buffer and
    /// committed without waiting, then `moe_collect_fn(layer)` is called to
    /// retrieve the expert output — letting the remote round trip overlap
    /// with the dense-FFN GPU work.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn decode_token_with_moe_fn(
        &self,
        kv_cache: &mut ops::kv_cache::KVCache,
        layers: &[larql_compute::FullPipelineLayer],
        x: &[f32],
        hidden: usize,
        inter: usize,
        q_dim: usize,
        kv_dim: usize,
        _num_q_heads: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        _rope_base: f32,
        moe_fn: Option<&mut dyn FnMut(usize, &[f32]) -> Vec<f32>>,
    ) -> Vec<f32> {
        // Backwards-compat wrapper: forward to the split-aware impl with no
        // collect callback and no state dump.
        self.decode_token_with_moe_split_fn(
            kv_cache,
            layers,
            x,
            hidden,
            inter,
            q_dim,
            kv_dim,
            _num_q_heads,
            _num_kv_heads,
            _head_dim,
            _rope_base,
            moe_fn,
            None,
            None,
            larql_compute::StateDumpMask::Full,
            None,
            None, // head — this wrapper returns the hidden state
        )
    }

    /// Decode one token AND capture per-layer state (W1-GPU step 2).
    ///
    /// Same compute path as `decode_token_with_moe_fn` (no MoE; the
    /// per-layer engines that consume state — markov_residual, codec,
    /// turbo_quant — target dense-FFN architectures). On exit, `state`
    /// is populated with per-layer `h_in` (pre-attention residual),
    /// `k_new` and `v_new` (newly-projected K/V row). Implementation
    /// forces a commit + CPU readback at the end of each layer so the
    /// scratch K/V buffers can be sampled before the next layer
    /// overwrites them; the per-layer-commit overhead (~50µs each ×
    /// num_layers) is the dominant cost vs the fully-fused
    /// single-commit path.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_token_with_state_dump_fn(
        &self,
        kv_cache: &mut ops::kv_cache::KVCache,
        layers: &[larql_compute::FullPipelineLayer],
        x: &[f32],
        hidden: usize,
        inter: usize,
        q_dim: usize,
        kv_dim: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope_base: f32,
        state: &mut larql_compute::DecodeStateDump,
    ) -> Vec<f32> {
        self.decode_token_with_state_dump_masked_fn(
            kv_cache,
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
            state,
            larql_compute::StateDumpMask::Full,
        )
    }

    /// Mask-aware variant of [`Self::decode_token_with_state_dump_fn`].
    /// W10 Phase B: engines that treat K/V as derivative can pass
    /// [`larql_compute::StateDumpMask::HOnly`] to skip the K/V staging +
    /// readback.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_token_with_state_dump_masked_fn(
        &self,
        kv_cache: &mut ops::kv_cache::KVCache,
        layers: &[larql_compute::FullPipelineLayer],
        x: &[f32],
        hidden: usize,
        inter: usize,
        q_dim: usize,
        kv_dim: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope_base: f32,
        state: &mut larql_compute::DecodeStateDump,
        mask: larql_compute::StateDumpMask,
    ) -> Vec<f32> {
        self.decode_token_with_moe_split_fn(
            kv_cache,
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
            None,
            None,
            Some(state),
            mask,
            None,
            None, // head — state-dump callers consume the hidden state
        )
    }

    /// Split fire / collect variant of `decode_token_with_moe_fn`.  See the
    /// trait method `DecodeBackend::decode_token_with_moe_split` for the
    /// motivating use case (within-layer GPU/MoE overlap).
    /// Local-expert path — delegates to `decode_token_with_moe_fn` with no hook.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_token(
        &self,
        kv_cache: &mut ops::kv_cache::KVCache,
        layers: &[larql_compute::FullPipelineLayer],
        x: &[f32],
        hidden: usize,
        inter: usize,
        q_dim: usize,
        kv_dim: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope_base: f32,
    ) -> Vec<f32> {
        self.decode_token_with_moe_fn(
            kv_cache,
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
            None,
        )
    }
}
