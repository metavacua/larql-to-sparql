//! GPT-OSS architecture — OpenAI's MoE model with MXFP4 packed experts.
//!
//! Key differences from standard MoE (Mixtral):
//! - Expert weights are packed as MXFP4 (e8m0 scales + 4-bit values)
//! - Gate and up projections are fused into `gate_up_proj_blocks`,
//!   **interleaved** along the output axis — gate on the even rows, up on the
//!   odd. This comment claimed "first half = gate" until 2026-07-30, and the
//!   loader and extractor both believed it; the resulting halves were each a
//!   50/50 mixture of the two projections. See `docs/k3-funnel.md` §4.7.
//! - The expert MLP is **not** SwiGLU: both halves are clamped at
//!   ±`swiglu_limit`, the sigmoid argument is scaled by 1.702, and the up
//!   branch gets +1 — see [`ExpertGatePolicy::ClampedGlu`]
//! - Router and both expert projections carry biases
//! - Router softmaxes over the **selected** top-k logits, not all experts
//! - All experts packed in one tensor per layer, not per-expert files
//! - Router at `mlp.router.weight` (not `block_sparse_moe.gate`)
//! - Attention has biases and sinks (both declared below and asserted in
//!   the tests — this comment claimed them for months while nothing
//!   returned the keys, so extraction dropped them), and uses GQA
//! - YaRN RoPE scaling

use crate::config::{
    ExpertFormat, ExpertGatePolicy, ExpertRoutingPolicy, GateUpLayout, ModelArchitecture,
    ModelConfig,
};
use crate::tensor_keys::{attn_bias, mxfp4_dequantised};

/// Multiplier on the sigmoid argument in the expert GLU. Not a config field —
/// `GptOssExperts.alpha` is a literal in the reference implementation.
const SWIGLU_ALPHA: f32 = 1.702;

/// Clamp bound used when `config.json` omits `swiglu_limit`. Every released
/// GPT-OSS checkpoint ships the field; this only covers a hand-written config.
const DEFAULT_SWIGLU_LIMIT: f32 = 7.0;

pub struct GptOssArch {
    config: ModelConfig,
}

impl GptOssArch {
    pub fn from_config(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArchitecture for GptOssArch {
    fn family(&self) -> &str {
        "gpt_oss"
    }

    /// `GptOssConfig.rms_norm_eps` defaults to 1e-5, not the crate-wide 1e-6.
    /// `openai/gpt-oss-20b` ships the field explicitly so this fallback does
    /// not fire for it — it is declared so a *sibling* checkpoint that omits
    /// it cannot inherit the wrong family's value, which is exactly how OLMoE
    /// was mis-served (see [`super::olmoe`]).
    fn default_norm_eps(&self) -> f32 {
        crate::defaults::DEFAULT_NORM_EPS_1E5
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn key_prefixes_to_strip(&self) -> &[&str] {
        &["model."]
    }

    // ── Attention ──

    fn attn_q_key(&self, layer: usize) -> String {
        format!("{}self_attn.q_proj.weight", self.layer_prefix(layer))
    }

    fn attn_k_key(&self, layer: usize) -> String {
        format!("{}self_attn.k_proj.weight", self.layer_prefix(layer))
    }

    fn attn_v_key(&self, layer: usize) -> String {
        format!("{}self_attn.v_proj.weight", self.layer_prefix(layer))
    }

    fn attn_o_key(&self, layer: usize) -> String {
        format!("{}self_attn.o_proj.weight", self.layer_prefix(layer))
    }

    // ── MoE ──

    fn is_moe(&self) -> bool {
        true
    }

    fn expert_format(&self) -> ExpertFormat {
        ExpertFormat::PackedMxfp4
    }

    /// `GptOssExperts` splits with `gate_up[..., ::2], gate_up[..., 1::2]`
    /// — interleaved, unlike every other packed family here. This describes
    /// the *checkpoint*; the MXFP4→Q6_K transcode canonicalises the operand
    /// it hands the execution path, and that transform owns its own
    /// invariant rather than inheriting this one.
    fn gate_up_layout(&self) -> Option<GateUpLayout> {
        Some(GateUpLayout::Interleaved)
    }

    fn num_experts(&self) -> usize {
        self.config.num_experts.unwrap_or(128)
    }

    fn num_experts_per_token(&self) -> usize {
        self.config.num_experts_per_token.unwrap_or(4)
    }

    fn moe_router_key(&self, layer: usize) -> Option<String> {
        Some(format!("{}mlp.router.weight", self.layer_prefix(layer)))
    }

    fn moe_router_bias_key(&self, layer: usize) -> Option<String> {
        Some(format!("{}mlp.router.bias", self.layer_prefix(layer)))
    }

    fn moe_router_kind(&self) -> crate::MoeRouterKind {
        crate::MoeRouterKind::TopKThenSoftmax
    }

    /// GPT-OSS's experts use the model's own `intermediate_size` — there is no
    /// separate `moe_intermediate_size` in its config. Without this override
    /// the trait default returns 0, which sizes every expert matmul to nothing.
    fn moe_intermediate_size(&self) -> usize {
        self.config
            .moe_intermediate_size
            .unwrap_or(self.config.intermediate_size)
    }

    fn expert_gate_policy(&self) -> ExpertGatePolicy {
        ExpertGatePolicy::ClampedGlu {
            limit: self
                .config
                .swiglu_limit
                .map(|v| v as f32)
                .unwrap_or(DEFAULT_SWIGLU_LIMIT),
            alpha: SWIGLU_ALPHA,
        }
    }

    /// GPT-OSS softmaxes over the **selected** logits, so the weights sum to 1.
    fn expert_routing_policy(&self) -> ExpertRoutingPolicy {
        ExpertRoutingPolicy::NormalisedOverSelected
    }

    /// The MLP-input norm, for serving paths where the norm is the MoE
    /// block's responsibility rather than the caller's.
    ///
    /// In the reference, `post_attention_layernorm` runs inside the layer
    /// before the router and experts, and GPT-OSS has no separate
    /// pre-experts tensor — so this *aliases* the post-attention norm
    /// rather than naming a new one. The f32 tier applies that norm in
    /// `run_ffn` before the FFN backend is called; the quantised block
    /// path hands the MoE the raw residual and resolves the norm through
    /// this key (`MoeRoutingPolicy::top_k_then_softmax` declares
    /// `PreExpertsNorm`). Without it, every router and expert ran on the
    /// un-normed stream and generation was incoherent while nothing
    /// crashed.
    fn moe_pre_experts_norm_key(&self, layer: usize) -> Option<String> {
        Some(self.post_attention_layernorm_key(layer))
    }

    // ── Attention biases + sinks ──
    //
    // The module header has claimed "attention has biases, sinks" since
    // this file was written, but nothing declared them, so extraction
    // silently dropped 5 of the 11 attention tensors each layer (four
    // projection biases + sinks = 120 tensors on the 20B). Declared here
    // 2026-07-29; see `docs/k3-funnel.md` §4.6.

    fn attn_q_bias_key(&self, layer: usize) -> Option<String> {
        attn_bias::q(&self.layer_prefix(layer))
    }

    fn attn_k_bias_key(&self, layer: usize) -> Option<String> {
        attn_bias::k(&self.layer_prefix(layer))
    }

    fn attn_v_bias_key(&self, layer: usize) -> Option<String> {
        attn_bias::v(&self.layer_prefix(layer))
    }

    fn attn_o_bias_key(&self, layer: usize) -> Option<String> {
        attn_bias::o(&self.layer_prefix(layer))
    }

    fn attn_sinks_key(&self, layer: usize) -> Option<String> {
        Some(format!("{}self_attn.sinks", self.layer_prefix(layer)))
    }

    // ── Packed MXFP4 expert keys ──

    fn packed_gate_up_blocks_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}mlp.experts.gate_up_proj_blocks",
            self.layer_prefix(layer)
        ))
    }

    fn packed_gate_up_scales_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}mlp.experts.gate_up_proj_scales",
            self.layer_prefix(layer)
        ))
    }

    fn packed_down_blocks_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}mlp.experts.down_proj_blocks",
            self.layer_prefix(layer)
        ))
    }

    fn packed_down_scales_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}mlp.experts.down_proj_scales",
            self.layer_prefix(layer)
        ))
    }

    fn packed_gate_up_bias_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}mlp.experts.gate_up_proj_bias",
            self.layer_prefix(layer)
        ))
    }

    fn packed_down_bias_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}mlp.experts.down_proj_bias",
            self.layer_prefix(layer)
        ))
    }

    // ── Per-expert keys, post-dequantisation ──
    //
    // On disk GPT-OSS has no per-expert tensors — everything is packed and
    // fused. But the safetensors loader dequantises and de-interleaves the
    // experts at load time, storing each one separately, so a compute backend
    // reading `ModelWeights` *does* see per-expert weights. Advertising them
    // here is what lets a generic per-expert FFN backend serve this model
    // without knowing anything about MXFP4.
    //
    // These keys therefore describe loaded state, not the checkpoint. Callers
    // that read the checkpoint directly (extraction) still want `packed_*`.

    fn expert_ffn_gate_key(&self, layer: usize, expert_id: usize) -> Option<String> {
        mxfp4_dequantised::gate_proj(&self.layer_prefix(layer), expert_id)
    }

    fn expert_ffn_up_key(&self, layer: usize, expert_id: usize) -> Option<String> {
        mxfp4_dequantised::up_proj(&self.layer_prefix(layer), expert_id)
    }

    fn expert_ffn_down_key(&self, layer: usize, expert_id: usize) -> Option<String> {
        mxfp4_dequantised::down_proj(&self.layer_prefix(layer), expert_id)
    }
}

#[cfg(test)]
mod tests {
    use super::{DEFAULT_SWIGLU_LIMIT, SWIGLU_ALPHA};
    use crate::config::{ExpertFormat, ExpertGatePolicy, ModelArchitecture};

    /// Minimal `config.json` for `openai/gpt-oss-20b`, matching the real
    /// checkpoint's shape fields.
    fn arch() -> Box<dyn ModelArchitecture> {
        crate::detect_from_json(&serde_json::json!({
            "model_type": "gpt_oss",
            "num_hidden_layers": 24,
            "hidden_size": 2880,
            "intermediate_size": 2880,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "head_dim": 64,
            "vocab_size": 201088,
            "num_local_experts": 32,
            "num_experts_per_tok": 4,
            "rope_theta": 150000.0,
        }))
    }

    #[test]
    fn detects_as_packed_mxfp4_moe() {
        let a = arch();
        assert_eq!(a.family(), "gpt_oss");
        assert!(a.is_moe());
        assert_eq!(a.expert_format(), ExpertFormat::PackedMxfp4);
        assert_eq!(a.num_experts(), 32);
        assert_eq!(a.num_experts_per_token(), 4);
    }

    // ── Attention biases and sinks ──────────────────────────────────────
    // Regression tests for the silent-drop bug: the module header claimed
    // these tensors existed while every accessor returned `None`, so
    // extraction discarded 5 of 11 attention tensors per layer.
    // See `docs/k3-funnel.md` §4.6.1.

    #[test]
    fn declares_all_four_projection_biases() {
        let a = arch();
        assert_eq!(
            a.attn_q_bias_key(3).as_deref(),
            Some("layers.3.self_attn.q_proj.bias")
        );
        assert_eq!(
            a.attn_k_bias_key(3).as_deref(),
            Some("layers.3.self_attn.k_proj.bias")
        );
        assert_eq!(
            a.attn_v_bias_key(3).as_deref(),
            Some("layers.3.self_attn.v_proj.bias")
        );
        assert_eq!(
            a.attn_o_bias_key(3).as_deref(),
            Some("layers.3.self_attn.o_proj.bias")
        );
    }

    #[test]
    fn declares_attention_sinks() {
        assert_eq!(
            arch().attn_sinks_key(7).as_deref(),
            Some("layers.7.self_attn.sinks")
        );
    }

    #[test]
    fn every_attention_tensor_in_the_checkpoint_is_named() {
        // The real checkpoint carries 11 attention-related tensors per
        // layer. All 11 must be reachable through the trait, or
        // extraction silently drops whatever is missing.
        let a = arch();
        let named: Vec<String> = [
            Some(a.attn_q_key(0)),
            Some(a.attn_k_key(0)),
            Some(a.attn_v_key(0)),
            Some(a.attn_o_key(0)),
            a.attn_q_bias_key(0),
            a.attn_k_bias_key(0),
            a.attn_v_bias_key(0),
            a.attn_o_bias_key(0),
            a.attn_sinks_key(0),
            Some(a.input_layernorm_key(0)),
            Some(a.post_attention_layernorm_key(0)),
        ]
        .into_iter()
        .flatten()
        .collect();
        assert_eq!(named.len(), 11, "named tensors: {named:?}");
    }

    #[test]
    fn packed_expert_keys_follow_the_fused_layout() {
        let a = arch();
        assert_eq!(
            a.packed_gate_up_blocks_key(1).as_deref(),
            Some("layers.1.mlp.experts.gate_up_proj_blocks")
        );
        assert_eq!(
            a.packed_gate_up_scales_key(1).as_deref(),
            Some("layers.1.mlp.experts.gate_up_proj_scales")
        );
        assert_eq!(
            a.packed_down_blocks_key(1).as_deref(),
            Some("layers.1.mlp.experts.down_proj_blocks")
        );
        assert_eq!(
            a.packed_down_scales_key(1).as_deref(),
            Some("layers.1.mlp.experts.down_proj_scales")
        );
    }

    #[test]
    fn router_key_is_mlp_router_not_block_sparse_gate() {
        assert_eq!(
            arch().moe_router_key(2).as_deref(),
            Some("layers.2.mlp.router.weight")
        );
    }

    // ── MoE biases and gating ───────────────────────────────────────────
    // Same defect class as the attention tests above, one subsystem over:
    // the checkpoint carries 8 MLP tensors per layer and only 5 were named,
    // so the router bias and both expert biases were silently dropped.
    // See `docs/k3-funnel.md` §4.7.

    #[test]
    fn declares_the_router_bias() {
        assert_eq!(
            arch().moe_router_bias_key(2).as_deref(),
            Some("layers.2.mlp.router.bias")
        );
    }

    #[test]
    fn declares_both_expert_biases() {
        let a = arch();
        assert_eq!(
            a.packed_gate_up_bias_key(1).as_deref(),
            Some("layers.1.mlp.experts.gate_up_proj_bias")
        );
        assert_eq!(
            a.packed_down_bias_key(1).as_deref(),
            Some("layers.1.mlp.experts.down_proj_bias")
        );
    }

    #[test]
    fn every_mlp_tensor_in_the_checkpoint_is_named() {
        // The real checkpoint carries 8 MLP tensors per layer. All 8 must be
        // reachable through the trait, or extraction silently drops whatever
        // is missing — exactly what happened to three of them.
        let a = arch();
        let named: Vec<String> = [
            a.moe_router_key(0),
            a.moe_router_bias_key(0),
            a.packed_gate_up_blocks_key(0),
            a.packed_gate_up_scales_key(0),
            a.packed_gate_up_bias_key(0),
            a.packed_down_blocks_key(0),
            a.packed_down_scales_key(0),
            a.packed_down_bias_key(0),
        ]
        .into_iter()
        .flatten()
        .collect();
        assert_eq!(named.len(), 8, "named tensors: {named:?}");
    }

    /// The MoE pre-experts norm ALIASES the post-attention norm — GPT-OSS
    /// has no separate tensor, and the reference applies
    /// `post_attention_layernorm` to the tensor the router and experts
    /// both read. A drift here (its own spelling, or `None`) puts the
    /// quantised MoE path back on the un-normed residual.
    #[test]
    fn moe_pre_experts_norm_aliases_the_post_attention_norm() {
        let a = arch();
        assert_eq!(
            a.moe_pre_experts_norm_key(3).as_deref(),
            Some(a.post_attention_layernorm_key(3).as_str()),
        );
    }

    #[test]
    fn expert_gate_policy_is_clamped_glu_from_config() {
        // `swiglu_limit` is absent from the fixture config, so this exercises
        // the documented fallback; the released checkpoints ship 7.0 anyway.
        match arch().expert_gate_policy() {
            ExpertGatePolicy::ClampedGlu { limit, alpha } => {
                assert_eq!(limit, DEFAULT_SWIGLU_LIMIT);
                assert_eq!(alpha, SWIGLU_ALPHA);
            }
            other => panic!("expected ClampedGlu, got {other:?}"),
        }
    }

    #[test]
    fn expert_gate_policy_reads_swiglu_limit_from_config() {
        let a = crate::detect_from_json(&serde_json::json!({
            "model_type": "gpt_oss",
            "num_hidden_layers": 24,
            "hidden_size": 2880,
            "intermediate_size": 2880,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "head_dim": 64,
            "num_local_experts": 32,
            "num_experts_per_tok": 4,
            "swiglu_limit": 5.5,
        }));
        match a.expert_gate_policy() {
            ExpertGatePolicy::ClampedGlu { limit, .. } => assert_eq!(limit, 5.5),
            other => panic!("expected ClampedGlu, got {other:?}"),
        }
    }

    /// The trait default is 0, which would size every expert matmul to
    /// nothing. GPT-OSS's experts use the model's own `intermediate_size`.
    #[test]
    fn moe_intermediate_size_falls_back_to_intermediate_size() {
        assert_eq!(arch().moe_intermediate_size(), 2880);
    }

    #[test]
    fn router_type_distinguishes_topk_then_softmax() {
        // Plain `top_k_softmax` means softmax-over-all-then-select, which
        // under-weights the expert branch for this family.
        assert_ne!(arch().moe_router_type(), "top_k_softmax");
        assert_eq!(
            arch().moe_router_kind(),
            crate::MoeRouterKind::TopKThenSoftmax
        );
        assert_eq!(arch().moe_router_type(), "gpt_oss_topk_then_softmax");
    }
}
