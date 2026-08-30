//! OLMoE architecture (Allen AI OLMoE-1B-7B) — Llama attention + QK-norm + MoE.
//!
//! Tensor naming is identical to Qwen3-MoE:
//! - Router at `mlp.gate.weight`
//! - Per-expert `mlp.experts.{E}.{gate,up,down}_proj.weight`
//! - QK norms at `self_attn.{q,k}_norm.weight`
//!
//! It gets its own module rather than aliasing `QwenArch` because of two
//! config differences that would otherwise be silent:
//!
//! 1. **No `moe_intermediate_size` field.** OLMoE stores the per-expert
//!    intermediate width in plain `intermediate_size` (1024 for 1B-7B).
//!    `QwenArch` reads `moe_intermediate_size.unwrap_or(0)`, so aliasing
//!    would report zero-width experts.
//! 2. **`norm_topk_prob: false`.** OLMoE keeps the raw softmax probabilities
//!    of the selected experts; it does not renormalize them to sum to 1.
//!    That is `MoeTopKWeightPolicy::RawSoftmax`, which is what the routing
//!    dispatch already selects for any model not tagged
//!    `gemma4_top_k_softmax`.
//!
//!    The dependency this note recorded turned out to be real: the reference
//!    MoE backend added on 2026-07-30 initially hardcoded GPT-OSS's
//!    normalise-over-selected order, which inflated OLMoE's expert branch and
//!    took bits/char from 0.390 (HF reference) to 2.677. `norm_topk_prob` is
//!    now read from the config into [`ExpertRoutingPolicy`] rather than
//!    inherited from a default. See `docs/k3-funnel.md` §4.7.
//!
//! OLMoE also sets `attention_bias: false` and `num_key_value_heads ==
//! num_attention_heads` (MHA, not GQA), so no bias keys are emitted.

use crate::config::{ModelArchitecture, ModelConfig, QkNormScope};
use crate::tensor_keys::{moe_experts, qk_norm};

pub struct OlmoeArch {
    config: ModelConfig,
}

impl OlmoeArch {
    pub fn from_config(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArchitecture for OlmoeArch {
    fn family(&self) -> &str {
        "olmoe"
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Difference 4, and the one that cost the most: `OlmoeConfig`'s
    /// `rms_norm_eps` class default is **1e-5**, not the 1e-6 that Llama,
    /// Qwen3 and Gemma use — and `allenai/OLMoE-1B-7B-0924-Instruct` ships no
    /// `rms_norm_eps` field at all, so the fallback is what actually runs.
    ///
    /// Serving it at 1e-6 was a systematic error in every norm of every
    /// layer. The Gate-B layer diff measured it directly: final-residual
    /// cosine 0.890 against the HF reference before, 0.991 after.
    fn default_norm_eps(&self) -> f32 {
        crate::defaults::DEFAULT_NORM_EPS_1E5
    }

    // ── MoE ──

    fn is_moe(&self) -> bool {
        self.config.num_experts.unwrap_or(0) > 0
    }

    fn num_experts(&self) -> usize {
        self.config.num_experts.unwrap_or(0)
    }

    fn num_experts_per_token(&self) -> usize {
        self.config
            .num_experts_per_token
            .or(self.config.top_k_experts)
            .unwrap_or(0)
    }

    /// OLMoE has no `moe_intermediate_size`; the per-expert width is the
    /// model's `intermediate_size`. Falling back to 0 here would size every
    /// expert to nothing.
    fn moe_intermediate_size(&self) -> usize {
        self.config
            .moe_intermediate_size
            .unwrap_or(self.config.intermediate_size)
    }

    // `expert_routing_policy` is deliberately *not* overridden here. It used to
    // be, reading `norm_topk_prob` — correct for OLMoE and invisible to every
    // other architecture, which is how `QwenArch` came to inherit the wrong
    // routing order for models shipping `norm_topk_prob: true`. The config read
    // now lives in the trait default so it applies everywhere; see
    // `ModelArchitecture::expert_routing_policy`.

    fn moe_router_key(&self, layer: usize) -> Option<String> {
        if !self.is_moe() {
            return None;
        }
        moe_experts::router(&self.layer_prefix(layer))
    }

    fn expert_ffn_gate_key(&self, layer: usize, expert_id: usize) -> Option<String> {
        if !self.is_moe() {
            return None;
        }
        moe_experts::gate_proj(&self.layer_prefix(layer), expert_id)
    }

    fn expert_ffn_up_key(&self, layer: usize, expert_id: usize) -> Option<String> {
        if !self.is_moe() {
            return None;
        }
        moe_experts::up_proj(&self.layer_prefix(layer), expert_id)
    }

    fn expert_ffn_down_key(&self, layer: usize, expert_id: usize) -> Option<String> {
        if !self.is_moe() {
            return None;
        }
        moe_experts::down_proj(&self.layer_prefix(layer), expert_id)
    }

    // ── QK norms ──

    fn attn_q_norm_key(&self, layer: usize) -> Option<String> {
        qk_norm::q(&self.layer_prefix(layer))
    }

    fn attn_k_norm_key(&self, layer: usize) -> Option<String> {
        qk_norm::k(&self.layer_prefix(layer))
    }

    /// Difference 3, and the one the header's "naming is identical to
    /// Qwen3-MoE" note actively concealed: the *names* match, the *semantics*
    /// do not. `OlmoeAttention` builds `OlmoeRMSNorm(config.hidden_size)` and
    /// applies it to the whole projection before any reshape into heads, where
    /// `Qwen3Attention` builds `Qwen3RMSNorm(head_dim)` and applies it after —
    /// `transformers` marks this in a Qwen3 source comment, "unlike olmo, only
    /// on the head dim!".
    ///
    /// Shapes cannot discriminate it here: OLMoE-1B-7B is MHA, so
    /// `num_heads * head_dim == hidden_size == 2048` and the stored `[2048]`
    /// weight is the same width under either reading. Only the reduction
    /// differs, and treating it as per-head normalises every head to a common
    /// magnitude — which is why it cost 1.9 bits/char against a 0.39 reference
    /// rather than a rounding-sized amount.
    fn qk_norm_scope(&self) -> QkNormScope {
        QkNormScope::FullProjection
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// OLMoE-1B-7B's real routing shape: 64 experts, 8 per token, and the
    /// per-expert width in plain `intermediate_size` (difference 1 in the
    /// module doc).
    fn olmoe_arch() -> Box<dyn ModelArchitecture> {
        crate::detect_from_json(&serde_json::json!({
            "model_type": "olmoe",
            "hidden_size": 2048,
            "intermediate_size": 1024,
            "num_hidden_layers": 16,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "num_experts": 64,
            "num_experts_per_tok": 8,
            "norm_topk_prob": false,
        }))
    }

    #[test]
    fn detects_as_olmoe_moe() {
        let a = olmoe_arch();
        assert_eq!(a.family(), "olmoe");
        assert!(a.is_moe());
        assert_eq!(a.num_experts(), 64);
        assert_eq!(a.num_experts_per_token(), 8);
    }

    /// Difference 1 vs Qwen3-MoE: no `moe_intermediate_size` field, so the
    /// expert width must come from `intermediate_size`. A
    /// `moe_intermediate_size.unwrap_or(0)` read reports zero-width experts.
    #[test]
    fn expert_width_falls_back_to_intermediate_size() {
        assert_eq!(olmoe_arch().moe_intermediate_size(), 1024);
    }

    /// Difference 2 vs GPT-OSS: `norm_topk_prob: false` keeps raw softmax
    /// probabilities. Inheriting normalise-over-selected rescaled the whole
    /// expert branch (bits/char 0.390 -> 2.677; module doc).
    #[test]
    fn routing_policy_keeps_raw_softmax() {
        assert_eq!(
            olmoe_arch().expert_routing_policy(),
            crate::config::ExpertRoutingPolicy::SoftmaxThenSelect
        );
    }

    #[test]
    fn moe_and_qk_norm_keys_use_qwen3_naming() {
        let a = olmoe_arch();
        let p = a.layer_prefix(2);
        assert_eq!(a.moe_router_key(2), Some(format!("{p}mlp.gate.weight")));
        assert_eq!(
            a.expert_ffn_gate_key(2, 5),
            Some(format!("{p}mlp.experts.5.gate_proj.weight"))
        );
        assert_eq!(
            a.expert_ffn_up_key(2, 5),
            Some(format!("{p}mlp.experts.5.up_proj.weight"))
        );
        assert_eq!(
            a.expert_ffn_down_key(2, 5),
            Some(format!("{p}mlp.experts.5.down_proj.weight"))
        );
        assert_eq!(
            a.attn_q_norm_key(2),
            Some(format!("{p}self_attn.q_norm.weight"))
        );
        assert_eq!(
            a.attn_k_norm_key(2),
            Some(format!("{p}self_attn.k_norm.weight"))
        );
    }

    /// Difference 3: same key names as Qwen3, different reduction.
    ///
    /// Asserted against the *other* family rather than against a bare literal,
    /// because the whole failure mode was inheriting Qwen3's convention along
    /// with its naming. A test that only pinned `FullProjection` would still
    /// pass if `PerHead` stopped being the default for everyone else.
    #[test]
    fn qk_norm_reduces_over_the_whole_projection_unlike_qwen3() {
        assert_eq!(olmoe_arch().qk_norm_scope(), QkNormScope::FullProjection);

        let qwen = crate::detect_from_json(&serde_json::json!({
            "model_type": "qwen3_moe",
            "hidden_size": 2048,
            "intermediate_size": 1024,
            "num_hidden_layers": 2,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "num_experts": 64,
            "num_experts_per_tok": 8,
        }));
        assert_eq!(
            qwen.qk_norm_scope(),
            QkNormScope::PerHead,
            "Qwen3 must stay per-head — `transformers` marks the split in a \
             source comment (\"unlike olmo, only on the head dim!\"), and both \
             families store `self_attn.q_norm.weight`"
        );
    }

    /// The shapes cannot discriminate the scope on this model, which is why
    /// the declaration has to be explicit. OLMoE-1B-7B is MHA, so
    /// `num_heads * head_dim == hidden_size`, and the real checkpoint ships a
    /// `[2048]` q_norm — exactly as wide as `hidden_size`, and also exactly
    /// `16 * 128`. Anything inferring scope from element count alone reads
    /// this checkpoint as either convention.
    #[test]
    fn head_geometry_makes_the_two_scopes_indistinguishable_by_width() {
        let a = olmoe_arch();
        let c = a.config();
        assert_eq!(
            c.num_q_heads * c.head_dim,
            c.hidden_size,
            "if this ever stops holding, width alone becomes a usable signal \
             and the cross-check in `rms_norm_qk_eps` gains teeth here"
        );
    }

    /// MHA, no bias (module doc): the arch must not invent bias keys.
    #[test]
    fn no_attention_bias_keys() {
        let a = olmoe_arch();
        assert_eq!(a.attn_q_bias_key(0), None);
        assert_eq!(a.attn_o_bias_key(0), None);
    }

    /// A config without experts is not MoE, and every MoE key vanishes with
    /// it — the guard arm of each key method.
    #[test]
    fn without_experts_every_moe_key_is_absent() {
        let a = crate::detect_from_json(&serde_json::json!({
            "model_type": "olmoe",
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
        }));
        assert!(!a.is_moe());
        assert_eq!(a.num_experts_per_token(), 0);
        assert_eq!(a.moe_router_key(0), None);
        assert_eq!(a.expert_ffn_gate_key(0, 0), None);
        assert_eq!(a.expert_ffn_up_key(0, 0), None);
        assert_eq!(a.expert_ffn_down_key(0, 0), None);
    }
}
