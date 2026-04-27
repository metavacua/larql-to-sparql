//! GPT-OSS architecture — OpenAI's MoE model with MXFP4 packed experts.
//!
//! Key differences from standard MoE (Mixtral):
//! - Expert weights are packed as MXFP4 (e8m0 scales + 4-bit values)
//! - Gate and up projections are fused: `gate_up_proj_blocks` (first half = gate)
//! - All experts packed in one tensor per layer, not per-expert files
//! - Router at `mlp.router.weight` (not `block_sparse_moe.gate`)
//! - Attention has biases, sinks, and uses GQA
//! - YaRN RoPE scaling

use crate::config::{ExpertFormat, ModelArchitecture, ModelConfig};

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

    fn num_experts(&self) -> usize {
        self.config.num_experts.unwrap_or(128)
    }

    fn num_experts_per_token(&self) -> usize {
        self.config.num_experts_per_token.unwrap_or(4)
    }

    fn moe_router_key(&self, layer: usize) -> Option<String> {
        Some(format!("{}mlp.router.weight", self.layer_prefix(layer)))
    }

    // ── Packed MXFP4 expert keys ──

    fn packed_gate_up_blocks_key(&self, layer: usize) -> Option<String> {
        Some(format!("{}mlp.experts.gate_up_proj_blocks", self.layer_prefix(layer)))
    }

    fn packed_gate_up_scales_key(&self, layer: usize) -> Option<String> {
        Some(format!("{}mlp.experts.gate_up_proj_scales", self.layer_prefix(layer)))
    }

    fn packed_down_blocks_key(&self, layer: usize) -> Option<String> {
        Some(format!("{}mlp.experts.down_proj_blocks", self.layer_prefix(layer)))
    }

    fn packed_down_scales_key(&self, layer: usize) -> Option<String> {
        Some(format!("{}mlp.experts.down_proj_scales", self.layer_prefix(layer)))
    }

    // Per-expert keys are not available for GPT-OSS (packed format).
    // Callers should check expert_format() and use packed_* keys instead.
}
