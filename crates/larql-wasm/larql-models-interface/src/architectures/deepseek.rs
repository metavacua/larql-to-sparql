//! DeepSeek v2/v3 architecture — MoE + MLA.
//!
//! Key differences from standard Llama:
//! - MoE: router selects top-K of N routed experts per token, plus shared experts
//! - MLA: compressed KV via low-rank projections (kv_a_proj → kv_b_proj)
//! - YaRN RoPE scaling for extended context
//! - Tensor key pattern: experts under mlp.experts.{id}, shared under mlp.shared_experts

use crate::config::{ModelArchitecture, ModelConfig};

pub struct DeepSeekArch {
    config: ModelConfig,
}

impl DeepSeekArch {
    pub fn from_config(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArchitecture for DeepSeekArch {
    fn family(&self) -> &str {
        "deepseek"
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    // ── MoE ──

    fn is_moe(&self) -> bool {
        self.config.num_experts.unwrap_or(0) > 0
    }

    fn num_experts(&self) -> usize {
        self.config.num_experts.unwrap_or(64)
    }

    fn num_experts_per_token(&self) -> usize {
        self.config.num_experts_per_token.unwrap_or(6)
    }

    fn num_shared_experts(&self) -> usize {
        self.config.num_shared_experts.unwrap_or(2)
    }

    fn moe_router_key(&self, layer: usize) -> Option<String> {
        Some(format!("{}mlp.gate.weight", self.layer_prefix(layer)))
    }

    fn expert_ffn_gate_key(&self, layer: usize, expert_id: usize) -> Option<String> {
        Some(format!(
            "{}mlp.experts.{expert_id}.gate_proj.weight",
            self.layer_prefix(layer)
        ))
    }

    fn expert_ffn_up_key(&self, layer: usize, expert_id: usize) -> Option<String> {
        Some(format!(
            "{}mlp.experts.{expert_id}.up_proj.weight",
            self.layer_prefix(layer)
        ))
    }

    fn expert_ffn_down_key(&self, layer: usize, expert_id: usize) -> Option<String> {
        Some(format!(
            "{}mlp.experts.{expert_id}.down_proj.weight",
            self.layer_prefix(layer)
        ))
    }

    fn shared_expert_gate_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}mlp.shared_experts.gate_proj.weight",
            self.layer_prefix(layer)
        ))
    }

    fn shared_expert_up_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}mlp.shared_experts.up_proj.weight",
            self.layer_prefix(layer)
        ))
    }

    fn shared_expert_down_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}mlp.shared_experts.down_proj.weight",
            self.layer_prefix(layer)
        ))
    }

    // ── MLA ──

    fn uses_mla(&self) -> bool {
        self.config.kv_lora_rank.is_some()
    }

    fn kv_lora_rank(&self) -> usize {
        self.config.kv_lora_rank.unwrap_or(512)
    }

    fn q_lora_rank(&self) -> usize {
        self.config.q_lora_rank.unwrap_or(1536)
    }

    fn mla_kv_a_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}self_attn.kv_a_proj_with_mqa.weight",
            self.layer_prefix(layer)
        ))
    }

    fn mla_kv_b_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}self_attn.kv_b_proj.weight",
            self.layer_prefix(layer)
        ))
    }

    fn mla_q_a_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}self_attn.q_a_proj.weight",
            self.layer_prefix(layer)
        ))
    }

    fn mla_q_b_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}self_attn.q_b_proj.weight",
            self.layer_prefix(layer)
        ))
    }

    // RoPE scaling: uses trait defaults which read from config.rope_scaling
}
