//! Qwen architecture (Qwen 2, 2.5, 3, MoE variants).
//!
//! Mostly Llama-compatible but with these differences:
//! - Qwen2/2.5: attention Q/K/V bias terms
//! - Qwen3: QK norms (no bias), optional MoE FFN
//! - Qwen3 MoE: router at `mlp.gate.weight`, per-expert `mlp.experts.{E}.{gate,up,down}_proj.weight`

use crate::config::{ModelArchitecture, ModelConfig};
#[cfg(target_arch = "wasm32")]
use crate::prelude::*;
use crate::tensor_keys::{attn_bias, moe_experts, qk_norm};

pub struct QwenArch {
    config: ModelConfig,
}

impl QwenArch {
    pub fn from_config(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArchitecture for QwenArch {
    fn family(&self) -> &str {
        &self.config.model_type
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    // ── MoE (Qwen3-MoE, Qwen2-MoE) ──

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

    fn moe_intermediate_size(&self) -> usize {
        self.config.moe_intermediate_size.unwrap_or(0)
    }

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

    // ── QK norms (Qwen3) ──
    // Returning keys for models that don't have them is harmless —
    // the forward pass checks if the vector exists before using it.

    fn attn_q_norm_key(&self, layer: usize) -> Option<String> {
        qk_norm::q(&self.layer_prefix(layer))
    }

    fn attn_k_norm_key(&self, layer: usize) -> Option<String> {
        qk_norm::k(&self.layer_prefix(layer))
    }

    // ── Attention bias (Qwen2/2.5 only; absent in Qwen3) ──
    // Returning keys for absent tensors is harmless.

    fn attn_q_bias_key(&self, layer: usize) -> Option<String> {
        attn_bias::q(&self.layer_prefix(layer))
    }

    fn attn_k_bias_key(&self, layer: usize) -> Option<String> {
        attn_bias::k(&self.layer_prefix(layer))
    }

    fn attn_v_bias_key(&self, layer: usize) -> Option<String> {
        attn_bias::v(&self.layer_prefix(layer))
    }
}
