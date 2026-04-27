//! Mixtral architecture — Llama attention + block-sparse MoE FFN.
//!
//! Key differences from standard Llama:
//! - FFN replaced by MoE: router selects top-K of N experts per token
//! - Expert weights use w1 (gate), w2 (down), w3 (up) naming
//! - Router and experts under `block_sparse_moe` prefix
//! - Attention is identical to Llama

use crate::config::{ModelArchitecture, ModelConfig};

pub struct MixtralArch {
    config: ModelConfig,
}

impl MixtralArch {
    pub fn from_config(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArchitecture for MixtralArch {
    fn family(&self) -> &str {
        "mixtral"
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    // ── MoE ──

    fn is_moe(&self) -> bool {
        true
    }

    fn num_experts(&self) -> usize {
        self.config.num_experts.unwrap_or(8)
    }

    fn num_experts_per_token(&self) -> usize {
        self.config.num_experts_per_token.unwrap_or(2)
    }

    fn moe_router_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}block_sparse_moe.gate.weight",
            self.layer_prefix(layer)
        ))
    }

    // Mixtral uses w1/w2/w3 naming:
    //   w1 = gate_proj, w2 = down_proj, w3 = up_proj

    fn expert_ffn_gate_key(&self, layer: usize, expert_id: usize) -> Option<String> {
        Some(format!(
            "{}block_sparse_moe.experts.{expert_id}.w1.weight",
            self.layer_prefix(layer)
        ))
    }

    fn expert_ffn_up_key(&self, layer: usize, expert_id: usize) -> Option<String> {
        Some(format!(
            "{}block_sparse_moe.experts.{expert_id}.w3.weight",
            self.layer_prefix(layer)
        ))
    }

    fn expert_ffn_down_key(&self, layer: usize, expert_id: usize) -> Option<String> {
        Some(format!(
            "{}block_sparse_moe.experts.{expert_id}.w2.weight",
            self.layer_prefix(layer)
        ))
    }
}
