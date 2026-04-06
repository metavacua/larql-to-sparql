//! Qwen architecture (Qwen 2, 2.5, 3, MoE variants).
//!
//! Mostly Llama-compatible but Qwen2/2.5 have attention Q/K/V bias terms.

use crate::config::{ModelArchitecture, ModelConfig};

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

    // Qwen3 has QK norms (no +1 offset — standard RMSNorm).
    // Returning keys for models that don't have them is harmless
    // (the forward pass checks if the vector exists).

    fn attn_q_norm_key(&self, layer: usize) -> Option<String> {
        Some(format!("{}self_attn.q_norm.weight", self.layer_prefix(layer)))
    }

    fn attn_k_norm_key(&self, layer: usize) -> Option<String> {
        Some(format!("{}self_attn.k_norm.weight", self.layer_prefix(layer)))
    }

    // Qwen2/2.5 have attention bias on Q, K, V projections.
    // Qwen3 does not — returning keys for absent tensors is harmless.

    fn attn_q_bias_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}self_attn.q_proj.bias",
            self.layer_prefix(layer)
        ))
    }

    fn attn_k_bias_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}self_attn.k_proj.bias",
            self.layer_prefix(layer)
        ))
    }

    fn attn_v_bias_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}self_attn.v_proj.bias",
            self.layer_prefix(layer)
        ))
    }
}
