//! Gemma 2 architecture.
//!
//! Key differences from Gemma 3:
//! - attn_logit_softcapping (typically 50.0)
//! - final_logit_softcapping (typically 30.0)
//! - No sliding window (uses full attention on all layers)
//! - No local RoPE base (single rope_theta for all layers)
//! - query_pre_attn_scalar may differ from head_dim

use crate::config::{Activation, ModelArchitecture, ModelConfig};

pub struct Gemma2Arch {
    config: ModelConfig,
}

impl Gemma2Arch {
    pub fn from_config(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArchitecture for Gemma2Arch {
    fn family(&self) -> &str {
        "gemma2"
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn attn_q_norm_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}self_attn.q_norm.weight",
            self.layer_prefix(layer)
        ))
    }

    fn attn_k_norm_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}self_attn.k_norm.weight",
            self.layer_prefix(layer)
        ))
    }

    fn norm_weight_offset(&self) -> f32 {
        1.0
    }

    fn qk_norm_weight_offset(&self) -> f32 {
        1.0
    }

    fn activation(&self) -> Activation {
        Activation::GeluTanh
    }

    fn embed_scale(&self) -> f32 {
        (self.config.hidden_size as f32).sqrt()
    }

    fn has_post_norms(&self) -> bool {
        true
    }

    // No sliding window — all layers use full attention with the same rope_theta
}
