//! StarCoder 2 architecture.
//!
//! Key differences from standard Llama:
//! - Non-gated FFN: activation(x @ c_fc.T + bias) @ c_proj.T + bias
//! - Uses c_fc/c_proj naming instead of gate_proj/up_proj/down_proj
//! - Has biases on attention projections, FFN, and layer norms
//! - Uses GQA with sliding window

use crate::config::{Activation, FfnType, NormType, ModelArchitecture, ModelConfig};

pub struct StarCoder2Arch {
    config: ModelConfig,
}

impl StarCoder2Arch {
    pub fn from_config(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArchitecture for StarCoder2Arch {
    fn family(&self) -> &str {
        "starcoder2"
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn norm_type(&self) -> NormType {
        NormType::LayerNorm
    }

    fn activation(&self) -> Activation {
        Activation::GeluTanh
    }

    fn ffn_type(&self) -> FfnType {
        FfnType::Standard
    }

    // StarCoder2 uses c_fc/c_proj naming
    fn ffn_up_key(&self, layer: usize) -> String {
        format!("{}mlp.c_fc.weight", self.layer_prefix(layer))
    }

    fn ffn_down_key(&self, layer: usize) -> String {
        format!("{}mlp.c_proj.weight", self.layer_prefix(layer))
    }

    // FFN biases
    fn ffn_up_bias_key(&self, layer: usize) -> Option<String> {
        Some(format!("{}mlp.c_fc.bias", self.layer_prefix(layer)))
    }

    fn ffn_down_bias_key(&self, layer: usize) -> Option<String> {
        Some(format!("{}mlp.c_proj.bias", self.layer_prefix(layer)))
    }

    // Attention biases (including O proj)
    fn attn_o_bias_key(&self, layer: usize) -> Option<String> {
        Some(format!("{}self_attn.o_proj.bias", self.layer_prefix(layer)))
    }

    fn attn_q_bias_key(&self, layer: usize) -> Option<String> {
        Some(format!("{}self_attn.q_proj.bias", self.layer_prefix(layer)))
    }

    fn attn_k_bias_key(&self, layer: usize) -> Option<String> {
        Some(format!("{}self_attn.k_proj.bias", self.layer_prefix(layer)))
    }

    fn attn_v_bias_key(&self, layer: usize) -> Option<String> {
        Some(format!("{}self_attn.v_proj.bias", self.layer_prefix(layer)))
    }
}
