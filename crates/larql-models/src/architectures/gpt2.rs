//! GPT-2 architecture.
//!
//! Key differences from standard Llama:
//! - Standard LayerNorm (with bias) instead of RMSNorm
//! - Non-gated FFN: activation(x @ c_fc.T + bias) @ c_proj.T + bias, GELU activation
//! - Conv1D-style fused QKV (`attn_qkv.weight`) instead of separate Q/K/V
//! - Learned position embeddings (`position_embd.weight`) instead of RoPE
//! - LM head tied to input embedding
//!
//! Tensor key naming after the GGUF→HF normalization in `loading/gguf.rs`
//! matches the trait defaults (`embed_tokens.weight`, `mlp.up_proj.weight`,
//! `mlp.down_proj.weight`, …), so this arch only overrides behavior flags.

use crate::config::{Activation, FfnType, ModelArchitecture, ModelConfig, NormType};

pub struct Gpt2Arch {
    config: ModelConfig,
}

impl Gpt2Arch {
    pub fn from_config(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArchitecture for Gpt2Arch {
    fn family(&self) -> &str {
        "gpt2"
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

    /// GPT-2 packs Q, K, V into a single Conv1D `c_attn` projection. The
    /// GGUF→HF normaliser maps `attn_qkv.` → `self_attn.qkv_proj.`; the
    /// loader's `split_fused_qkv` pass then materialises the per-projection
    /// q/k/v tensors at the trait's standard `attn_q_key`/`attn_k_key`/
    /// `attn_v_key` so downstream code stays family-agnostic.
    fn fused_qkv_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}self_attn.qkv_proj.weight",
            self.layer_prefix(layer)
        ))
    }

    fn fused_qkv_bias_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}self_attn.qkv_proj.bias",
            self.layer_prefix(layer)
        ))
    }

    /// GPT-2 has bias on every projection (Conv1D layers carry bias).
    fn attn_q_bias_key(&self, layer: usize) -> Option<String> {
        Some(format!("{}self_attn.q_proj.bias", self.layer_prefix(layer)))
    }

    fn attn_k_bias_key(&self, layer: usize) -> Option<String> {
        Some(format!("{}self_attn.k_proj.bias", self.layer_prefix(layer)))
    }

    fn attn_v_bias_key(&self, layer: usize) -> Option<String> {
        Some(format!("{}self_attn.v_proj.bias", self.layer_prefix(layer)))
    }

    fn attn_o_bias_key(&self, layer: usize) -> Option<String> {
        Some(format!("{}self_attn.o_proj.bias", self.layer_prefix(layer)))
    }

    fn ffn_up_bias_key(&self, layer: usize) -> Option<String> {
        Some(format!("{}mlp.up_proj.bias", self.layer_prefix(layer)))
    }

    fn ffn_down_bias_key(&self, layer: usize) -> Option<String> {
        Some(format!("{}mlp.down_proj.bias", self.layer_prefix(layer)))
    }

    /// GPT-2 uses learned absolute position embeddings (`wpe`) added to the
    /// token embedding at the input — no rotary positional signal.
    fn position_embed_key(&self) -> Option<&str> {
        Some("wpe.weight")
    }
}
