//! TinyModel architecture.
//!
//! Research-scale decoder-only transformer used as the reference target
//! for the LARQL compile/walk work. Same shape family as Llama (RMSNorm,
//! RoPE, GQA, gated SwiGLU FFN, tied embeddings, 2 norms per layer) but
//! with Gemma-style `sqrt(hidden_size)` embedding scaling and a flatter
//! native tensor key layout (no `model.` prefix, `attn.*`/`ffn.*`
//! instead of `self_attn.*`/`mlp.*`).
//!
//! Versions: v11, v11a, v12, … all share this architecture. Weights
//! live at `<tiny-model>/model/<version>/artifacts/`.

use crate::config::{ModelArchitecture, ModelConfig};

pub struct TinyModelArch {
    config: ModelConfig,
}

impl TinyModelArch {
    pub fn from_config(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArchitecture for TinyModelArch {
    fn family(&self) -> &str {
        "tinymodel"
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    // ── Embedding scaling (Gemma-style) ──
    fn embed_scale(&self) -> f32 {
        (self.config.hidden_size as f32).sqrt()
    }

    // ── Native key layout (no `model.` prefix, flat attn/ffn) ──
    fn key_prefixes_to_strip(&self) -> &[&str] {
        &[]
    }

    fn embed_key(&self) -> &str {
        "embed.weight"
    }

    fn final_norm_key(&self) -> &str {
        "norm.weight"
    }

    fn attn_q_key(&self, layer: usize) -> String {
        format!("{}attn.q_proj.weight", self.layer_prefix(layer))
    }

    fn attn_k_key(&self, layer: usize) -> String {
        format!("{}attn.k_proj.weight", self.layer_prefix(layer))
    }

    fn attn_v_key(&self, layer: usize) -> String {
        format!("{}attn.v_proj.weight", self.layer_prefix(layer))
    }

    fn attn_o_key(&self, layer: usize) -> String {
        format!("{}attn.o_proj.weight", self.layer_prefix(layer))
    }

    fn ffn_gate_key(&self, layer: usize) -> String {
        format!("{}ffn.gate.weight", self.layer_prefix(layer))
    }

    fn ffn_up_key(&self, layer: usize) -> String {
        format!("{}ffn.up.weight", self.layer_prefix(layer))
    }

    fn ffn_down_key(&self, layer: usize) -> String {
        format!("{}ffn.down.weight", self.layer_prefix(layer))
    }

    fn input_layernorm_key(&self, layer: usize) -> String {
        format!("{}attn_norm.weight", self.layer_prefix(layer))
    }

    fn post_attention_layernorm_key(&self, layer: usize) -> String {
        format!("{}ffn_norm.weight", self.layer_prefix(layer))
    }
}
