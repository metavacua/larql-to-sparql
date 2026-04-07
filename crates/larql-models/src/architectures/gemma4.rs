//! Gemma 4 architecture — Google's multimodal model family (2025).
//!
//! Key differences from Gemma 3:
//! - Dual head_dim: sliding layers use head_dim (256), global layers use global_head_dim (512)
//! - Fewer KV heads on global layers (num_global_key_value_heads)
//! - Partial rotary: global layers apply RoPE to only 25% of head dims
//! - K=V sharing: later global layers have no v_proj (value = key)
//! - Per-layer scalar multiplier (layer_scalar)
//! - QK-norm (inherited from Gemma 3)
//! - 4 norms per layer (inherited from Gemma 3)
//! - Logit softcapping (inherited from Gemma 2)
//!
//! Layer types are determined from:
//! 1. Explicit `layer_types` array in config.json (["sliding_attention", "full_attention", ...])
//! 2. `sliding_window_pattern` field (every Nth layer is full)
//! 3. Default pattern of 6 (every 6th layer is full)

use crate::config::{Activation, ModelArchitecture, ModelConfig};

pub struct Gemma4Arch {
    config: ModelConfig,
    /// Precomputed: which layer indices are full (global) attention.
    global_layers: Vec<bool>,
}

impl Gemma4Arch {
    pub fn from_config(config: ModelConfig) -> Self {
        let num_layers = config.num_layers;

        // Determine global layers from explicit layer_types or pattern
        let global_layers = if let Some(ref types) = config.layer_types {
            types.iter()
                .map(|t| t == "full_attention")
                .collect()
        } else {
            let pattern = config.sliding_window_pattern.unwrap_or(6);
            (0..num_layers)
                .map(|layer| (layer + 1) % pattern == 0)
                .collect()
        };

        Self {
            config,
            global_layers,
        }
    }

    fn is_global_layer(&self, layer: usize) -> bool {
        self.global_layers.get(layer).copied().unwrap_or(false)
    }
}

impl ModelArchitecture for Gemma4Arch {
    fn family(&self) -> &str {
        "gemma4"
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Gemma 4 weights use `model.language_model.` prefix (multimodal wrapper).
    fn key_prefixes_to_strip(&self) -> &[&str] {
        &["model.language_model.model.", "model.language_model.", "language_model.model.", "model."]
    }

    // ── Per-layer attention geometry ──

    fn head_dim_for_layer(&self, layer: usize) -> usize {
        if self.is_global_layer(layer) {
            self.config.global_head_dim.unwrap_or(self.config.head_dim)
        } else {
            self.config.head_dim
        }
    }

    fn num_kv_heads_for_layer(&self, layer: usize) -> usize {
        if self.is_global_layer(layer) {
            self.config.num_global_kv_heads.unwrap_or(self.config.num_kv_heads)
        } else {
            self.config.num_kv_heads
        }
    }

    fn num_q_heads_for_layer(&self, _layer: usize) -> usize {
        // Gemma 4 keeps num_q_heads constant across all layers.
        // At global layers, each head uses global_head_dim instead of head_dim,
        // so Q projection output is larger (num_q * global_head_dim).
        self.config.num_q_heads
    }

    fn rotary_fraction_for_layer(&self, layer: usize) -> f64 {
        if self.is_global_layer(layer) {
            self.config.partial_rotary_factor.unwrap_or(1.0)
        } else {
            1.0
        }
    }

    fn v_shares_k(&self, _layer: usize) -> bool {
        // K=V sharing is enabled at the config level (attention_k_eq_v=true).
        // The forward pass detects it at runtime by checking for missing v_proj.
        false
    }

    fn has_v_norm(&self) -> bool {
        true
    }

    // Gemma 4 uses QK-norm which already normalizes dot products.
    // No additional 1/sqrt(head_dim) scaling is applied (scaling = 1.0).
    fn attention_scale(&self) -> f64 {
        1.0
    }

    fn attention_scale_for_layer(&self, _layer: usize) -> f64 {
        1.0
    }

    fn layer_scalar_key(&self, layer: usize) -> Option<String> {
        Some(format!("{}layer_scalar", self.layer_prefix(layer)))
    }

    // ── QK norm (inherited from Gemma 3) ──

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

    // ── Gemma-family behavior ──

    // Gemma 4 stores norm weights as the full multiplier (no +1 offset).
    // Unlike Gemma 2/3 which used 1+weight, Gemma 4's Gemma4RMSNorm applies weight directly.
    fn norm_weight_offset(&self) -> f32 {
        0.0
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

    fn is_sliding_window_layer(&self, layer: usize) -> bool {
        !self.is_global_layer(layer)
    }

    fn rope_base_for_layer(&self, layer: usize) -> f64 {
        if self.is_sliding_window_layer(layer) {
            self.config.rope_local_base.unwrap_or(10_000.0)
        } else {
            self.config.rope_base
        }
    }
}
