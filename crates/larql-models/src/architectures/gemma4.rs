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

use crate::config::{Activation, ModelArchitecture, ModelConfig};

pub struct Gemma4Arch {
    config: ModelConfig,
    /// Precomputed: which layer indices are full (global) attention.
    global_layers: Vec<bool>,
}

impl Gemma4Arch {
    pub fn from_config(config: ModelConfig) -> Self {
        let pattern = config.sliding_window_pattern.unwrap_or(6);
        let num_layers = config.num_layers;
        let global_layers = (0..num_layers)
            .map(|layer| (layer + 1) % pattern == 0)
            .collect();

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

    fn num_q_heads_for_layer(&self, layer: usize) -> usize {
        if self.is_global_layer(layer) {
            // Q projection output dim is constant, but head_dim changes
            // num_q_heads = q_proj_dim / global_head_dim
            let q_proj_dim = self.config.num_q_heads * self.config.head_dim;
            let global_hd = self.config.global_head_dim.unwrap_or(self.config.head_dim);
            q_proj_dim / global_hd
        } else {
            self.config.num_q_heads
        }
    }

    fn rotary_fraction_for_layer(&self, layer: usize) -> f64 {
        if self.is_global_layer(layer) {
            self.config.partial_rotary_factor.unwrap_or(1.0)
        } else {
            1.0
        }
    }

    fn v_shares_k(&self, layer: usize) -> bool {
        // Gemma 4 uses K=V on later global layers. We detect this at runtime
        // by checking whether the v_proj tensor exists. The architecture just
        // signals that this *can* happen; the forward pass checks the weights.
        // Always return false here — let the forward pass do the lookup.
        let _ = layer;
        false
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

    fn norm_weight_offset(&self) -> f32 {
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
