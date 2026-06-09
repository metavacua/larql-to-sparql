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

    /// Per-expert intermediate dimension. Mixtral has no separate
    /// `moe_intermediate_size` field — each expert's w1/w3 are
    /// `[intermediate_size, hidden]`, so the model-wide
    /// `intermediate_size` IS the expert hidden dim. Honor an explicit
    /// `moe_intermediate_size` override if a config provides one.
    ///
    /// Without this override the trait default (0) flows into the Q4_K
    /// MoE writer, which records `moe_inter = 0` in the per-layer
    /// weights header (silently corrupting the loader's view) and, for
    /// non-256-aligned dims, panics in `quantize_q4_k`.
    fn moe_intermediate_size(&self) -> usize {
        self.config
            .moe_intermediate_size
            .unwrap_or(self.config.intermediate_size)
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
