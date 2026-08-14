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

    /// `MixtralSparseMoeBlock` softmaxes over all experts, selects top-k, then
    /// divides by the selected sum:
    ///
    /// ```text
    /// router_logits = softmax(router_logits, dim=-1)
    /// router_top_value, router_indices = topk(router_logits, top_k, dim=-1)
    /// router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
    /// ```
    ///
    /// That renormalisation is algebraically softmax over the selected logits
    /// — `softmax(x_i) / Σ_{j∈S} softmax(x_j) == exp(x_i) / Σ_{j∈S} exp(x_j)`
    /// — so it is `TopKThenSoftmax`, not the inherited softmax-then-select.
    /// Same defect and same fix as GraniteMoE, which measured 49.762%
    /// bits/char against its HF reference with the inherited default.
    ///
    /// Declared from the reference source; unlike Granite this is **not**
    /// backed by a Shannon measurement here, because the smallest Mixtral is
    /// 8x7B and nothing of that family is available to score against.
    fn moe_router_kind(&self) -> crate::config::MoeRouterKind {
        crate::config::MoeRouterKind::TopKThenSoftmax
    }

    /// The reference tier's spelling of the same fact. Mixtral ships no
    /// `norm_topk_prob` — it always renormalises — so the config-reading
    /// default answers `SoftmaxThenSelect` for a model that does not.
    fn expert_routing_policy(&self) -> crate::config::ExpertRoutingPolicy {
        crate::config::ExpertRoutingPolicy::NormalisedOverSelected
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
