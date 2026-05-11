//! DeepSeek-V4 architecture — MoE + MLA + MXFP4 expert weights + HCA attention.
//!
//! Distinct from DeepSeek-V3 (`deepseek.rs`) in several ways:
//!
//! - **No `model.` prefix.** V4 stores tensors as `embed.weight`,
//!   `layers.X.attn.*`, `layers.X.ffn.*`. V3 used `model.embed_tokens.weight`,
//!   `model.layers.X.self_attn.*`, `model.layers.X.mlp.*`.
//! - **`ffn` not `mlp`** for the feed-forward block.
//! - **`w1`/`w2`/`w3` for expert weights** (LLaMA-1 / OG SwiGLU naming) instead
//!   of V3's `gate_proj`/`down_proj`/`up_proj`.
//! - **MXFP4 expert weights** stored as I8 packed nibbles + F8_E8M0 per-32
//!   element scales. The cross-tensor unpacker in
//!   `crates/larql-models/src/loading/safetensors.rs::dequantize_per_expert_mxfp4`
//!   handles the I8 + F8_E8M0 pairing automatically based on the tensor naming.
//! - **HCA / CSA attention** new in V4 — for now this arch impl exposes the
//!   MLA shape (V4 retains MLA as well) and lets the loader skip HCA-specific
//!   tensors via `key_prefixes_to_strip`'s default behavior (no special prefix).
//!
//! Currently scoped to **browse-tier extraction** — gate vectors + embeddings
//! + down_meta. Inference (HCA forward pass) is out of scope for this impl.

use crate::config::{ModelArchitecture, ModelConfig};

pub struct DeepSeekV4Arch {
    config: ModelConfig,
}

impl DeepSeekV4Arch {
    pub fn from_config(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArchitecture for DeepSeekV4Arch {
    fn family(&self) -> &str {
        "deepseek_v4"
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    // ── Tensor key conventions (V4 has no `model.` prefix; uses `attn` / `ffn`) ──

    fn key_prefixes_to_strip(&self) -> &[&str] {
        // No `model.` wrapper in V4 safetensors.
        &[]
    }

    fn embed_key(&self) -> &str {
        "embed.weight"
    }

    fn final_norm_key(&self) -> &str {
        "norm.weight"
    }

    // Attention: `layers.X.attn.*` (V3 used `self_attn.*`)
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

    // Layer norms: V4 names them `attn_norm` / `ffn_norm` (not
    // `input_layernorm` / `post_attention_layernorm`).
    fn input_layernorm_key(&self, layer: usize) -> String {
        format!("{}attn_norm.weight", self.layer_prefix(layer))
    }
    fn post_attention_layernorm_key(&self, layer: usize) -> String {
        format!("{}ffn_norm.weight", self.layer_prefix(layer))
    }
    fn pre_feedforward_layernorm_key(&self, _layer: usize) -> Option<String> {
        None
    }
    fn post_feedforward_layernorm_key(&self, _layer: usize) -> Option<String> {
        None
    }

    // Dense FFN keys (used for non-MoE layers, if any). V4 uses `ffn.w1/w2/w3`.
    fn ffn_gate_key(&self, layer: usize) -> String {
        format!("{}ffn.w1.weight", self.layer_prefix(layer))
    }
    fn ffn_up_key(&self, layer: usize) -> String {
        format!("{}ffn.w3.weight", self.layer_prefix(layer))
    }
    fn ffn_down_key(&self, layer: usize) -> String {
        format!("{}ffn.w2.weight", self.layer_prefix(layer))
    }

    // ── MoE ──

    fn is_moe(&self) -> bool {
        self.config.num_experts.unwrap_or(0) > 0
    }

    fn num_experts(&self) -> usize {
        self.config.num_experts.unwrap_or(256)
    }

    fn num_experts_per_token(&self) -> usize {
        self.config.num_experts_per_token.unwrap_or(6)
    }

    fn num_shared_experts(&self) -> usize {
        self.config.num_shared_experts.unwrap_or(1)
    }

    fn moe_router_key(&self, layer: usize) -> Option<String> {
        Some(format!("{}ffn.gate.weight", self.layer_prefix(layer)))
    }

    fn expert_ffn_gate_key(&self, layer: usize, expert_id: usize) -> Option<String> {
        Some(format!(
            "{}ffn.experts.{expert_id}.w1.weight",
            self.layer_prefix(layer)
        ))
    }

    fn expert_ffn_up_key(&self, layer: usize, expert_id: usize) -> Option<String> {
        Some(format!(
            "{}ffn.experts.{expert_id}.w3.weight",
            self.layer_prefix(layer)
        ))
    }

    fn expert_ffn_down_key(&self, layer: usize, expert_id: usize) -> Option<String> {
        Some(format!(
            "{}ffn.experts.{expert_id}.w2.weight",
            self.layer_prefix(layer)
        ))
    }

    fn shared_expert_gate_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}ffn.shared_experts.w1.weight",
            self.layer_prefix(layer)
        ))
    }

    fn shared_expert_up_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}ffn.shared_experts.w3.weight",
            self.layer_prefix(layer)
        ))
    }

    fn shared_expert_down_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}ffn.shared_experts.w2.weight",
            self.layer_prefix(layer)
        ))
    }

    // ── MLA — V4 retains MLA semantics; wq_a / wq_b / wkv pattern ──

    fn uses_mla(&self) -> bool {
        // V4 uses MLA. The exact tensor names differ (wq_a / wq_b / wkv),
        // but the semantic shape matches V3's MLA.
        self.config.kv_lora_rank.is_some() || self.config.q_lora_rank.is_some()
    }

    fn kv_lora_rank(&self) -> usize {
        self.config.kv_lora_rank.unwrap_or(1024)
    }

    fn q_lora_rank(&self) -> usize {
        self.config.q_lora_rank.unwrap_or(1024)
    }

    fn mla_kv_a_key(&self, layer: usize) -> Option<String> {
        Some(format!("{}attn.wkv.weight", self.layer_prefix(layer)))
    }

    fn mla_kv_b_key(&self, _layer: usize) -> Option<String> {
        // V4 fuses kv into wkv; no separate kv_b projection.
        None
    }

    fn mla_q_a_key(&self, layer: usize) -> Option<String> {
        Some(format!("{}attn.wq_a.weight", self.layer_prefix(layer)))
    }

    fn mla_q_b_key(&self, layer: usize) -> Option<String> {
        Some(format!("{}attn.wq_b.weight", self.layer_prefix(layer)))
    }
}
