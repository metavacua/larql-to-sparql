//! Model-architecture config carried in `index.json` so the
//! architecture can be reconstructed without the original
//! `config.json`.
//!
//! Carved out of the monolithic `config/types.rs` in the 2026-04-25
//! round-2 cleanup.

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
pub struct VindexModelConfig {
    pub model_type: String,
    pub head_dim: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub rope_base: f64,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    /// MoE configuration (None for dense models).
    #[serde(default)]
    pub moe: Option<MoeConfig>,

    // ── Gemma 4 per-layer attention geometry ──
    // All optional for backward compatibility with existing vindexes.
    /// Head dimension for global (full) attention layers. If None, all layers use head_dim.
    /// Gemma 4: 512 for global layers, head_dim (256) for sliding.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub global_head_dim: Option<usize>,
    /// Number of KV heads for global attention layers. If None, all layers use num_kv_heads.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_global_kv_heads: Option<usize>,
    /// Fraction of head_dim to apply RoPE to (0.0–1.0). If None, full rotation.
    /// Gemma 4 global layers: 0.25.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub partial_rotary_factor: Option<f64>,
    /// Sliding window pattern: every Nth layer is full attention.
    /// Gemma 4: 6 (layers 5, 11, 17, ... are full).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sliding_window_pattern: Option<usize>,
    /// Explicit per-layer type array (e.g., ["sliding_attention", "full_attention", ...]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer_types: Option<Vec<String>>,
    /// Whether value projection shares key projection (K=V).
    #[serde(default)]
    pub attention_k_eq_v: bool,
    /// Number of layers at the end that share KV from earlier layers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_kv_shared_layers: Option<usize>,
    /// Per-layer embedding dimension (PLE). 0 or None = no PLE.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub per_layer_embed_dim: Option<usize>,
    /// RoPE base for local/sliding window layers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_local_base: Option<f64>,
    /// Query pre-attention scalar (overrides 1/sqrt(head_dim)).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query_pre_attn_scalar: Option<f64>,
    /// Final-logit tanh softcap (Gemma 2/3/4: 30.0). Applied to logits
    /// immediately before softmax in `logits_to_predictions`. Omitting it
    /// leaves logits uncapped — on E2B this peaked the softmax on the
    /// wrong token (observed: "Paris" → "hyperparameters").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_logit_softcapping: Option<f64>,
}

/// MoE (Mixture of Experts) configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoeConfig {
    /// Number of experts per layer.
    pub num_experts: usize,
    /// Number of experts selected per token (top-K routing).
    pub top_k: usize,
    /// Whether there's a shared expert always active (DeepSeek V2/V3).
    #[serde(default)]
    pub shared_expert: bool,
    /// Router type (e.g., "top_k_softmax", "gemma4_top_k_softmax").
    #[serde(default = "default_router_type")]
    pub router_type: String,
    /// Per-expert intermediate (hidden) dimension.
    /// Differs from the dense FFN intermediate_size in hybrid models (Gemma 4 A4B).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub moe_intermediate_size: Option<usize>,
    /// Hybrid MoE: dense MLP and expert block coexist in each layer, outputs summed.
    /// True for Gemma 4 A4B. False for pure MoE (Mixtral, DeepSeek).
    #[serde(default)]
    pub hybrid: bool,
}

fn default_router_type() -> String {
    "top_k_softmax".to_string()
}

impl VindexModelConfig {
    /// Build the serialisable vindex architecture config from the detected
    /// model architecture. Keeping this mapping in one place prevents vector
    /// imports, f32 writers, and Q4K writers from drifting.
    pub fn from_arch(arch: &dyn larql_models::ModelArchitecture) -> Self {
        let cfg = arch.config();
        Self {
            model_type: cfg.model_type.clone(),
            head_dim: cfg.head_dim,
            num_q_heads: cfg.num_q_heads,
            num_kv_heads: cfg.num_kv_heads,
            rope_base: cfg.rope_base,
            sliding_window: cfg.sliding_window,
            moe: if arch.is_moe() {
                Some(MoeConfig {
                    num_experts: arch.num_experts(),
                    top_k: arch.num_experts_per_token(),
                    shared_expert: arch.num_shared_experts() > 0,
                    router_type: arch.moe_router_type().into(),
                    moe_intermediate_size: if arch.moe_intermediate_size() > 0 {
                        Some(arch.moe_intermediate_size())
                    } else {
                        None
                    },
                    hybrid: arch.is_hybrid_moe(),
                })
            } else {
                None
            },
            global_head_dim: cfg.global_head_dim,
            num_global_kv_heads: cfg.num_global_kv_heads,
            partial_rotary_factor: cfg.partial_rotary_factor,
            sliding_window_pattern: cfg.sliding_window_pattern,
            layer_types: cfg.layer_types.clone(),
            attention_k_eq_v: cfg.attention_k_eq_v,
            num_kv_shared_layers: cfg.num_kv_shared_layers,
            per_layer_embed_dim: cfg.per_layer_embed_dim,
            rope_local_base: cfg.rope_local_base,
            query_pre_attn_scalar: cfg.query_pre_attn_scalar,
            final_logit_softcapping: cfg.final_logit_softcapping,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_model_config() -> VindexModelConfig {
        VindexModelConfig {
            model_type: "gemma3".into(),
            head_dim: 256,
            num_q_heads: 8,
            num_kv_heads: 4,
            rope_base: 10000.0,
            sliding_window: None,
            moe: None,
            global_head_dim: None,
            num_global_kv_heads: None,
            partial_rotary_factor: None,
            sliding_window_pattern: None,
            layer_types: None,
            attention_k_eq_v: false,
            num_kv_shared_layers: None,
            per_layer_embed_dim: None,
            rope_local_base: None,
            query_pre_attn_scalar: None,
            final_logit_softcapping: None,
        }
    }

    #[test]
    fn model_config_serde_round_trip() {
        let cfg = minimal_model_config();
        let j = serde_json::to_string(&cfg).unwrap();
        let back: VindexModelConfig = serde_json::from_str(&j).unwrap();
        assert_eq!(back.model_type, "gemma3");
        assert_eq!(back.head_dim, 256);
        assert_eq!(back.num_q_heads, 8);
        assert_eq!(back.num_kv_heads, 4);
    }

    #[test]
    fn optional_fields_absent_in_json_when_none() {
        let cfg = minimal_model_config();
        let j = serde_json::to_string(&cfg).unwrap();
        assert!(
            !j.contains("global_head_dim"),
            "None optional should be omitted"
        );
        assert!(
            !j.contains("sliding_window_pattern"),
            "None optional should be omitted"
        );
    }

    #[test]
    fn model_config_with_softcap_round_trips() {
        let mut cfg = minimal_model_config();
        cfg.final_logit_softcapping = Some(30.0);
        let j = serde_json::to_string(&cfg).unwrap();
        let back: VindexModelConfig = serde_json::from_str(&j).unwrap();
        assert_eq!(back.final_logit_softcapping, Some(30.0));
    }

    #[test]
    fn model_config_with_moe() {
        let mut cfg = minimal_model_config();
        cfg.moe = Some(MoeConfig {
            num_experts: 8,
            top_k: 2,
            shared_expert: false,
            router_type: "top_k_softmax".into(),
            moe_intermediate_size: Some(2048),
            hybrid: false,
        });
        let j = serde_json::to_string(&cfg).unwrap();
        let back: VindexModelConfig = serde_json::from_str(&j).unwrap();
        let moe = back.moe.unwrap();
        assert_eq!(moe.num_experts, 8);
        assert_eq!(moe.top_k, 2);
    }

    #[test]
    fn moe_config_default_router_type_via_serde() {
        let json = r#"{"num_experts":4,"top_k":1,"shared_expert":false}"#;
        let moe: MoeConfig = serde_json::from_str(json).unwrap();
        assert_eq!(moe.router_type, "top_k_softmax");
        assert!(!moe.hybrid);
    }

    #[test]
    fn moe_shared_expert_default_false() {
        let json = r#"{"num_experts":4,"top_k":2,"router_type":"custom"}"#;
        let moe: MoeConfig = serde_json::from_str(json).unwrap();
        assert!(!moe.shared_expert);
        assert!(!moe.hybrid);
    }
}
