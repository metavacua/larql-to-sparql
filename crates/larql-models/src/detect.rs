//! Auto-detect model architecture from config.json.

use std::path::Path;

use crate::architectures::deepseek::DeepSeekArch;
use crate::architectures::gemma2::Gemma2Arch;
use crate::architectures::gemma3::Gemma3Arch;
use crate::architectures::generic::GenericArch;
use crate::architectures::gpt_oss::GptOssArch;
use crate::architectures::granite::GraniteArch;
use crate::architectures::llama::LlamaArch;
use crate::architectures::mistral::MistralArch;
use crate::architectures::mixtral::MixtralArch;
use crate::architectures::qwen::QwenArch;
use crate::architectures::starcoder2::StarCoder2Arch;
use crate::config::{ModelArchitecture, ModelConfig, RopeScaling};

/// Error from model detection/config parsing.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("parse error: {0}")]
    Parse(String),
    #[error("unsupported dtype: {0}")]
    UnsupportedDtype(String),
    #[error("missing tensor: {0}")]
    MissingTensor(String),
    #[error("not a directory: {0}")]
    NotADirectory(std::path::PathBuf),
    #[error("no safetensors files in {0}")]
    NoSafetensors(std::path::PathBuf),
}

/// Read config.json from a model directory and return the architecture.
pub fn detect_architecture(model_dir: &Path) -> Result<Box<dyn ModelArchitecture>, ModelError> {
    let config_path = model_dir.join("config.json");
    let config_json = if config_path.exists() {
        let text = std::fs::read_to_string(&config_path)?;
        serde_json::from_str::<serde_json::Value>(&text)?
    } else {
        serde_json::json!({})
    };

    Ok(detect_from_json(&config_json))
}

/// Detect architecture from an already-parsed config.json value.
pub fn detect_from_json(config: &serde_json::Value) -> Box<dyn ModelArchitecture> {
    let model_config = parse_model_config(config);
    let model_type = model_config.model_type.as_str();

    match model_type {
        // Gemma family
        t if t.starts_with("gemma3") => Box::new(Gemma3Arch::from_config(model_config)),
        t if t.starts_with("gemma2") || t == "gemma" => Box::new(Gemma2Arch::from_config(model_config)),
        // Llama family
        t if t.starts_with("llama") => Box::new(LlamaArch::from_config(model_config)),
        // Mistral (dense)
        "mistral" => Box::new(MistralArch::from_config(model_config)),
        // Mixtral (MoE) — block_sparse_moe pattern
        "mixtral" => Box::new(MixtralArch::from_config(model_config)),
        // GPT-OSS (MoE, MXFP4 packed experts)
        "gpt_oss" => Box::new(GptOssArch::from_config(model_config)),
        // Qwen family (dense and MoE share same keys)
        t if t.starts_with("qwen") => Box::new(QwenArch::from_config(model_config)),
        // DeepSeek family (MoE + MLA)
        t if t.starts_with("deepseek") => Box::new(DeepSeekArch::from_config(model_config)),
        // StarCoder 2
        "starcoder2" => Box::new(StarCoder2Arch::from_config(model_config)),
        // Granite family (dense and MoE share same base keys)
        t if t.starts_with("granite") => Box::new(GraniteArch::from_config(model_config)),
        // Unknown — generic fallback
        _ => Box::new(GenericArch::from_config(model_config)),
    }
}

/// Parse ModelConfig from a config.json value.
/// Handles both top-level and nested text_config (multimodal models).
fn parse_model_config(config: &serde_json::Value) -> ModelConfig {
    let text_config = config.get("text_config").unwrap_or(config);

    // Detect model_type from text_config or top level.
    let model_type = text_config["model_type"]
        .as_str()
        .or_else(|| config["model_type"].as_str())
        .unwrap_or("")
        .to_string();

    // Pick defaults based on model type.
    let is_gemma = model_type.starts_with("gemma3") || model_type.starts_with("gemma2");
    let rope_default = if is_gemma { 1_000_000.0 } else { 10_000.0 };

    let num_layers = text_config["num_hidden_layers"].as_u64().unwrap_or(32) as usize;
    let hidden_size = text_config["hidden_size"].as_u64().unwrap_or(2048) as usize;
    let intermediate_size = text_config["intermediate_size"].as_u64().unwrap_or(8192) as usize;
    // Gemma3 HF configs omit num_attention_heads, head_dim, num_key_value_heads
    // from text_config — they're architecture-class defaults in transformers.
    let default_head_dim: usize = if is_gemma { 256 } else { 0 };
    let num_q_heads = text_config["num_attention_heads"].as_u64().unwrap_or(8) as usize;
    // head_dim: explicit config value, Gemma default (256), or compute from hidden/heads.
    let head_dim = text_config["head_dim"]
        .as_u64()
        .map(|v| v as usize)
        .unwrap_or(if default_head_dim > 0 { default_head_dim } else { hidden_size / num_q_heads });
    let num_kv_heads = text_config["num_key_value_heads"].as_u64().unwrap_or(4) as usize;
    let rope_base = text_config["rope_theta"].as_f64().unwrap_or(rope_default);
    let rope_local_base = text_config["rope_local_base_freq"].as_f64();
    let vocab_size = text_config["vocab_size"].as_u64().map(|v| v as usize);
    let sliding_window = text_config["sliding_window"].as_u64().map(|v| v as usize);

    // MoE fields
    let num_experts = text_config["n_routed_experts"]
        .as_u64()
        .or_else(|| text_config["num_local_experts"].as_u64())
        .map(|v| v as usize);
    let num_experts_per_token = text_config["num_experts_per_tok"]
        .as_u64()
        .or_else(|| text_config["num_experts_per_token"].as_u64())
        .map(|v| v as usize);
    let num_shared_experts = text_config["n_shared_experts"]
        .as_u64()
        .map(|v| v as usize);

    // MLA fields
    let kv_lora_rank = text_config["kv_lora_rank"].as_u64().map(|v| v as usize);
    let q_lora_rank = text_config["q_lora_rank"].as_u64().map(|v| v as usize);

    // RoPE scaling
    let rope_scaling = text_config.get("rope_scaling").and_then(|rs| {
        // HF uses "type" for most models, but Llama 3.1+ uses "rope_type"
        let scaling_type = rs
            .get("type")
            .or_else(|| rs.get("rope_type"))
            .and_then(|v| v.as_str())?
            .to_string();
        let factor = rs.get("factor")?.as_f64()?;
        Some(RopeScaling {
            scaling_type,
            factor,
        })
    });

    // Softcapping and attention scale
    let attn_logit_softcapping = text_config["attn_logit_softcapping"].as_f64();
    let final_logit_softcapping = text_config["final_logit_softcapping"].as_f64();
    let query_pre_attn_scalar = text_config["query_pre_attn_scalar"].as_f64();

    // Granite-style scaling multipliers
    let embedding_multiplier = text_config["embedding_multiplier"].as_f64();
    let residual_multiplier = text_config["residual_multiplier"].as_f64();
    let attention_multiplier = text_config["attention_multiplier"].as_f64();
    let logits_scaling = text_config["logits_scaling"].as_f64();

    ModelConfig {
        model_type,
        num_layers,
        hidden_size,
        intermediate_size,
        head_dim,
        num_q_heads,
        num_kv_heads,
        vocab_size,
        rope_base,
        rope_local_base,
        sliding_window,
        num_experts,
        num_experts_per_token,
        num_shared_experts,
        kv_lora_rank,
        q_lora_rank,
        rope_scaling,
        attn_logit_softcapping,
        final_logit_softcapping,
        query_pre_attn_scalar,
        embedding_multiplier,
        residual_multiplier,
        attention_multiplier,
        logits_scaling,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_gemma3() {
        let config = serde_json::json!({
            "model_type": "gemma3",
            "text_config": {
                "model_type": "gemma3_text",
                "hidden_size": 2560,
                "head_dim": 256,
                "num_hidden_layers": 34,
                "num_attention_heads": 8,
                "intermediate_size": 10240,
                "sliding_window": 1024
            }
        });

        let arch = detect_from_json(&config);
        assert_eq!(arch.family(), "gemma3");
        assert_eq!(arch.config().num_layers, 34);
        assert_eq!(arch.config().hidden_size, 2560);
        assert_eq!(arch.config().rope_base, 1_000_000.0);
        assert_eq!(arch.norm_weight_offset(), 1.0);
        assert_eq!(arch.embed_scale(), (2560.0f32).sqrt());
        assert!(arch.has_post_norms());
        assert!(arch.attn_q_norm_key(0).is_some());

        // Sliding window: layer 4 is sliding, layer 5 is full
        assert!(arch.is_sliding_window_layer(4));
        assert!(!arch.is_sliding_window_layer(5));
    }

    #[test]
    fn test_detect_llama() {
        let config = serde_json::json!({
            "model_type": "llama",
            "hidden_size": 4096,
            "num_hidden_layers": 32
        });

        let arch = detect_from_json(&config);
        assert_eq!(arch.family(), "llama");
        assert_eq!(arch.config().hidden_size, 4096);
        assert_eq!(arch.config().rope_base, 10_000.0);
        assert_eq!(arch.norm_weight_offset(), 0.0);
        assert_eq!(arch.embed_scale(), 1.0);
        assert!(!arch.has_post_norms());
        assert!(arch.attn_q_norm_key(0).is_none());
    }

    #[test]
    fn test_detect_mistral() {
        let config = serde_json::json!({
            "model_type": "mistral",
            "hidden_size": 4096,
            "num_hidden_layers": 32
        });

        let arch = detect_from_json(&config);
        assert_eq!(arch.family(), "mistral");
    }

    #[test]
    fn test_detect_qwen2() {
        let config = serde_json::json!({
            "model_type": "qwen2",
            "hidden_size": 4096,
            "num_hidden_layers": 32
        });

        let arch = detect_from_json(&config);
        assert_eq!(arch.family(), "qwen2");
    }

    #[test]
    fn test_detect_qwen3() {
        let config = serde_json::json!({
            "model_type": "qwen3",
            "hidden_size": 2048,
            "num_hidden_layers": 28
        });

        let arch = detect_from_json(&config);
        assert_eq!(arch.family(), "qwen3");
    }

    #[test]
    fn test_detect_unknown_defaults_to_generic() {
        let config = serde_json::json!({
            "model_type": "some_unknown_model",
            "hidden_size": 2048,
            "num_hidden_layers": 24
        });

        let arch = detect_from_json(&config);
        assert_eq!(arch.family(), "generic");
    }

    #[test]
    fn test_tensor_keys() {
        let config = serde_json::json!({"model_type": "gemma3_text"});
        let arch = detect_from_json(&config);

        assert_eq!(arch.attn_q_key(5), "layers.5.self_attn.q_proj.weight");
        assert_eq!(arch.ffn_gate_key(10), "layers.10.mlp.gate_proj.weight");
        assert_eq!(
            arch.input_layernorm_key(0),
            "layers.0.input_layernorm.weight"
        );
        assert_eq!(arch.final_norm_key(), "norm.weight");
        assert_eq!(arch.embed_key(), "embed_tokens.weight");

        assert_eq!(
            arch.attn_q_norm_key(3),
            Some("layers.3.self_attn.q_norm.weight".to_string())
        );
    }

    #[test]
    fn test_detect_llama2() {
        // Real Llama 2 7B config — no head_dim, no rope_theta, no GQA
        let config = serde_json::json!({
            "model_type": "llama",
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "vocab_size": 32000
        });

        let arch = detect_from_json(&config);
        assert_eq!(arch.family(), "llama");
        assert_eq!(arch.config().num_layers, 32);
        assert_eq!(arch.config().hidden_size, 4096);
        assert_eq!(arch.config().num_q_heads, 32);
        assert_eq!(arch.config().num_kv_heads, 32); // no GQA in Llama 2
        // head_dim computed: 4096 / 32 = 128
        assert_eq!(arch.config().head_dim, 128);
        // rope_theta absent → defaults to 10000
        assert_eq!(arch.config().rope_base, 10_000.0);
        assert!(!arch.is_moe());
        assert!(!arch.uses_mla());

        // Standard tensor keys
        assert_eq!(arch.attn_q_key(0), "layers.0.self_attn.q_proj.weight");
        assert_eq!(arch.ffn_gate_key(5), "layers.5.mlp.gate_proj.weight");
        assert_eq!(
            arch.input_layernorm_key(0),
            "layers.0.input_layernorm.weight"
        );
        assert_eq!(
            arch.post_attention_layernorm_key(0),
            "layers.0.post_attention_layernorm.weight"
        );
        assert_eq!(arch.embed_key(), "embed_tokens.weight");
        assert_eq!(arch.final_norm_key(), "norm.weight");
    }

    #[test]
    fn test_detect_llama3() {
        // Real Llama 3 8B config — no head_dim, GQA (8 KV heads), higher rope_theta
        let config = serde_json::json!({
            "model_type": "llama",
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 128256,
            "rope_theta": 500000.0
        });

        let arch = detect_from_json(&config);
        assert_eq!(arch.family(), "llama");
        assert_eq!(arch.config().num_kv_heads, 8); // GQA in Llama 3
        assert_eq!(arch.config().head_dim, 128); // computed: 4096/32
        assert_eq!(arch.config().rope_base, 500_000.0);
        assert_eq!(arch.config().vocab_size, Some(128256));
        assert!(arch.rope_scaling_type().is_none()); // no scaling in base Llama 3
    }

    #[test]
    fn test_detect_llama31() {
        // Real Llama 3.1 8B config — uses "rope_type" instead of "type"
        let config = serde_json::json!({
            "model_type": "llama",
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 128256,
            "rope_theta": 500000.0,
            "rope_scaling": {
                "rope_type": "llama3",
                "factor": 8.0
            }
        });

        let arch = detect_from_json(&config);
        assert_eq!(arch.family(), "llama");
        assert_eq!(arch.rope_scaling_type(), Some("llama3"));
        assert_eq!(arch.rope_scaling_factor(), 8.0);
    }

    #[test]
    fn test_detect_mistral_7b() {
        // Real Mistral 7B config — no head_dim, GQA, sliding window
        let config = serde_json::json!({
            "model_type": "mistral",
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "sliding_window": 4096
        });

        let arch = detect_from_json(&config);
        assert_eq!(arch.family(), "mistral");
        assert_eq!(arch.config().num_kv_heads, 8);
        assert_eq!(arch.config().head_dim, 128); // computed: 4096/32
        assert_eq!(arch.sliding_window_size(), Some(4096));
    }

    #[test]
    fn test_detect_deepseek_v2() {
        let config = serde_json::json!({
            "model_type": "deepseek_v2",
            "hidden_size": 5120,
            "intermediate_size": 12288,
            "num_hidden_layers": 60,
            "num_attention_heads": 128,
            "num_key_value_heads": 128,
            "head_dim": 128,
            "n_routed_experts": 160,
            "num_experts_per_tok": 6,
            "n_shared_experts": 2,
            "kv_lora_rank": 512,
            "q_lora_rank": 1536,
            "rope_scaling": {
                "type": "yarn",
                "factor": 40.0
            }
        });

        let arch = detect_from_json(&config);
        assert_eq!(arch.family(), "deepseek");

        // MoE
        assert!(arch.is_moe());
        assert_eq!(arch.num_experts(), 160);
        assert_eq!(arch.num_experts_per_token(), 6);
        assert_eq!(arch.num_shared_experts(), 2);

        // MoE tensor keys
        assert_eq!(
            arch.moe_router_key(0),
            Some("layers.0.mlp.gate.weight".to_string())
        );
        assert_eq!(
            arch.expert_ffn_gate_key(5, 3),
            Some("layers.5.mlp.experts.3.gate_proj.weight".to_string())
        );
        assert_eq!(
            arch.shared_expert_down_key(10),
            Some("layers.10.mlp.shared_experts.down_proj.weight".to_string())
        );

        // MLA
        assert!(arch.uses_mla());
        assert_eq!(arch.kv_lora_rank(), 512);
        assert_eq!(arch.q_lora_rank(), 1536);
        assert_eq!(
            arch.mla_kv_a_key(0),
            Some("layers.0.self_attn.kv_a_proj_with_mqa.weight".to_string())
        );
        assert_eq!(
            arch.mla_q_b_key(5),
            Some("layers.5.self_attn.q_b_proj.weight".to_string())
        );

        // RoPE
        assert_eq!(arch.rope_scaling_type(), Some("yarn"));
        assert_eq!(arch.rope_scaling_factor(), 40.0);
    }

    #[test]
    fn test_detect_deepseek_v3() {
        let config = serde_json::json!({
            "model_type": "deepseek_v3",
            "hidden_size": 7168,
            "num_hidden_layers": 61,
            "n_routed_experts": 256,
            "num_experts_per_tok": 8,
            "n_shared_experts": 1,
            "kv_lora_rank": 512,
            "q_lora_rank": 1536
        });

        let arch = detect_from_json(&config);
        assert_eq!(arch.family(), "deepseek");
        assert!(arch.is_moe());
        assert_eq!(arch.num_experts(), 256);
        assert_eq!(arch.num_experts_per_token(), 8);
        assert_eq!(arch.num_shared_experts(), 1);
    }

    #[test]
    fn test_non_moe_model_defaults() {
        let config = serde_json::json!({
            "model_type": "llama",
            "hidden_size": 4096,
            "num_hidden_layers": 32
        });

        let arch = detect_from_json(&config);
        assert!(!arch.is_moe());
        assert_eq!(arch.num_experts(), 0);
        assert!(!arch.uses_mla());
        assert_eq!(arch.kv_lora_rank(), 0);
        assert!(arch.moe_router_key(0).is_none());
        assert!(arch.mla_kv_a_key(0).is_none());
        assert!(arch.rope_scaling_type().is_none());
        assert_eq!(arch.rope_scaling_factor(), 1.0);
    }

    // ── Tests against real HuggingFace configs ──

    #[test]
    fn test_real_llama32_3b() {
        // Exact config from meta-llama/Llama-3.2-3B-Instruct
        let config = serde_json::json!({
            "model_type": "llama",
            "hidden_size": 3072,
            "intermediate_size": 8192,
            "num_hidden_layers": 28,
            "num_attention_heads": 24,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 128256,
            "rope_theta": 500000.0,
            "rope_scaling": {
                "factor": 32.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3"
            }
        });

        let arch = detect_from_json(&config);
        assert_eq!(arch.family(), "llama");
        assert_eq!(arch.config().hidden_size, 3072);
        assert_eq!(arch.config().head_dim, 128);
        assert_eq!(arch.config().num_q_heads, 24);
        assert_eq!(arch.config().num_kv_heads, 8);
        assert_eq!(arch.config().num_layers, 28);
        assert_eq!(arch.config().rope_base, 500_000.0);
        assert_eq!(arch.rope_scaling_type(), Some("llama3"));
        assert_eq!(arch.rope_scaling_factor(), 32.0);
    }

    #[test]
    fn test_real_llama32_1b() {
        // Exact config from meta-llama/Llama-3.2-1B — head_dim=64 (not 128!)
        let config = serde_json::json!({
            "model_type": "llama",
            "hidden_size": 2048,
            "intermediate_size": 8192,
            "num_hidden_layers": 16,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 64,
            "vocab_size": 128256,
            "rope_theta": 500000.0,
            "rope_scaling": {
                "factor": 32.0,
                "rope_type": "llama3"
            }
        });

        let arch = detect_from_json(&config);
        assert_eq!(arch.family(), "llama");
        assert_eq!(arch.config().head_dim, 64); // explicit, not computed
        assert_eq!(arch.config().num_q_heads, 32);
        // Without explicit head_dim, compute would give 2048/32=64 — same result
        assert_eq!(arch.rope_scaling_type(), Some("llama3"));
    }

    #[test]
    fn test_real_mistral_7b_v03() {
        // Exact config from mistralai/Mistral-7B-Instruct-v0.3 — head_dim null
        let config = serde_json::json!({
            "model_type": "mistral",
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": null,
            "vocab_size": 32768,
            "rope_theta": 1000000.0,
            "sliding_window": null
        });

        let arch = detect_from_json(&config);
        assert_eq!(arch.family(), "mistral");
        assert_eq!(arch.config().head_dim, 128); // computed: 4096/32
        assert_eq!(arch.config().rope_base, 1_000_000.0);
        assert!(arch.sliding_window_size().is_none());
    }

    #[test]
    fn test_real_tinyllama() {
        // Exact config from TinyLlama/TinyLlama-1.1B-Chat-v1.0
        let config = serde_json::json!({
            "model_type": "llama",
            "hidden_size": 2048,
            "intermediate_size": 5632,
            "num_hidden_layers": 22,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "vocab_size": 32000,
            "rope_theta": 10000.0
        });

        let arch = detect_from_json(&config);
        assert_eq!(arch.family(), "llama");
        assert_eq!(arch.config().head_dim, 64); // computed: 2048/32
        assert_eq!(arch.config().num_kv_heads, 4);
        assert_eq!(arch.config().rope_base, 10_000.0);
    }

    #[test]
    fn test_real_mixtral_8x7b() {
        // Exact config from mistralai/Mixtral-8x7B-Instruct-v0.1
        let config = serde_json::json!({
            "model_type": "mixtral",
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 32000,
            "rope_theta": 1000000.0,
            "num_local_experts": 8,
            "num_experts_per_tok": 2
        });

        let arch = detect_from_json(&config);
        assert_eq!(arch.family(), "mixtral");
        assert!(arch.is_moe());
        assert_eq!(arch.num_experts(), 8);
        assert_eq!(arch.num_experts_per_token(), 2);

        // Mixtral MoE tensor keys — block_sparse_moe + w1/w2/w3
        assert_eq!(
            arch.moe_router_key(0),
            Some("layers.0.block_sparse_moe.gate.weight".to_string())
        );
        assert_eq!(
            arch.expert_ffn_gate_key(5, 3),
            Some("layers.5.block_sparse_moe.experts.3.w1.weight".to_string())
        );
        assert_eq!(
            arch.expert_ffn_down_key(5, 3),
            Some("layers.5.block_sparse_moe.experts.3.w2.weight".to_string())
        );
        assert_eq!(
            arch.expert_ffn_up_key(5, 3),
            Some("layers.5.block_sparse_moe.experts.3.w3.weight".to_string())
        );

        // Attention is standard Llama
        assert_eq!(arch.attn_q_key(0), "layers.0.self_attn.q_proj.weight");
    }

    #[test]
    fn test_real_starcoder2_3b() {
        // Exact config from bigcode/starcoder2-3b
        let config = serde_json::json!({
            "model_type": "starcoder2",
            "hidden_size": 3072,
            "intermediate_size": 12288,
            "num_hidden_layers": 30,
            "num_attention_heads": 24,
            "num_key_value_heads": 2,
            "vocab_size": 49152,
            "rope_theta": 999999.4420358813,
            "sliding_window": 4096
        });

        let arch = detect_from_json(&config);
        assert_eq!(arch.family(), "starcoder2");
        assert_eq!(arch.config().head_dim, 128); // 3072/24
        assert_eq!(arch.config().num_kv_heads, 2);
        assert_eq!(arch.sliding_window_size(), Some(4096));
        assert!(!arch.is_moe());
    }

    #[test]
    fn test_real_granite_2b() {
        // Exact config from ibm-granite/granite-3.1-2b-base
        let config = serde_json::json!({
            "model_type": "granite",
            "hidden_size": 2048,
            "intermediate_size": 8192,
            "num_hidden_layers": 40,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 49155,
            "rope_theta": 5000000.0
        });

        let arch = detect_from_json(&config);
        assert_eq!(arch.family(), "granite");
        assert_eq!(arch.config().head_dim, 64); // 2048/32
        assert_eq!(arch.config().rope_base, 5_000_000.0);
        assert!(!arch.is_moe());
    }

    #[test]
    fn test_real_granitemoe() {
        // Exact config from ibm-granite/granite-3.0-1b-a400m-instruct
        let config = serde_json::json!({
            "model_type": "granitemoe",
            "hidden_size": 1024,
            "intermediate_size": 512,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "vocab_size": 49155,
            "rope_theta": 10000,
            "num_local_experts": 32,
            "num_experts_per_tok": 8
        });

        let arch = detect_from_json(&config);
        assert_eq!(arch.family(), "granitemoe");
        assert_eq!(arch.config().num_experts, Some(32));
        assert_eq!(arch.config().num_experts_per_token, Some(8));
    }

    #[test]
    fn test_real_qwen2_moe() {
        // Exact config from Qwen/Qwen1.5-MoE-A2.7B-Chat
        let config = serde_json::json!({
            "model_type": "qwen2_moe",
            "hidden_size": 2048,
            "intermediate_size": 5632,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "vocab_size": 151936,
            "rope_theta": 1000000.0,
            "sliding_window": 32768,
            "num_experts_per_tok": 4
        });

        let arch = detect_from_json(&config);
        assert_eq!(arch.family(), "qwen2_moe");
    }

    #[test]
    fn test_empty_config() {
        let config = serde_json::json!({});
        let arch = detect_from_json(&config);
        assert_eq!(arch.family(), "generic");
        assert_eq!(arch.config().num_layers, 32);
    }
}
