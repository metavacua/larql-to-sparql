//! Architecture-detection tests.
//!
//! Organised here rather than alongside their implementation files so the
//! existing assertion suite can keep using a single `super::*` import
//! covering the whole `detect` module.

use super::config_io::{
    CONFIG_KEY_HIDDEN_SIZE, CONFIG_KEY_INTERMEDIATE_SIZE, CONFIG_KEY_NUM_HIDDEN_LAYERS,
    REQUIRED_CONFIG_FIELDS,
};
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
fn test_detect_tinymodel() {
    let config = serde_json::json!({
        "model_type": "tinymodel",
        "hidden_size": 512,
        "num_hidden_layers": 20,
        "intermediate_size": 2048,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "vocab_size": 71261,
        "max_position_embeddings": 256
    });

    let arch = detect_from_json(&config);
    assert_eq!(arch.family(), "tinymodel");
    assert_eq!(arch.config().hidden_size, 512);
    assert_eq!(arch.config().num_layers, 20);
    assert_eq!(arch.config().rope_base, 10_000.0);
    assert_eq!(arch.embed_scale(), (512.0_f32).sqrt());
    assert_eq!(arch.embed_key(), "embed.weight");
    assert_eq!(arch.final_norm_key(), "norm.weight");
    assert_eq!(arch.attn_q_key(5), "layers.5.attn.q_proj.weight");
    assert_eq!(arch.ffn_gate_key(5), "layers.5.ffn.gate.weight");
    assert_eq!(arch.ffn_down_key(5), "layers.5.ffn.down.weight");
    assert_eq!(arch.input_layernorm_key(5), "layers.5.attn_norm.weight");
    assert_eq!(
        arch.post_attention_layernorm_key(5),
        "layers.5.ffn_norm.weight"
    );
    assert_eq!(arch.key_prefixes_to_strip(), &[] as &[&str]);
    assert!(!arch.has_post_norms());
}

#[test]
fn test_tinymodel_full_key_coverage() {
    let config = serde_json::json!({
        "model_type": "tinymodel",
        "hidden_size": 512,
        "num_hidden_layers": 20,
        "intermediate_size": 2048,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
    });
    let arch = detect_from_json(&config);

    // Complete attention key set
    assert_eq!(arch.attn_q_key(7), "layers.7.attn.q_proj.weight");
    assert_eq!(arch.attn_k_key(7), "layers.7.attn.k_proj.weight");
    assert_eq!(arch.attn_v_key(7), "layers.7.attn.v_proj.weight");
    assert_eq!(arch.attn_o_key(7), "layers.7.attn.o_proj.weight");

    // Complete FFN key set
    assert_eq!(arch.ffn_gate_key(7), "layers.7.ffn.gate.weight");
    assert_eq!(arch.ffn_up_key(7), "layers.7.ffn.up.weight");
    assert_eq!(arch.ffn_down_key(7), "layers.7.ffn.down.weight");

    // Not MoE, not MLA, no QK norm
    assert!(!arch.is_moe());
    assert!(!arch.uses_mla());
    assert!(arch.attn_q_norm_key(0).is_none());
    assert!(arch.attn_k_norm_key(0).is_none());
}

#[test]
fn test_gemma4_key_formats() {
    let config = serde_json::json!({
        "model_type": "gemma4",
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 1536,
            "intermediate_size": 6144,
            "num_hidden_layers": 8,
            "num_attention_heads": 8,
            "num_key_value_heads": 1,
            "head_dim": 256,
        }
    });
    let arch = detect_from_json(&config);

    // Gemma 4 uses HF-style llama keys (no architecture-specific override in gemma4.rs)
    assert_eq!(arch.attn_q_key(3), "layers.3.self_attn.q_proj.weight");
    assert_eq!(arch.attn_k_key(3), "layers.3.self_attn.k_proj.weight");
    assert_eq!(arch.attn_v_key(3), "layers.3.self_attn.v_proj.weight");
    assert_eq!(arch.attn_o_key(3), "layers.3.self_attn.o_proj.weight");
    assert_eq!(arch.ffn_gate_key(3), "layers.3.mlp.gate_proj.weight");
    assert_eq!(arch.ffn_up_key(3), "layers.3.mlp.up_proj.weight");
    assert_eq!(arch.ffn_down_key(3), "layers.3.mlp.down_proj.weight");

    // Multimodal wrapper prefixes (stripped on load)
    let prefixes = arch.key_prefixes_to_strip();
    assert!(prefixes.contains(&"model.language_model.model."));
    assert!(prefixes.contains(&"model.language_model."));
    assert!(prefixes.contains(&"language_model.model."));
    assert!(prefixes.contains(&"model."));

    // QK norm keys (inherited from Gemma 3)
    assert_eq!(
        arch.attn_q_norm_key(3),
        Some("layers.3.self_attn.q_norm.weight".to_string())
    );
    assert_eq!(
        arch.attn_k_norm_key(3),
        Some("layers.3.self_attn.k_norm.weight".to_string())
    );

    // Gemma 4's shipped tokenizer.json drops BOS from its post-processor
    // `single` template (Gemma 2/3 kept it), so the arch must advertise
    // the BOS id so the inference tokenizer helper can prepend it.
    assert_eq!(arch.bos_token_id(), Some(2));
}

#[test]
fn test_bos_token_id_gemma4_only() {
    // Only Gemma 4 advertises an explicit BOS id — every other
    // architecture's tokenizer.json already includes BOS in its
    // post-processor so callers don't need to prepend it.
    let non_gemma4 = [
        serde_json::json!({"model_type": "llama", "hidden_size": 4096,
            "num_hidden_layers": 32, "intermediate_size": 14336,
            "num_attention_heads": 32, "num_key_value_heads": 8}),
        serde_json::json!({"model_type": "gemma3", "hidden_size": 2560,
            "num_hidden_layers": 34}),
        serde_json::json!({"model_type": "gemma2", "hidden_size": 2304,
            "num_hidden_layers": 26}),
        serde_json::json!({"model_type": "mistral", "hidden_size": 4096,
            "num_hidden_layers": 32}),
        serde_json::json!({"model_type": "qwen2", "hidden_size": 2048,
            "num_hidden_layers": 24, "intermediate_size": 5504,
            "num_attention_heads": 16, "num_key_value_heads": 2}),
        serde_json::json!({"model_type": "tinymodel", "hidden_size": 512,
            "num_hidden_layers": 20, "intermediate_size": 2048,
            "num_attention_heads": 8, "num_key_value_heads": 4}),
    ];
    for cfg in &non_gemma4 {
        let arch = detect_from_json(cfg);
        assert!(
            arch.bos_token_id().is_none(),
            "{} should not advertise a BOS id",
            arch.family()
        );
    }
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
    assert!(!arch.is_moe());
}

#[test]
fn test_detect_qwen3_moe_30b() {
    // Matches Qwen/Qwen3-30B-A3B config.json
    let config = serde_json::json!({
        "model_type": "qwen3_moe",
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "moe_intermediate_size": 768,
        "num_hidden_layers": 48,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_experts": 128,
        "num_experts_per_tok": 8
    });

    let arch = detect_from_json(&config);
    assert!(arch.is_moe());
    assert!(!arch.is_hybrid_moe());
    assert_eq!(arch.num_experts(), 128);
    assert_eq!(arch.num_experts_per_token(), 8);
    assert_eq!(arch.moe_intermediate_size(), 768);
    assert_eq!(arch.moe_router_key(0).unwrap(), "layers.0.mlp.gate.weight");
    assert_eq!(
        arch.expert_ffn_gate_key(0, 5).unwrap(),
        "layers.0.mlp.experts.5.gate_proj.weight"
    );
    assert_eq!(
        arch.expert_ffn_up_key(0, 5).unwrap(),
        "layers.0.mlp.experts.5.up_proj.weight"
    );
    assert_eq!(
        arch.expert_ffn_down_key(0, 5).unwrap(),
        "layers.0.mlp.experts.5.down_proj.weight"
    );
}

#[test]
fn test_detect_gpt2() {
    // GPT-2 small config. Architecture must dispatch to Gpt2Arch with
    // LayerNorm + Standard (non-gated) FFN + GELU-tanh activation.
    let config = serde_json::json!({
        "model_type": "gpt2",
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "num_key_value_heads": 12,
        "vocab_size": 50257
    });

    let arch = detect_from_json(&config);
    assert_eq!(arch.family(), "gpt2");
    assert_eq!(arch.config().hidden_size, 768);
    assert_eq!(arch.config().intermediate_size, 3072);
    assert_eq!(arch.config().num_layers, 12);
    assert_eq!(arch.norm_type(), crate::config::NormType::LayerNorm);
    assert_eq!(arch.activation(), crate::config::Activation::GeluTanh);
    assert_eq!(arch.ffn_type(), crate::config::FfnType::Standard);
    assert!(!arch.is_moe());

    // Fused QKV + every-projection biases are GPT-2-specific; trait
    // defaults return None elsewhere.
    assert_eq!(
        arch.fused_qkv_key(3),
        Some("layers.3.self_attn.qkv_proj.weight".to_string())
    );
    assert_eq!(
        arch.fused_qkv_bias_key(3),
        Some("layers.3.self_attn.qkv_proj.bias".to_string())
    );
    assert_eq!(
        arch.attn_q_bias_key(3),
        Some("layers.3.self_attn.q_proj.bias".to_string())
    );
    assert_eq!(
        arch.attn_k_bias_key(3),
        Some("layers.3.self_attn.k_proj.bias".to_string())
    );
    assert_eq!(
        arch.attn_v_bias_key(3),
        Some("layers.3.self_attn.v_proj.bias".to_string())
    );
    assert_eq!(
        arch.attn_o_bias_key(3),
        Some("layers.3.self_attn.o_proj.bias".to_string())
    );
    assert_eq!(
        arch.ffn_up_bias_key(3),
        Some("layers.3.mlp.up_proj.bias".to_string())
    );
    assert_eq!(
        arch.ffn_down_bias_key(3),
        Some("layers.3.mlp.down_proj.bias".to_string())
    );

    // Learned positional embeddings — wpe lookup key.
    assert_eq!(arch.position_embed_key(), Some("wpe.weight"));
}

#[test]
fn test_non_gpt2_archs_have_no_fused_qkv_or_position_embed() {
    // Defaults must remain None for everyone else, otherwise the loader
    // would try to split projections that are already separate.
    let llama = serde_json::json!({
        "model_type": "llama",
        "hidden_size": 4096,
        "num_hidden_layers": 32
    });
    let arch = detect_from_json(&llama);
    assert!(arch.fused_qkv_key(0).is_none());
    assert!(arch.fused_qkv_bias_key(0).is_none());
    assert!(arch.position_embed_key().is_none());
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
fn test_detect_gemma4_31b() {
    // Real Gemma 4 31B config — matches actual HuggingFace config.json
    let config = serde_json::json!({
        "model_type": "gemma4",
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 5376,
            "intermediate_size": 21504,
            "num_hidden_layers": 60,
            "num_attention_heads": 32,
            "num_key_value_heads": 16,
            "head_dim": 256,
            "global_head_dim": 512,
            "num_global_key_value_heads": 4,
            "vocab_size": 262144,
            "attention_k_eq_v": true,
            "sliding_window": 1024,
            "final_logit_softcapping": 30.0,
            "rope_parameters": {
                "full_attention": {
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 1000000.0,
                    "rope_type": "proportional"
                },
                "sliding_attention": {
                    "rope_theta": 10000.0,
                    "rope_type": "default"
                }
            },
            "layer_types": [
                "sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "sliding_attention", "full_attention",
                "sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "sliding_attention", "full_attention",
                "sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "sliding_attention", "full_attention",
                "sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "sliding_attention", "full_attention",
                "sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "sliding_attention", "full_attention",
                "sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "sliding_attention", "full_attention",
                "sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "sliding_attention", "full_attention",
                "sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "sliding_attention", "full_attention",
                "sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "sliding_attention", "full_attention",
                "sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "sliding_attention", "full_attention"
            ]
        }
    });

    let arch = detect_from_json(&config);
    assert_eq!(arch.family(), "gemma4");
    assert_eq!(arch.config().num_layers, 60);
    assert_eq!(arch.config().hidden_size, 5376);
    assert_eq!(arch.config().head_dim, 256);
    assert_eq!(arch.config().global_head_dim, Some(512));
    assert_eq!(arch.config().num_global_kv_heads, Some(4));

    // Sliding layer (layer 0): uses base head_dim and kv_heads
    assert!(arch.is_sliding_window_layer(0));
    assert_eq!(arch.head_dim_for_layer(0), 256);
    assert_eq!(arch.num_kv_heads_for_layer(0), 16);
    assert_eq!(arch.num_q_heads_for_layer(0), 32);
    assert_eq!(arch.rotary_fraction_for_layer(0), 1.0);

    // Global layer (layer 5): uses global_head_dim and global kv_heads
    assert!(!arch.is_sliding_window_layer(5));
    assert_eq!(arch.head_dim_for_layer(5), 512);
    assert_eq!(arch.num_kv_heads_for_layer(5), 4);
    // Q heads constant across all layers
    assert_eq!(arch.num_q_heads_for_layer(5), 32);
    assert_eq!(arch.rotary_fraction_for_layer(5), 0.25);

    // RoPE bases
    assert_eq!(arch.rope_base_for_layer(0), 10_000.0); // sliding
    assert_eq!(arch.rope_base_for_layer(5), 1_000_000.0); // global

    // Gemma 4 stores norm weights as full multiplier (no +1 offset, unlike Gemma 2/3)
    assert_eq!(arch.norm_weight_offset(), 0.0);
    assert_eq!(arch.embed_scale(), (5376.0f32).sqrt());
    assert!(arch.has_post_norms());
    assert!(arch.attn_q_norm_key(0).is_some());
    assert_eq!(arch.final_logit_softcapping(), Some(30.0));

    // Layer scalar key
    assert_eq!(
        arch.layer_scalar_key(5),
        Some("layers.5.layer_scalar".to_string())
    );

    // Gemma 4 uses QK-norm, so attention scale is 1.0 (no 1/sqrt(head_dim))
    assert_eq!(arch.attention_scale_for_layer(0), 1.0);
    assert_eq!(arch.attention_scale_for_layer(5), 1.0);

    // K=V flag parsed — v_shares_k() exposes it via the trait.
    // On 31B, attention_k_eq_v=true applies only to global (full_attention) layers;
    // sliding layers still ship v_proj in safetensors.
    assert!(arch.config().attention_k_eq_v);
    assert!(!arch.v_shares_k(0)); // sliding
    assert!(arch.v_shares_k(5)); // global

    // V-norm (parameter-free RMSNorm on V states)
    assert!(arch.has_v_norm());

    // 31B has no KV sharing (num_kv_shared_layers absent)
    assert!(arch.kv_shared_source_layer(0).is_none());
    assert!(arch.kv_shared_source_layer(30).is_none());

    // 31B has no PLE
    assert!(!arch.has_per_layer_embeddings());
}

#[test]
fn test_detect_gemma4_e2b() {
    // Real E2B config with PLE, KV sharing, global_head_dim, layer_types
    let config = serde_json::json!({
        "model_type": "gemma4",
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 1536,
            "intermediate_size": 6144,
            "num_hidden_layers": 35,
            "num_attention_heads": 8,
            "num_key_value_heads": 1,
            "head_dim": 256,
            "global_head_dim": 512,
            "vocab_size": 262144,
            "sliding_window": 512,
            "final_logit_softcapping": 30.0,
            "hidden_size_per_layer_input": 256,
            "num_kv_shared_layers": 20,
            "attention_k_eq_v": false,
            "use_double_wide_mlp": true,
            "rope_parameters": {
                "full_attention": {
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 1000000.0,
                    "rope_type": "proportional"
                },
                "sliding_attention": {
                    "rope_theta": 10000.0,
                    "rope_type": "default"
                }
            },
            "layer_types": [
                "sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "full_attention",
                "sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "full_attention",
                "sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "full_attention",
                "sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "full_attention",
                "sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "full_attention",
                "sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "full_attention",
                "sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "full_attention"
            ]
        }
    });

    let arch = detect_from_json(&config);
    assert_eq!(arch.family(), "gemma4");
    assert_eq!(arch.config().num_layers, 35);

    // Layer types from explicit array
    assert!(arch.is_sliding_window_layer(0));
    assert!(arch.is_sliding_window_layer(3));
    assert!(!arch.is_sliding_window_layer(4)); // global
    assert!(arch.is_sliding_window_layer(5));
    assert!(!arch.is_sliding_window_layer(9)); // global

    // Per-layer head_dim: sliding=256, global=512
    assert_eq!(arch.head_dim_for_layer(0), 256);
    assert_eq!(arch.head_dim_for_layer(4), 512);
    assert_eq!(arch.num_q_heads_for_layer(0), 8);
    assert_eq!(arch.num_q_heads_for_layer(4), 8); // constant across layers

    // Partial rotary on global layers
    assert_eq!(arch.rotary_fraction_for_layer(0), 1.0);
    assert_eq!(arch.rotary_fraction_for_layer(4), 0.25);

    // RoPE bases from rope_parameters
    assert_eq!(arch.rope_base_for_layer(0), 10_000.0);
    assert_eq!(arch.rope_base_for_layer(4), 1_000_000.0);

    // PLE (Per-Layer Embeddings)
    assert!(arch.has_per_layer_embeddings());
    assert_eq!(arch.per_layer_embed_dim(), 256);

    // KV sharing: layers 15-34 share from source layers
    // First 15 layers are non-shared
    assert!(arch.kv_shared_source_layer(0).is_none());
    assert!(arch.kv_shared_source_layer(14).is_none());
    // Layers 15+ are shared: sliding→L13, global→L14
    assert_eq!(arch.kv_shared_source_layer(15), Some(13)); // sliding shared
    assert_eq!(arch.kv_shared_source_layer(19), Some(14)); // global shared
    assert_eq!(arch.kv_shared_source_layer(34), Some(14)); // last layer (global)

    // V-norm, attention scale
    assert!(arch.has_v_norm());
    assert_eq!(arch.attention_scale(), 1.0);
    assert_eq!(arch.norm_weight_offset(), 0.0);

    // No K=V on E2B
    assert!(!arch.config().attention_k_eq_v);
    assert!(!arch.v_shares_k(0));
}

#[test]
fn test_detect_gemma4_real_config() {
    // Test against the actual HuggingFace config.json if available
    let config_path = std::env::var("HOME").ok().map(|h| {
        std::path::PathBuf::from(h).join(".cache/huggingface/hub/models--google--gemma-4-31B-it")
    });
    let config_path = match config_path {
        Some(p) if p.exists() => {
            // Find the snapshot
            let snapshots = p.join("snapshots");
            std::fs::read_dir(&snapshots)
                .ok()
                .and_then(|mut entries| entries.next())
                .and_then(|e| e.ok())
                .map(|e| e.path().join("config.json"))
        }
        _ => None,
    };
    let config_path = match config_path {
        Some(p) if p.exists() => p,
        _ => return, // skip if model not cached
    };

    let text = std::fs::read_to_string(&config_path).unwrap();
    let config: serde_json::Value = serde_json::from_str(&text).unwrap();
    let arch = detect_from_json(&config);

    assert_eq!(arch.family(), "gemma4");
    assert_eq!(arch.config().num_layers, 60);
    assert_eq!(arch.config().hidden_size, 5376);
    assert_eq!(arch.config().head_dim, 256);
    assert_eq!(arch.config().global_head_dim, Some(512));
    assert_eq!(arch.config().num_kv_heads, 16);
    assert_eq!(arch.config().num_global_kv_heads, Some(4));
    assert_eq!(arch.config().partial_rotary_factor, Some(0.25));
    assert!(arch.config().attention_k_eq_v);

    // Verify layer_types parsed correctly (60 layers: 50 sliding + 10 full)
    assert!(arch.config().layer_types.is_some());
    let types = arch.config().layer_types.as_ref().unwrap();
    assert_eq!(types.len(), 60);
    let full_count = types.iter().filter(|t| *t == "full_attention").count();
    assert_eq!(full_count, 10);

    // Layer 5 is full_attention in the real config
    assert!(!arch.is_sliding_window_layer(5));
    assert_eq!(arch.head_dim_for_layer(5), 512);
    assert_eq!(arch.num_kv_heads_for_layer(5), 4);
    assert_eq!(arch.rotary_fraction_for_layer(5), 0.25);

    // RoPE bases from rope_parameters
    assert_eq!(arch.rope_base_for_layer(0), 10_000.0);
    assert_eq!(arch.rope_base_for_layer(5), 1_000_000.0);
}

#[test]
fn test_detect_gemma4_26b_a4b() {
    // Gemma 4 26B A4B — hybrid dense-MLP + MoE per layer.
    // Architecture: 30 layers, hidden=2816, dense_intermediate=9216,
    // 128 experts each with moe_intermediate=704, top_k=8.
    let config = serde_json::json!({
        "model_type": "gemma4",
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 2816,
            "intermediate_size": 9216,
            "num_hidden_layers": 30,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 256,
            "global_head_dim": 512,
            "num_global_key_value_heads": 4,
            "vocab_size": 262144,
            "enable_moe_block": true,
            "num_experts": 128,
            "top_k_experts": 8,
            "moe_intermediate_size": 704,
            "final_logit_softcapping": 30.0,
            "rope_parameters": {
                "full_attention": {
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 1000000.0
                },
                "sliding_attention": {
                    "rope_theta": 10000.0
                }
            }
        }
    });

    let arch = detect_from_json(&config);
    assert_eq!(arch.family(), "gemma4");
    assert_eq!(arch.config().num_layers, 30);
    assert_eq!(arch.config().hidden_size, 2816);
    assert_eq!(arch.config().intermediate_size, 9216);

    // MoE
    assert!(arch.is_moe());
    assert!(arch.is_hybrid_moe());
    assert_eq!(arch.num_experts(), 128);
    assert_eq!(arch.num_experts_per_token(), 8);
    assert_eq!(arch.moe_intermediate_size(), 704);

    // Router keys
    assert_eq!(
        arch.moe_router_key(0),
        Some("layers.0.router.proj.weight".to_string())
    );
    assert_eq!(
        arch.moe_router_scale_key(3),
        Some("layers.3.router.scale".to_string())
    );
    assert_eq!(
        arch.moe_router_per_expert_scale_key(3),
        Some("layers.3.router.per_expert_scale".to_string())
    );

    // Packed expert keys
    assert_eq!(
        arch.packed_experts_gate_up_key(5),
        Some("layers.5.experts.gate_up_proj".to_string())
    );
    assert_eq!(
        arch.packed_experts_down_key(5),
        Some("layers.5.experts.down_proj".to_string())
    );

    // Hybrid MoE norm keys — dense branch gets _1 suffix
    assert_eq!(
        arch.post_feedforward_layernorm_key(0),
        Some("layers.0.post_feedforward_layernorm_1.weight".to_string())
    );
    assert_eq!(
        arch.moe_pre_experts_norm_key(0),
        Some("layers.0.pre_feedforward_layernorm_2.weight".to_string())
    );
    assert_eq!(
        arch.moe_post_experts_norm_key(0),
        Some("layers.0.post_feedforward_layernorm_2.weight".to_string())
    );

    // Dense FFN keys still present (both branches coexist)
    assert_eq!(arch.ffn_gate_key(0), "layers.0.mlp.gate_proj.weight");
    assert_eq!(arch.ffn_up_key(0), "layers.0.mlp.up_proj.weight");
    assert_eq!(arch.ffn_down_key(0), "layers.0.mlp.down_proj.weight");

    // ExpertFormat
    use crate::config::ExpertFormat;
    assert_eq!(arch.expert_format(), ExpertFormat::PackedBF16);

    // Gemma 4 features still work
    assert_eq!(arch.norm_weight_offset(), 0.0);
    assert!(arch.has_v_norm());
    assert!(arch.has_post_norms());
    assert_eq!(arch.bos_token_id(), Some(2));
}

#[test]
fn test_empty_config_has_zero_topology_not_a_silent_default() {
    // `detect_from_json` is infallible to keep in-memory test ergonomics
    // simple, but it must NOT invent topology values. A guess-default
    // like 32/2048/8192 would let an empty config impersonate a Llama-7B
    // shape and propagate that lie into matmul, where it would surface
    // as a broadcast panic (issue #22). The contract is: unset fields
    // round-trip as 0, and the validator catches them.
    let config = serde_json::json!({});
    let arch = detect_from_json(&config);
    assert_eq!(arch.family(), "generic");
    assert_eq!(arch.config().num_layers, 0);
    assert_eq!(arch.config().hidden_size, 0);
    assert_eq!(arch.config().intermediate_size, 0);
}

// ── Disk-path tests: require_config_fields + missing config.json ──

fn write_config_json(dir: &std::path::Path, body: &serde_json::Value) {
    std::fs::write(
        dir.join(CONFIG_FILE_NAME),
        serde_json::to_string(body).unwrap(),
    )
    .unwrap();
}

fn expect_detect_err(model_dir: &std::path::Path) -> ModelError {
    // `Box<dyn ModelArchitecture>` isn't Debug, so `Result::expect_err`
    // doesn't apply. Match instead.
    match detect_architecture(model_dir) {
        Ok(_) => panic!("expected detect_architecture to fail"),
        Err(e) => e,
    }
}

#[test]
fn detect_architecture_errors_when_config_json_is_missing() {
    let tmp = tempfile::tempdir().unwrap();
    // No config.json at all — the failure mode reported in issue #22
    // (user pointed extract-index at a directory containing only
    // safetensors + tokenizer.json).
    let err = expect_detect_err(tmp.path());
    match err {
        ModelError::ConfigMissing(p) => {
            assert_eq!(p, tmp.path().join(CONFIG_FILE_NAME));
        }
        other => panic!("expected ConfigMissing, got {other:?}"),
    }
}

#[test]
fn detect_architecture_errors_when_required_fields_are_missing() {
    let tmp = tempfile::tempdir().unwrap();
    // config.json exists but is empty — previously the silent
    // `unwrap_or(2048)` / `unwrap_or(32)` defaults made this look like
    // a 32-layer 2048-hidden model and panicked on broadcast against
    // the real embed shape (issue #22).
    write_config_json(tmp.path(), &serde_json::json!({}));
    let err = expect_detect_err(tmp.path());
    match err {
        ModelError::ConfigFieldsMissing { path, missing } => {
            assert_eq!(path, tmp.path().join(CONFIG_FILE_NAME));
            // Every required field should be reported as missing, in
            // declared order, so the user sees the full set to fix.
            assert_eq!(missing, REQUIRED_CONFIG_FIELDS.to_vec());
        }
        other => panic!("expected ConfigFieldsMissing, got {other:?}"),
    }
}

#[test]
fn detect_architecture_reports_only_the_missing_required_fields() {
    let tmp = tempfile::tempdir().unwrap();
    // Two of three required fields present — only the one absent
    // should be reported, so the user can fix one entry at a time.
    write_config_json(
        tmp.path(),
        &serde_json::json!({
            "model_type": "llama",
            CONFIG_KEY_HIDDEN_SIZE: 4096,
            CONFIG_KEY_INTERMEDIATE_SIZE: 11008,
        }),
    );
    let err = expect_detect_err(tmp.path());
    match err {
        ModelError::ConfigFieldsMissing { missing, .. } => {
            assert_eq!(missing, vec![CONFIG_KEY_NUM_HIDDEN_LAYERS]);
        }
        other => panic!("expected ConfigFieldsMissing, got {other:?}"),
    }
}

#[test]
fn detect_architecture_accepts_nested_text_config() {
    let tmp = tempfile::tempdir().unwrap();
    // Multimodal layout (Gemma 3 IT): required fields live under
    // `text_config`. Must not be reported as missing.
    write_config_json(
        tmp.path(),
        &serde_json::json!({
            "model_type": "gemma3",
            CONFIG_KEY_TEXT_CONFIG: {
                "model_type": "gemma3_text",
                CONFIG_KEY_HIDDEN_SIZE: 2560,
                CONFIG_KEY_NUM_HIDDEN_LAYERS: 34,
                CONFIG_KEY_INTERMEDIATE_SIZE: 10240,
            }
        }),
    );
    let arch = detect_architecture(tmp.path()).expect("nested text_config must resolve");
    assert_eq!(arch.config().hidden_size, 2560);
    assert_eq!(arch.config().num_layers, 34);
    assert_eq!(arch.config().intermediate_size, 10240);
}

#[test]
fn detect_architecture_accepts_flat_config() {
    let tmp = tempfile::tempdir().unwrap();
    // Text-only model with required fields at the top level (no
    // text_config wrapper). Must also be accepted.
    write_config_json(
        tmp.path(),
        &serde_json::json!({
            "model_type": "llama",
            CONFIG_KEY_HIDDEN_SIZE: 4096,
            CONFIG_KEY_NUM_HIDDEN_LAYERS: 32,
            CONFIG_KEY_INTERMEDIATE_SIZE: 11008,
            "num_attention_heads": 32,
        }),
    );
    let arch = detect_architecture(tmp.path()).expect("flat config must resolve");
    assert_eq!(arch.config().hidden_size, 4096);
    assert_eq!(arch.config().num_layers, 32);
    assert_eq!(arch.config().intermediate_size, 11008);
}

#[test]
fn detect_architecture_falls_back_to_top_level_when_text_config_omits_field() {
    let tmp = tempfile::tempdir().unwrap();
    // Mixed layout: `text_config` carries some required fields, the
    // rest sit at the top level. The presence check accepts either
    // location so users assembling configs by hand aren't tripped up.
    write_config_json(
        tmp.path(),
        &serde_json::json!({
            "model_type": "gemma3",
            CONFIG_KEY_INTERMEDIATE_SIZE: 10240,
            CONFIG_KEY_TEXT_CONFIG: {
                "model_type": "gemma3_text",
                CONFIG_KEY_HIDDEN_SIZE: 2560,
                CONFIG_KEY_NUM_HIDDEN_LAYERS: 34,
            }
        }),
    );
    let arch = detect_architecture(tmp.path()).expect("mixed layout must resolve required fields");
    assert_eq!(arch.config().hidden_size, 2560);
    assert_eq!(arch.config().num_layers, 34);
    assert_eq!(arch.config().intermediate_size, 10240);
}

#[test]
fn detect_architecture_validated_propagates_missing_config_error() {
    // The validated entrypoint is what the streaming extractor calls
    // (`build_streaming_index` in larql-vindex). It must surface the
    // same clean error rather than panic deeper down.
    let tmp = tempfile::tempdir().unwrap();
    let err = match detect_architecture_validated(tmp.path()) {
        Ok(_) => panic!("expected validated detect to fail"),
        Err(e) => e,
    };
    assert!(matches!(err, ModelError::ConfigMissing(_)));
}

#[test]
fn test_detect_deepseek_v4() {
    // DeepSeek-V4 detection routes via the explicit `model_type ==
    // "deepseek_v4"` arm in detect.rs (added in PR #76). Distinct from
    // V3 in tensor naming: no `model.` prefix, `attn`/`ffn` instead of
    // `self_attn`/`mlp`, and `w1`/`w2`/`w3` for expert weights.
    let config = serde_json::json!({
        "model_type": "deepseek_v4",
        "hidden_size": 4096,
        "intermediate_size": 16384,
        "num_hidden_layers": 43,
        "num_attention_heads": 64,
        "num_key_value_heads": 64,
        "head_dim": 128,
        "n_routed_experts": 256,
        "num_experts_per_tok": 8,
        "n_shared_experts": 1,
        "kv_lora_rank": 1024,
        "q_lora_rank": 1024,
    });

    let arch = detect_from_json(&config);

    // ── family / config ───────────────────────────────────────────
    assert_eq!(arch.family(), "deepseek_v4");
    assert_eq!(arch.config().hidden_size, 4096);

    // ── prefix stripping ──────────────────────────────────────────
    // V4 has no `model.` wrapper.
    assert!(arch.key_prefixes_to_strip().is_empty());

    // ── single-tensor keys (embed / norm) ─────────────────────────
    assert_eq!(arch.embed_key(), "embed.weight");
    assert_eq!(arch.final_norm_key(), "norm.weight");

    // ── attention keys (V4 uses `attn`, not `self_attn`) ──────────
    assert_eq!(arch.attn_q_key(7), "layers.7.attn.q_proj.weight");
    assert_eq!(arch.attn_k_key(7), "layers.7.attn.k_proj.weight");
    assert_eq!(arch.attn_v_key(7), "layers.7.attn.v_proj.weight");
    assert_eq!(arch.attn_o_key(7), "layers.7.attn.o_proj.weight");

    // ── layer-norm keys (V4 uses `attn_norm` / `ffn_norm`) ────────
    assert_eq!(arch.input_layernorm_key(3), "layers.3.attn_norm.weight");
    assert_eq!(
        arch.post_attention_layernorm_key(3),
        "layers.3.ffn_norm.weight"
    );
    assert_eq!(arch.pre_feedforward_layernorm_key(0), None);
    assert_eq!(arch.post_feedforward_layernorm_key(0), None);

    // ── dense FFN keys (V4 uses `ffn.w1/w2/w3`) ───────────────────
    assert_eq!(arch.ffn_gate_key(2), "layers.2.ffn.w1.weight");
    assert_eq!(arch.ffn_up_key(2), "layers.2.ffn.w3.weight");
    assert_eq!(arch.ffn_down_key(2), "layers.2.ffn.w2.weight");

    // ── MoE ───────────────────────────────────────────────────────
    assert!(arch.is_moe());
    assert_eq!(arch.num_experts(), 256);
    assert_eq!(arch.num_experts_per_token(), 8);
    assert_eq!(arch.num_shared_experts(), 1);
    assert_eq!(
        arch.moe_router_key(0),
        Some("layers.0.ffn.gate.weight".to_string())
    );

    // Expert weights (per-expert, w1/w2/w3 naming).
    assert_eq!(
        arch.expert_ffn_gate_key(5, 12),
        Some("layers.5.ffn.experts.12.w1.weight".to_string())
    );
    assert_eq!(
        arch.expert_ffn_up_key(5, 12),
        Some("layers.5.ffn.experts.12.w3.weight".to_string())
    );
    assert_eq!(
        arch.expert_ffn_down_key(5, 12),
        Some("layers.5.ffn.experts.12.w2.weight".to_string())
    );

    // Shared experts.
    assert_eq!(
        arch.shared_expert_gate_key(0),
        Some("layers.0.ffn.shared_experts.w1.weight".to_string())
    );
    assert_eq!(
        arch.shared_expert_up_key(0),
        Some("layers.0.ffn.shared_experts.w3.weight".to_string())
    );
    assert_eq!(
        arch.shared_expert_down_key(0),
        Some("layers.0.ffn.shared_experts.w2.weight".to_string())
    );

    // ── MLA (V4 retains MLA shape; tensor names differ) ───────────
    assert!(arch.uses_mla());
    assert_eq!(arch.kv_lora_rank(), 1024);
    assert_eq!(arch.q_lora_rank(), 1024);
    assert_eq!(
        arch.mla_kv_a_key(11),
        Some("layers.11.attn.wkv.weight".to_string())
    );
    // V4 fuses kv into wkv — no separate kv_b projection.
    assert_eq!(arch.mla_kv_b_key(11), None);
    assert_eq!(
        arch.mla_q_a_key(11),
        Some("layers.11.attn.wq_a.weight".to_string())
    );
    assert_eq!(
        arch.mla_q_b_key(11),
        Some("layers.11.attn.wq_b.weight".to_string())
    );
}

#[test]
fn test_detect_deepseek_v4_defaults_when_optional_fields_missing() {
    // V4's MoE / MLA defaults fire when the upstream config omits the
    // expert-count / lora-rank fields. Pin those defaults so accidental
    // changes break this test rather than silently shifting model
    // behaviour.
    let config = serde_json::json!({
        "model_type": "deepseek_v4",
        "hidden_size": 4096,
        "intermediate_size": 16384,
        "num_hidden_layers": 43,
    });

    let arch = detect_from_json(&config);
    assert_eq!(arch.family(), "deepseek_v4");

    // No expert count → is_moe() returns false (defaults to 0 experts).
    assert!(!arch.is_moe());
    // num_experts() falls back to 256 (V4-Flash default).
    assert_eq!(arch.num_experts(), 256);
    // num_experts_per_token() falls back to 6.
    assert_eq!(arch.num_experts_per_token(), 6);
    // num_shared_experts() falls back to 1.
    assert_eq!(arch.num_shared_experts(), 1);

    // No kv_lora_rank / q_lora_rank → uses_mla() returns false.
    assert!(!arch.uses_mla());
    // Defaults still pin to 1024 even when MLA is off (callers may read
    // them for arch-comparison purposes).
    assert_eq!(arch.kv_lora_rank(), 1024);
    assert_eq!(arch.q_lora_rank(), 1024);
}
