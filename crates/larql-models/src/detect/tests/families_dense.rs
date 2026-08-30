//! Per-family detection over dense architectures — hand-written configs
//! per family, plus the generic fallback and empty-config contracts.

use crate::detect::*;

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
    assert_eq!(arch.embed_scale(), Some((2560.0f32).sqrt()));
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
    // No embedding-scale operation declared — `None`, not `Some(1.0)`.
    assert_eq!(arch.embed_scale(), None);
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
    assert_eq!(arch.embed_scale(), Some((512.0_f32).sqrt()));
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
fn test_detect_bitnet_is_explicit_not_generic() {
    // BitNet must be recognised explicitly — never silently collapse to the
    // generic fallback (which would mask wrong/default config behind a
    // "generic" label). Both the HF and GGUF-derived model_type spellings.
    for model_type in ["bitnet", "bitnet-b1.58", "bitnet_b1_58"] {
        let config = serde_json::json!({
            "model_type": model_type,
            "hidden_size": 2560,
            "num_hidden_layers": 30,
            "rms_norm_eps": 1e-5
        });
        let arch = detect_from_json(&config);
        assert_eq!(arch.family(), "bitnet", "model_type={model_type}");
        // Epsilon is honoured from config, not hardcoded.
        assert!(
            (arch.norm_eps() - 1e-5).abs() < 1e-9,
            "model_type={model_type}"
        );
    }
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
